#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-hop retrieval benchmark for DCEE (and exact inner-product oracle).

Motivation
----------
Single-vector ANN answers “nearest neighbors to this query”. Multi-hop retrieval
answers “what nodes appear after repeatedly retrieving neighbors-of-neighbors?”
Common in iterative RAG / graph expansion where the answer embedding is not a
direct neighbor of the query embedding.

This script does **not** implement a graph DB; it simulates expansion by taking
top-K cosine neighbors under DCEE (or exact IP on L2-normalized rows), round by
round, like beam expansion over the embedding space – similar in spirit to how
multi-hop RAG benchmarks (e.g. HotpotQA / 2WikiMultiHopQA / MultiHop-RAG) track
whether gold passages become reachable within a few hops of retrieval.

Synthetic chains
------------------
Each trial builds an indexed corpus containing:

- A chain ``c0 → c1 → … → c_{L-1}`` (``c_{L-1}`` is the **target**) where
  consecutive nodes have high cosine similarity but the query aligns mostly
  with ``e0`` while the target lives mostly in ``e_{L-1}`` (orthogonal subspaces).
- Strong distractors near the query (push target out of hop-1 top-K).
- Weaker distractors + random background.

Metrics
-------
For each chain length and method (DCEE vs **Exact** top-K):

- **Success@≤D**: target found within ``D`` expansion rounds (default D = chain length).
- **Mean hops when success**: average discovered depth (Exact uses minimal depth).

Examples
--------
  python benchmark_multihop_retrieval_dcee.py --trials 80 --beam-k 12

  python benchmark_multihop_retrieval_dcee.py --chain-lens 3 4 5 --trials 200 --quiet-dcee

Interpretation for stakeholders
-------------------------------
Report the table rows: “Under this expansion policy (beam K, max depth), DCEE
recovers the chain end at X% vs exact Y%.” If X lags Y, raise ``beam-k`` or
use ``--tuned`` / higher ``n_probe`` (see DCEEConfig).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcee import DCEEConfig, DCEEEngine


def _normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32, copy=False)


def _orthonormal_noise(
    rng: np.random.Generator,
    dim: int,
    n: int,
    exclude_subspace: np.ndarray | None = None,
) -> np.ndarray:
    """n unit vectors; optional (m,dim) exclude_subspace — noise orthogonal to those rows."""
    m = np.zeros((0, dim), dtype=np.float32) if exclude_subspace is None else exclude_subspace.astype(np.float32, copy=False)
    out: list[np.ndarray] = []
    for _ in range(n):
        x = rng.standard_normal(dim).astype(np.float32)
        if m.shape[0]:
            x = x - m.T @ (m @ x)
        x = _normalize(x)
        out.append(x)
        m = np.vstack([m, x[None, :]])
    return np.stack(out, axis=0)


def exact_topk_from_query(emb: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    s = emb @ q.astype(np.float64)
    if k >= len(s):
        return np.argsort(-s).astype(np.int64)
    part = np.argpartition(-s, k - 1)[:k]
    return part[np.argsort(-s[part])].astype(np.int64)


def _sample_chain_corpus(
    rng: np.random.Generator,
    *,
    chain_len: int,
    dim: int,
    e_rows: list[np.ndarray],
    E: np.ndarray,
    n_strong_distractors: int,
    n_weak_distractors: int,
    n_background: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    noise_scale = 0.04
    chain_vecs: list[np.ndarray] = []
    n0 = _orthonormal_noise(rng, dim, 1, exclude_subspace=E)[0]
    chain_vecs.append(_normalize(0.88 * e_rows[0] + 0.42 * e_rows[1] + noise_scale * n0))
    for i in range(1, chain_len - 1):
        ni = _orthonormal_noise(rng, dim, 1, exclude_subspace=E)[0]
        chain_vecs.append(_normalize(0.52 * e_rows[i] + 0.78 * e_rows[i + 1] + noise_scale * ni))
    nL = _orthonormal_noise(rng, dim, 1, exclude_subspace=E)[0]
    chain_vecs.append(
        _normalize(
            0.86 * e_rows[chain_len - 1]
            + 0.42 * e_rows[chain_len - 2]
            + noise_scale * nL
        )
    )
    chain = np.stack(chain_vecs, axis=0)

    distractors: list[np.ndarray] = []
    for _ in range(n_strong_distractors):
        nu = _orthonormal_noise(rng, dim, 1, exclude_subspace=E)[0]
        u = np.zeros(dim, dtype=np.float32)
        u[0] = 1.0
        distractors.append(_normalize(0.93 * u + 0.05 * nu - 0.08 * e_rows[-1]))
    for _ in range(n_weak_distractors):
        nu = _orthonormal_noise(rng, dim, 1, exclude_subspace=E)[0]
        u = np.zeros(dim, dtype=np.float32)
        u[0] = 1.0
        distractors.append(_normalize(0.62 * u + 0.78 * nu - 0.12 * e_rows[-1]))
    distr_mat = np.asarray(distractors, dtype=np.float32)

    bg = rng.standard_normal((n_background, dim)).astype(np.float32)
    bg = np.stack([_normalize(bg[i]) for i in range(len(bg))], axis=0)
    return chain, distr_mat, bg


def build_chain_trial(
    rng: np.random.Generator,
    *,
    chain_len: int,
    dim: int,
    n_strong_distractors: int,
    n_weak_distractors: int,
    n_background: int,
    beam_k: int,
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """
    Returns (emb, query, target_idx, chain_indices) where chain_indices = [c0..c_{L-1}].
    """
    if chain_len < 2:
        raise ValueError("chain_len must be >= 2")
    if dim < chain_len + 8:
        raise ValueError("dim too small for chain_len")

    eye = np.eye(dim, dtype=np.float32)
    e_rows = [eye[i].copy() for i in range(chain_len)]
    E = np.stack(e_rows, axis=0)
    query = _normalize(e_rows[0])
    target_idx = chain_len - 1

    # We *prefer* cases where the target is not in hop-1, but do not hard-fail
    # when beam_k is large and the structure makes that unlikely. This keeps
    # runs stable for wide beams while still producing many genuinely multi-hop
    # trials in aggregate.
    best_emb: np.ndarray | None = None
    best_hop1_has_target = True
    for _ in range(32):
        chain, distr_mat, bg = _sample_chain_corpus(
            rng,
            chain_len=chain_len,
            dim=dim,
            e_rows=e_rows,
            E=E,
            n_strong_distractors=n_strong_distractors,
            n_weak_distractors=n_weak_distractors,
            n_background=n_background,
        )
        emb = np.vstack([chain, distr_mat, bg]).astype(np.float32, copy=False)
        hop1 = exact_topk_from_query(emb, query, beam_k)
        has_target = int(target_idx) in hop1.tolist()
        if not has_target:
            return emb, query, target_idx, list(range(chain_len))
        if best_emb is None:
            best_emb = emb
        rng = np.random.default_rng(int(rng.integers(1, 2**31 - 1)))

    # Fall back to a corpus where the target is sometimes in hop-1; these
    # chains are still useful, and Exact% will reflect how often multi-hop is
    # actually required versus trivial 1-hop.
    assert best_emb is not None
    return best_emb, query, target_idx, list(range(chain_len))


def multihop_depth_exact(
    emb: np.ndarray,
    q: np.ndarray,
    target_idx: int,
    *,
    beam_k: int,
    max_depth: int,
) -> int | None:
    """Return 1-based depth when target appears, else None."""
    seen: set[int] = set()
    frontier = set(exact_topk_from_query(emb, q, beam_k).tolist())
    seen |= frontier
    if target_idx in frontier:
        return 1
    for d in range(2, max_depth + 1):
        nxt: set[int] = set()
        for i in frontier:
            nxt.update(exact_topk_from_query(emb, emb[i], beam_k).tolist())
        nxt -= seen
        if target_idx in nxt:
            return d
        if not nxt:
            return None
        seen |= nxt
        frontier = nxt
    return None


def multihop_depth_dcee(
    engine: DCEEEngine,
    emb: np.ndarray,
    q: np.ndarray,
    target_idx: int,
    *,
    beam_k: int,
    max_depth: int,
) -> int | None:
    seen: set[int] = set()

    def topk(query_vec: np.ndarray) -> list[int]:
        return [int(i) for i, _ in engine.search(query_vec.astype(np.float32, copy=False), top_k=beam_k)]

    frontier = set(topk(q))
    seen |= frontier
    if target_idx in frontier:
        return 1
    for d in range(2, max_depth + 1):
        nxt: set[int] = set()
        for i in frontier:
            nxt.update(topk(emb[i]))
        nxt -= seen
        if target_idx in nxt:
            return d
        if not nxt:
            return None
        seen |= nxt
        frontier = nxt
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark multi-hop retrieval: DCEE vs exact IP expansion",
    )
    p.add_argument("--trials", type=int, default=200, help="Monte Carlo trials per chain length")
    p.add_argument(
        "--chain-lens",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="Chain lengths (number of indexed chain nodes; target is last)",
    )
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--beam-k", type=int, default=32, help="Top-K neighbors per expansion step")
    p.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Cap expansion depth (default: 8 ‒ enough for L up to 5 in the default setting)",
    )
    p.add_argument("--n-strong-dis", type=int, default=4)
    p.add_argument("--n-weak-dis", type=int, default=12)
    p.add_argument("--n-background", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tuned",
        action="store_true",
        help="Use DCEEConfig.tuned_for(N, dim) for each trial index",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Override cluster count (when not using --tuned)",
    )
    p.add_argument("--n-probe", type=int, default=16)
    p.add_argument("--n-probe-max", type=int, default=48)
    p.add_argument("--top-k-refine", type=int, default=64)
    p.add_argument("--quantization", choices=("int8", "float16", "float32"), default="int8")
    p.add_argument("--no-adaptive", action="store_true", help="Disable AMP on DCEE")
    p.add_argument("--quiet-dcee", action="store_true", help="DCEE verbose=False + less logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet_dcee else logging.INFO, format="%(message)s")

    rng_master = np.random.default_rng(args.seed)
    max_depth = args.max_depth or max(args.chain_lens)

    print("\nDCEE multi-hop retrieval benchmark")
    print("  Expansion: union of top-{} neighbors each round (frontier BFS)".format(args.beam_k))
    print("  Max expansion depth (--max-depth): {}".format(max_depth))
    print("  Trials per chain length: {}".format(args.trials))
    print("  dim={}  distractors: strong={} weak={} background={}".format(
        args.dim, args.n_strong_dis, args.n_weak_dis, args.n_background
    ))
    print("  DCEE config: tuned={}  n_probe={}  n_probe_max={}  top_k_refine={}  AMP={}\n".format(
        args.tuned, args.n_probe, args.n_probe_max, args.top_k_refine, not args.no_adaptive,
    ))

    header = (
        f"{'L':>3} {'Exact%':>8} {'DCEE%':>8} "
        f"{'ExHops':>7} {'DcHops':>7} {'DCEE_build_s':>12} {'DCEE_hops_s':>12}"
    )
    print(header)
    print("-" * len(header))

    for L in sorted(set(args.chain_lens)):
        ok_ex = 0
        ok_dc = 0
        hops_ex: list[float] = []
        hops_dc: list[float] = []
        t_build = 0.0
        t_exp = 0.0

        for _ in range(args.trials):
            trial_seed = int(rng_master.integers(0, 2**31 - 1))
            rng_t = np.random.default_rng(trial_seed)

            emb, q, target_idx, _ = build_chain_trial(
                rng_t,
                chain_len=L,
                dim=args.dim,
                n_strong_distractors=args.n_strong_dis,
                n_weak_distractors=args.n_weak_dis,
                n_background=args.n_background,
                beam_k=args.beam_k,
            )
            n, d = emb.shape

            t0 = time.perf_counter()
            if args.tuned:
                cfg = DCEEConfig.tuned_for(n, d)
            else:
                cfg = DCEEConfig(dim=d, n_clusters=args.n_clusters or max(32, min(256, int(np.sqrt(n)))))
                cfg.n_probe = args.n_probe
                cfg.n_probe_max = args.n_probe_max
                cfg.top_k_refine = args.top_k_refine
            cfg.quantization = args.quantization
            cfg.verbose = False
            cfg.adaptive_probe = not args.no_adaptive
            if args.tuned and args.n_clusters is not None:
                cfg.n_clusters = args.n_clusters
            if not args.tuned:
                cfg.n_probe_max = max(cfg.n_probe_max, cfg.n_probe)

            engine = DCEEEngine(cfg)
            engine.build(emb)
            t_build += time.perf_counter() - t0

            dex = multihop_depth_exact(emb, q, target_idx, beam_k=args.beam_k, max_depth=max_depth)
            d0 = time.perf_counter()
            ddc = multihop_depth_dcee(engine, emb, q, target_idx, beam_k=args.beam_k, max_depth=max_depth)
            t_exp += time.perf_counter() - d0

            if dex is not None:
                ok_ex += 1
                hops_ex.append(float(dex))
            if ddc is not None:
                ok_dc += 1
                hops_dc.append(float(ddc))

        tri = max(args.trials, 1)
        mean_ex_h = float(np.mean(hops_ex)) if hops_ex else float("nan")
        mean_dc_h = float(np.mean(hops_dc)) if hops_dc else float("nan")

        print(
            f"{L:3d} {100 * ok_ex / tri:8.1f} {100 * ok_dc / tri:8.1f} "
            f"{mean_ex_h:7.2f} {mean_dc_h:7.2f} {t_build / tri:12.4f} {t_exp / tri:12.4f}"
        )

    print(
        "\nNotes:\n"
        "  • 'Exact%' is the oracle (NumPy IP) under the same hop policy; should be ~100% if --max-depth is large enough.\n"
        "  • If DCEE% < Exact%, increase --beam-k, --top-k-refine, or n_probe / use --tuned.\n"
        "  • This simulates iterative neighbor expansion in embedding space, not a separate knowledge graph edge list.\n"
    )


if __name__ == "__main__":
    main()
