#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TurboQuant-style nearest-neighbor benchmark for DCEE.

Goal
----
Mirror the evaluation protocol commonly cited for TurboQuant Section 4.4:
- base vectors (up to ~100k),
- query vectors (up to ~1k),
- cosine / inner-product retrieval on L2-normalized embeddings,
- Recall@10 against exact brute-force neighbors.

This script does NOT implement TurboQuant itself; it benchmarks DCEE under a
comparable protocol so you can report DCEE results side-by-side on the same
embedding dumps (e.g., GloVe d=200, DBpedia d=1536/3072 if you have them).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from dcee import DCEEConfig, DCEEEngine


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _exact_topk_ids(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    scores = queries @ base.T
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for i in range(queries.shape[0]):
        cols = part[i]
        out[i] = cols[np.argsort(-scores[i, cols])]
    return out


def _mean_recall_at_k(pred: np.ndarray, truth: np.ndarray) -> float:
    n, k = truth.shape
    hit = 0
    for i in range(n):
        hit += len(np.intersect1d(pred[i], truth[i], assume_unique=True))
    return hit / (n * k)


def _dcee_estimated_bits_per_vector(cfg: DCEEConfig, dim: int) -> float:
    """
    Rough payload estimate (vector storage only):
    - raw values by quantization
    - keyframe values every keyframe_every rows
    - per-vector scale (float32) for int8 mode
    This mirrors DCEE's storage style directionally, not exact file bytes.
    """
    q_bits = {"int8": 8.0, "float16": 16.0, "float32": 32.0}.get(cfg.quantization, 32.0)
    kfe = max(int(getattr(cfg, "keyframe_every", 16)), 1)
    keyframe_overhead = (32.0 * dim) / float(kfe)
    scale_overhead = 32.0 if cfg.quantization == "int8" else 0.0
    return q_bits * dim + keyframe_overhead + scale_overhead


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TurboQuant-style Recall@10 benchmark for DCEE",
    )
    p.add_argument("--base", type=Path, default=None, help="Path to base vectors .npy (shape: N x D)")
    p.add_argument("--queries", type=Path, default=None, help="Path to query vectors .npy (shape: Q x D)")
    p.add_argument(
        "--glove-dir",
        type=Path,
        default=Path("DATA_GLOVE"),
        help="Directory containing glove.6B.<dim>d.txt (used when --base/--queries are not provided)",
    )
    p.add_argument("--glove-dim", type=int, default=200, choices=(50, 100, 200, 300))
    p.add_argument(
        "--glove-dims",
        type=int,
        nargs="+",
        default=None,
        help="Run multiple GloVe dimensions in one command, e.g. --glove-dims 50 100 200 300",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-base", type=int, default=100_000)
    p.add_argument("--max-queries", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-clusters", type=int, default=None)
    p.add_argument("--n-probe", type=int, default=16)
    p.add_argument("--n-probe-max", type=int, default=48)
    p.add_argument("--top-k-refine", type=int, default=64)
    p.add_argument("--quantization", choices=("int8", "float16", "float32"), default="int8")
    p.add_argument("--keyframe-every", type=int, default=16)
    p.add_argument("--tuned", action="store_true", help="Use DCEEConfig.tuned_for(N, D)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _load_glove_txt(path: Path, max_rows: int) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"GloVe file not found: {path}")
    vecs: list[np.ndarray] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            # format: token val1 val2 ... valD
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            v = np.fromstring(parts[1], sep=" ", dtype=np.float32)
            if v.size > 0:
                vecs.append(v)
    if not vecs:
        raise SystemExit(f"No vectors parsed from {path}")
    m = np.stack(vecs, axis=0).astype(np.float32, copy=False)
    return m


def main() -> None:
    args = parse_args()
    if args.glove_dims is not None:
        bad = [d for d in args.glove_dims if d not in (50, 100, 200, 300)]
        if bad:
            raise SystemExit(f"Unsupported values in --glove-dims: {bad}")
    if args.glove_dims is not None and (args.base is not None or args.queries is not None):
        raise SystemExit("Use either --glove-dims OR --base/--queries, not both.")

    run_dims = args.glove_dims if args.glove_dims is not None else [args.glove_dim]
    rows: list[tuple[int, float, float, float, float, float, float]] = []

    for d_run in run_dims:
        rng = np.random.default_rng(args.seed + int(d_run))

        if args.base is not None and args.queries is not None:
            base = np.load(args.base)
            queries = np.load(args.queries)
            source = f"npy ({args.base.name}, {args.queries.name})"
        else:
            glove_txt = args.glove_dir / f"glove.6B.{d_run}d.txt"
            need_rows = args.max_base + args.max_queries
            m = _load_glove_txt(glove_txt, max_rows=need_rows)
            if len(m) < need_rows:
                raise SystemExit(
                    f"GloVe rows loaded={len(m)}, need at least {need_rows}. "
                    f"Lower --max-base/--max-queries or use a larger source file.",
                )
            idx = rng.choice(len(m), size=need_rows, replace=False)
            m = m[idx]
            base = m[: args.max_base]
            queries = m[args.max_base : args.max_base + args.max_queries]
            source = f"GloVe txt ({glove_txt.name})"

        if base.ndim != 2 or queries.ndim != 2:
            raise SystemExit("Both --base and --queries must be 2-D arrays.")
        if base.shape[1] != queries.shape[1]:
            raise SystemExit(f"Dimension mismatch: base={base.shape}, queries={queries.shape}")

        if args.base is not None and args.queries is not None and len(base) > args.max_base:
            idx = rng.choice(len(base), size=args.max_base, replace=False)
            base = base[idx]
        if args.base is not None and args.queries is not None and len(queries) > args.max_queries:
            idx = rng.choice(len(queries), size=args.max_queries, replace=False)
            queries = queries[idx]

        base = _l2_normalize_rows(base)
        queries = _l2_normalize_rows(queries)
        n, d = base.shape
        k = min(args.top_k, n)

        print("\nTurboQuant-style protocol benchmark (DCEE)")
        print(f"  base vectors: {n:,}")
        print(f"  query vectors: {len(queries):,}")
        print(f"  source: {source}")
        print(f"  dim: {d}")
        print(f"  metric: cosine / inner-product on L2-normalized vectors")
        print(f"  Recall@{k} vs exact brute-force\n")

        t0 = time.perf_counter()
        truth = _exact_topk_ids(base, queries, k)
        t_exact = time.perf_counter() - t0

        if args.tuned:
            cfg = DCEEConfig.tuned_for(n, d)
        else:
            cfg = DCEEConfig(
                dim=d,
                n_clusters=args.n_clusters or max(32, min(512, int(np.sqrt(n)))),
                keyframe_every=args.keyframe_every,
                quantization=args.quantization,
                n_probe=args.n_probe,
                n_probe_max=args.n_probe_max,
                top_k_refine=args.top_k_refine,
            )
        cfg.verbose = not args.quiet
        cfg.quantization = args.quantization

        t1 = time.perf_counter()
        engine = DCEEEngine(cfg)
        engine.build(base)
        t_build = time.perf_counter() - t1

        t2 = time.perf_counter()
        pred = np.empty_like(truth)
        for i, q in enumerate(queries):
            hits = engine.search(q, top_k=k)
            pred[i] = np.array([idx for idx, _ in hits], dtype=np.int64)
        t_query = time.perf_counter() - t2

        recall = _mean_recall_at_k(pred, truth)
        qps = len(queries) / max(t_query, 1e-12)
        p50_ms = (t_query / len(queries)) * 1e3
        bpv = _dcee_estimated_bits_per_vector(cfg, d)
        est_mb = (bpv * n) / 8.0 / 1e6

        print("Results")
        print(f"  exact precompute time: {t_exact:.4f}s")
        print(f"  DCEE build time: {t_build:.4f}s")
        print(f"  DCEE query time ({len(queries)} queries): {t_query:.4f}s")
        print(f"  DCEE QPS: {qps:.1f}")
        print(f"  avg per-query latency: {p50_ms:.3f} ms")
        print(f"  Recall@{k}: {100.0 * recall:.2f}%")
        print(f"  est bits-per-vector (DCEE payload): {bpv:.1f}")
        print(f"  est compressed payload size: {est_mb:.2f} MB")

        rows.append((d, 100.0 * recall, t_build, t_query, qps, bpv, est_mb))

    if len(rows) > 1:
        print("\nSummary across dimensions")
        print(
            f"{'dim':>5} {'Recall@K %':>12} {'build_s':>10} {'query_s':>10} "
            f"{'QPS':>10} {'bits/vec':>10} {'est_MB':>10}"
        )
        print("-" * 78)
        for d, r, b, q, qps, bpv, mb in rows:
            print(f"{d:5d} {r:12.2f} {b:10.4f} {q:10.4f} {qps:10.1f} {bpv:10.1f} {mb:10.2f}")

    print(
        "\nTip: for a closer TurboQuant-style comparison, run this on the same embedding "
        "dumps and sampling protocol used in TurboQuant Section 4.4.",
    )


if __name__ == "__main__":
    main()
