#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random-search hyperparameter tuning for DCEE (maximize recall vs latency tradeoff).

Uses the same recall definition as benchmark_dcee.py (overlap with exact top-K inner product).

Examples
--------
  python tune_dcee.py --n 20000 --trials 40 --top-k 5

  python tune_dcee.py --embeddings data/embeddings.npy --trials 60 \\
      --latency-penalty 0.015 --out best_params.json

Embeddings file: ``.npy`` float32 array shape (N, dim), row-normalized recommended.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcee import DCEEConfig, DCEEEngine
from test_realworld_dcee import make_synthetic_embeddings


def exact_topk_ids(emb: np.ndarray, q_indices: np.ndarray, k: int) -> np.ndarray:
    q = emb[q_indices].astype(np.float32, copy=False)
    scores = q @ emb.T
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    nq = scores.shape[0]
    out = np.empty((nq, k), dtype=np.int64)
    for i in range(nq):
        cols = part[i]
        out[i] = cols[np.argsort(-scores[i, cols])]
    return out


def mean_recall_at_k(pred: np.ndarray, truth: np.ndarray) -> float:
    n, k = truth.shape
    hits = 0
    for i in range(n):
        hits += len(np.intersect1d(pred[i], truth[i], assume_unique=True))
    return hits / (n * k)


def evaluate_config(
    cfg: DCEEConfig,
    emb: np.ndarray,
    q_indices: np.ndarray,
    gt: np.ndarray,
    warmup: int,
) -> Tuple[float, float, float, float]:
    """Returns recall, p50_ms, p95_ms, build_s."""
    engine = DCEEEngine(cfg)
    t0 = time.time()
    engine.build(emb)
    build_s = time.time() - t0

    k = gt.shape[1]
    for w in range(warmup):
        j = w % len(q_indices)
        engine.search(emb[int(q_indices[j])], top_k=k)

    lat: List[float] = []
    pred = np.empty_like(gt)
    for j in range(len(q_indices)):
        t0 = time.perf_counter()
        res = engine.search(emb[int(q_indices[j])], top_k=k)
        lat.append((time.perf_counter() - t0) * 1000)
        pred[j] = [int(r[0]) for r in res]

    rec = mean_recall_at_k(pred, gt)
    p50 = float(np.percentile(lat, 50))
    p95 = float(np.percentile(lat, 95))
    return rec, p50, p95, build_s


def sample_config(rng: np.random.Generator, dim: int, n_vectors: int) -> DCEEConfig:
    """Sample one plausible hyperparameter set for dataset scale."""
    nc_lo = max(8, int(np.sqrt(n_vectors) * 0.5))
    nc_hi = max(nc_lo + 1, min(512, int(np.sqrt(n_vectors) * 2.5)))
    n_clusters = int(rng.integers(nc_lo, nc_hi + 1))

    n_probe = int(rng.integers(4, min(48, n_clusters) + 1))
    n_probe_max = int(rng.integers(n_probe, n_clusters + 1))

    kfe = int(rng.integers(8, 25))
    refine_lo = max(16, n_vectors // 400)
    refine_hi = max(refine_lo + 1, min(384, max(refine_lo + 2, n_vectors // 50)))
    top_k_refine = int(rng.integers(refine_lo, refine_hi))

    adaptive = rng.random() > 0.15
    margin = float(rng.uniform(0.012, 0.055)) if adaptive else 0.0

    return DCEEConfig(
        dim=dim,
        n_clusters=n_clusters,
        keyframe_every=kfe,
        quantization="int8",
        top_k_refine=top_k_refine,
        n_probe=n_probe,
        n_probe_max=max(n_probe, n_probe_max),
        adaptive_probe=adaptive,
        adaptive_probe_margin=margin,
        batch_size=max(4096, min(8192, n_vectors // 4)),
        verbose=False,
    )


def config_to_dict(cfg: DCEEConfig) -> Dict[str, Any]:
    return {
        "dim": cfg.dim,
        "n_clusters": cfg.n_clusters,
        "keyframe_every": cfg.keyframe_every,
        "quantization": cfg.quantization,
        "top_k_refine": cfg.top_k_refine,
        "n_probe": cfg.n_probe,
        "n_probe_max": cfg.n_probe_max,
        "adaptive_probe": cfg.adaptive_probe,
        "adaptive_probe_margin": cfg.adaptive_probe_margin,
        "batch_size": cfg.batch_size,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DCEE hyperparameter tuning")
    p.add_argument("--n", type=int, default=20_000, help="Synthetic corpus size if no --embeddings")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--n-topics", type=int, default=64)
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--embeddings", type=Path, default=None, help=".npy float32 (N, dim)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-queries", type=int, default=150)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument(
        "--latency-penalty",
        type=float,
        default=0.012,
        help="Objective = recall - penalty * p95_ms (higher penalty favors speed)",
    )
    p.add_argument("--out", type=Path, default=Path("best_dcee_params.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    rng = np.random.default_rng(args.seed)

    if args.embeddings:
        emb = np.load(args.embeddings).astype(np.float32)
    else:
        emb = make_synthetic_embeddings(args.n, args.dim, args.n_topics, args.noise, args.seed)

    n, dim = emb.shape
    q_indices = rng.integers(0, n, size=args.n_queries)
    gt = exact_topk_ids(emb, q_indices, args.top_k)

    best_score = float("-inf")
    best: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    print(f"Tuning on N={n:,} dim={dim}  trials={args.trials}  metric=recall - {args.latency_penalty}*p95_ms\n")

    for t in range(args.trials):
        cfg = sample_config(rng, dim, n)
        try:
            rec, p50, p95, build_s = evaluate_config(cfg, emb, q_indices, gt, args.warmup)
        except Exception as exc:
            print(f"trial {t+1}/{args.trials}: FAIL {exc}")
            continue

        score = rec - args.latency_penalty * p95
        rows.append(
            {
                "trial": t,
                "score": score,
                "recall": rec,
                "p50_ms": p50,
                "p95_ms": p95,
                "build_s": build_s,
                **config_to_dict(cfg),
            }
        )

        tag = "★" if score > best_score else " "
        print(
            f"{tag} trial {t+1:>3}  recall={rec:.3f}  p95={p95:.2f}ms  score={score:.4f}  "
            f"nc={cfg.n_clusters} probe={cfg.n_probe}-{cfg.n_probe_max} kfe={cfg.keyframe_every} refine={cfg.top_k_refine}"
        )

        if score > best_score:
            best_score = score
            best = {
                "objective": score,
                "recall": rec,
                "p95_ms": p95,
                "p50_ms": p50,
                "build_s": build_s,
                "latency_penalty": args.latency_penalty,
                "config": config_to_dict(cfg),
            }

    if not best:
        raise SystemExit("All trials failed.")

    args.out.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"\nBest objective={best_score:.4f}  → wrote {args.out}\n")
    print("Python snippet:")
    c = best["config"]
    print(
        f"""cfg = DCEEConfig(
    dim={c["dim"]},
    n_clusters={c["n_clusters"]},
    keyframe_every={c["keyframe_every"]},
    quantization="{c["quantization"]}",
    top_k_refine={c["top_k_refine"]},
    n_probe={c["n_probe"]},
    n_probe_max={c["n_probe_max"]},
    adaptive_probe={c["adaptive_probe"]},
    adaptive_probe_margin={c["adaptive_probe_margin"]:.6f},
)\n"""
    )


if __name__ == "__main__":
    main()
