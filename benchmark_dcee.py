#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare DCEE against standard baselines on the same embeddings and queries.

Baselines
---------
- **Exact (NumPy)** — batched matmul + argpartition (reference neighbor sets).
- **FAISS IndexFlatIP** — same math as exact on L2-normalized data (sanity check ~100% recall).
- **FAISS HNSW** — graph ANN (popular approximate index).
- **FAISS IVF-Flat** — inverted-file ANN (requires training; skipped if too small / fails).

Metrics
-------
- **Recall@K**: mean overlap fraction between each method's top-K ids and exact top-K ids.
- **Latency**: p50 / p95 over the same query batch (after warmup).

Examples
--------
  python benchmark_dcee.py --mode synthetic --n 50000 --dim 128 --n-queries 200 --top-k 5

  python benchmark_dcee.py --mode documents --max-docs 2000 --n-queries 100 --top-k 10 \\
      --n-probe 8 --n-clusters 64
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dcee import DCEEConfig, DCEEEngine
from test_realworld_dcee import (
    DEFAULT_PARAGRAPHS,
    _repeat_corpus,
    default_n_clusters,
    embed_texts,
    load_text_lines,
    make_synthetic_embeddings,
)

try:
    import faiss

    _FAISS_OK = True
except ImportError:
    _FAISS_OK = False


def exact_topk_ids(emb: np.ndarray, q_indices: np.ndarray, k: int) -> np.ndarray:
    """Ground truth: top-k inner product neighbors per query row (emb is L2-normalized)."""
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
    """pred, truth: (n_queries, k) int indices."""
    n, k = truth.shape
    hits = 0
    for i in range(n):
        hits += len(np.intersect1d(pred[i], truth[i], assume_unique=True))
    return hits / (n * k)


def dcee_index_bytes(engine: DCEEEngine) -> int:
    cfg = engine.cfg
    bpv = {"float32": 4, "float16": 2, "int8": 1}.get(cfg.quantization, 4)
    n = sum(len(b.global_indices) for b in engine.index.clusters)
    # Same accounting as build print: payload dominated by delta storage
    return int(n * cfg.dim * bpv)


def faiss_index_bytes(index) -> int:
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "idx.faiss")
        faiss.write_index(index, path)
        return os.path.getsize(path)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def build_faiss_flat(emb: np.ndarray) -> "faiss.Index":
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb.astype(np.float32, copy=False))
    return index


def build_faiss_hnsw(
    emb: np.ndarray,
    M: int,
    ef_construction: int,
    ef_search: int,
) -> "faiss.Index":
    d = emb.shape[1]
    index = faiss.index_factory(d, f"HNSW{M}", faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(emb.astype(np.float32, copy=False))
    return index


def build_faiss_ivf(
    emb: np.ndarray,
    nlist: int,
    nprobe: int,
) -> Optional["faiss.Index"]:
    n, d = emb.shape
    if nlist <= 0 or nlist >= n or n < max(nlist, 39):
        return None
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(emb.astype(np.float32, copy=False))
    index.add(emb.astype(np.float32, copy=False))
    index.nprobe = max(1, min(nprobe, nlist))
    return index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DCEE vs FAISS / exact benchmark")
    p.add_argument("--mode", choices=("synthetic", "documents"), default="synthetic")
    p.add_argument("--n", type=int, default=50_000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--n-topics", type=int, default=64)
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--max-docs", type=int, default=2000)
    p.add_argument("--text-file", type=Path, default=None)
    p.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--n-clusters", type=int, default=None)
    p.add_argument("--keyframe-every", type=int, default=16)
    p.add_argument("--quantization", choices=("int8", "float16", "float32"), default="int8")
    p.add_argument("--n-probe", type=int, default=8)
    p.add_argument(
        "--tuned",
        action="store_true",
        help="Use DCEEConfig.tuned_for(N, dim) + AMP (adaptive margin probing) as baseline",
    )
    p.add_argument("--no-adaptive", action="store_true", help="Disable AMP cluster expansion")
    p.add_argument("--n-probe-max", type=int, default=None, help="Max clusters to search after AMP (default: max(n_probe,32) or tuned)")
    p.add_argument("--adaptive-margin", type=float, default=None, help="Score gap threshold for AMP (default 0.028)")
    p.add_argument("--top-k-refine", type=int, default=20)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--hnsw-M", type=int, default=32)
    p.add_argument("--hnsw-ef-construct", type=int, default=200)
    p.add_argument("--hnsw-ef-search", type=int, default=64)
    p.add_argument("--ivf-nlist", type=int, default=None, help="default: ~4*sqrt(N)")
    p.add_argument("--ivf-nprobe", type=int, default=8)
    p.add_argument("--skip-ivf", action="store_true")
    p.add_argument("--skip-hnsw", action="store_true")
    p.add_argument("--quiet", action="store_true", help="Less DCEE logging / no tqdm progress")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )
    rng = np.random.default_rng(args.seed)

    if args.mode == "synthetic":
        emb = make_synthetic_embeddings(
            args.n, args.dim, args.n_topics, args.noise, args.seed
        )
        n_clusters = args.n_clusters or default_n_clusters(len(emb))
    else:
        if args.text_file:
            texts = load_text_lines(args.text_file)
        else:
            texts = _repeat_corpus(DEFAULT_PARAGRAPHS, args.max_docs)
        texts = texts[: args.max_docs]
        if not texts:
            raise SystemExit("No input texts.")
        emb, _ = embed_texts(texts, args.model)
        n_clusters = args.n_clusters or default_n_clusters(len(emb))

    n, dim = emb.shape
    q_indices = rng.integers(0, n, size=args.n_queries)
    k = args.top_k

    print(f"\nDataset: N={n:,}  dim={dim}  queries={args.n_queries}  K={k}\n")

    t0 = time.time()
    gt = exact_topk_ids(emb, q_indices, k)
    t_exact = time.time() - t0
    print(f"Exact top-K precompute (batched rows): {t_exact:.3f}s (used as ground truth)\n")

    rows: List[Dict[str, object]] = []

    # --- DCEE (optional tuned defaults + Adaptive Margin Probing) ---
    if args.tuned:
        cfg = DCEEConfig.tuned_for(n, dim)
        if args.n_clusters is not None:
            cfg.n_clusters = args.n_clusters
        cfg.quantization = args.quantization
        cfg.top_k_refine = max(cfg.top_k_refine, args.top_k_refine, k)
        if args.n_probe_max is not None:
            cfg.n_probe_max = max(cfg.n_probe, args.n_probe_max)
        if args.adaptive_margin is not None:
            cfg.adaptive_probe_margin = args.adaptive_margin
        if args.no_adaptive:
            cfg.adaptive_probe = False
            cfg.adaptive_probe_margin = 0.0
        if args.quiet:
            cfg.verbose = False
    else:
        npm = args.n_probe_max if args.n_probe_max is not None else max(args.n_probe, 32)
        margin = args.adaptive_margin if args.adaptive_margin is not None else 0.028
        cfg = DCEEConfig(
            dim=dim,
            n_clusters=n_clusters,
            keyframe_every=args.keyframe_every,
            quantization=args.quantization,
            top_k_refine=max(args.top_k_refine, k),
            n_probe=args.n_probe,
            n_probe_max=max(args.n_probe, npm),
            adaptive_probe=not args.no_adaptive,
            adaptive_probe_margin=0.0 if args.no_adaptive else margin,
        )
        if args.quiet:
            cfg.verbose = False
    dcee_label = "DCEE+AMP (tuned)" if args.tuned else ("DCEE+AMP" if cfg.adaptive_probe else "DCEE")
    engine = DCEEEngine(cfg)
    t0 = time.time()
    engine.build(emb)
    build_dcee = time.time() - t0

    for w in range(args.warmup):
        jw = w % args.n_queries
        engine.search(emb[int(q_indices[jw])], top_k=k)

    dcee_lat: List[float] = []
    dcee_pred = np.empty((args.n_queries, k), dtype=np.int64)
    for j in range(args.n_queries):
        t0 = time.perf_counter()
        res = engine.search(emb[int(q_indices[j])], top_k=k)
        dcee_lat.append((time.perf_counter() - t0) * 1000)
        dcee_pred[j] = [int(r[0]) for r in res]
    dcee_p50, dcee_p95, dcee_med = (
        float(np.percentile(dcee_lat, 50)),
        float(np.percentile(dcee_lat, 95)),
        float(np.median(dcee_lat)),
    )
    rec_dcee = mean_recall_at_k(dcee_pred, gt)
    rows.append(
        {
            "method": dcee_label,
            "recall": rec_dcee,
            "p50_ms": dcee_p50,
            "p95_ms": dcee_p95,
            "qps": 1000.0 / dcee_med if dcee_med > 0 else 0.0,
            "build_s": build_dcee,
            "size_mb": dcee_index_bytes(engine) / 1e6,
        }
    )

    if not _FAISS_OK:
        print("Install faiss-cpu for FAISS baselines: pip install faiss-cpu")
        _print_table(rows)
        return

    # --- FAISS Flat (exact IP) ---
    t0 = time.time()
    idx_flat = build_faiss_flat(emb)
    build_flat = time.time() - t0

    for w in range(args.warmup):
        jw = w % args.n_queries
        qrow = emb[int(q_indices[jw]) : int(q_indices[jw]) + 1].astype(np.float32, copy=False)
        idx_flat.search(qrow, k)

    flat_lat: List[float] = []
    flat_pred = np.empty((args.n_queries, k), dtype=np.int64)
    for j in range(args.n_queries):
        t0 = time.perf_counter()
        qrow = emb[int(q_indices[j]) : int(q_indices[j]) + 1].astype(np.float32, copy=False)
        _, I = idx_flat.search(qrow, k)
        flat_pred[j] = I[0]
        flat_lat.append((time.perf_counter() - t0) * 1000)
    flat_p50 = float(np.percentile(flat_lat, 50))
    flat_med = float(np.median(flat_lat))
    rec_flat = mean_recall_at_k(flat_pred, gt)
    rows.append(
        {
            "method": "FAISS FlatIP",
            "recall": rec_flat,
            "p50_ms": flat_p50,
            "p95_ms": float(np.percentile(flat_lat, 95)),
            "qps": 1000.0 / flat_med if flat_med > 0 else 0.0,
            "build_s": build_flat,
            "size_mb": faiss_index_bytes(idx_flat) / 1e6,
        }
    )

    # --- FAISS HNSW ---
    if not args.skip_hnsw:
        t0 = time.time()
        idx_hnsw = build_faiss_hnsw(
            emb,
            M=args.hnsw_M,
            ef_construction=args.hnsw_ef_construct,
            ef_search=args.hnsw_ef_search,
        )
        build_hnsw = time.time() - t0

        for w in range(args.warmup):
            jw = w % args.n_queries
            qrow = emb[int(q_indices[jw]) : int(q_indices[jw]) + 1].astype(np.float32, copy=False)
            idx_hnsw.search(qrow, k)

        hnsw_lat: List[float] = []
        hnsw_pred = np.empty((args.n_queries, k), dtype=np.int64)
        for j in range(args.n_queries):
            t0 = time.perf_counter()
            qrow = emb[int(q_indices[j]) : int(q_indices[j]) + 1].astype(np.float32, copy=False)
            _, I = idx_hnsw.search(qrow, k)
            hnsw_pred[j] = I[0]
            hnsw_lat.append((time.perf_counter() - t0) * 1000)
        hnsw_med = float(np.median(hnsw_lat))
        rec_hnsw = mean_recall_at_k(hnsw_pred, gt)
        rows.append(
            {
                "method": f"HNSW (M={args.hnsw_M}, ef={args.hnsw_ef_search})",
                "recall": rec_hnsw,
                "p50_ms": float(np.percentile(hnsw_lat, 50)),
                "p95_ms": float(np.percentile(hnsw_lat, 95)),
                "qps": 1000.0 / hnsw_med if hnsw_med > 0 else 0.0,
                "build_s": build_hnsw,
                "size_mb": faiss_index_bytes(idx_hnsw) / 1e6,
            }
        )

    # --- FAISS IVF ---
    if not args.skip_ivf:
        nlist = args.ivf_nlist or max(4, min(int(4 * np.sqrt(n)), n // 4))
        nlist = max(1, min(nlist, max(1, n // 2)))
        idx_ivf = None
        build_ivf = 0.0
        ivf_err: Optional[str] = None
        try:
            t0 = time.time()
            idx_ivf = build_faiss_ivf(emb, nlist=nlist, nprobe=args.ivf_nprobe)
            build_ivf = time.time() - t0 if idx_ivf is not None else 0.0
        except Exception as exc:
            ivf_err = str(exc)
        if idx_ivf is None:
            extra = f" ({ivf_err})" if ivf_err else ""
            print(f"IVF not in benchmark (n={n}, nlist={nlist}){extra}.\n")
        else:
            for w in range(args.warmup):
                jw = w % args.n_queries
                qrow = emb[int(q_indices[jw]) : int(q_indices[jw]) + 1].astype(np.float32, copy=False)
                idx_ivf.search(qrow, k)

            ivf_lat: List[float] = []
            ivf_pred = np.empty((args.n_queries, k), dtype=np.int64)
            for j in range(args.n_queries):
                t0 = time.perf_counter()
                qrow = emb[int(q_indices[j]) : int(q_indices[j]) + 1].astype(np.float32, copy=False)
                _, I = idx_ivf.search(qrow, k)
                ivf_pred[j] = I[0]
                ivf_lat.append((time.perf_counter() - t0) * 1000)
            ivf_med = float(np.median(ivf_lat))
            rec_ivf = mean_recall_at_k(ivf_pred, gt)
            rows.append(
                {
                    "method": f"IVF (nlist={nlist}, nprobe={idx_ivf.nprobe})",
                    "recall": rec_ivf,
                    "p50_ms": float(np.percentile(ivf_lat, 50)),
                    "p95_ms": float(np.percentile(ivf_lat, 95)),
                    "qps": 1000.0 / ivf_med if ivf_med > 0 else 0.0,
                    "build_s": build_ivf,
                    "size_mb": faiss_index_bytes(idx_ivf) / 1e6,
                }
            )

    _print_table(rows)


def _print_table(rows: List[Dict[str, object]]) -> None:
    print("══════════════════════════════════════════════════════════════════════════════")
    print(f"  {'Method':<28} {'Recall@K':>10} {'P50 ms':>10} {'P95 ms':>10} {'QPS':>8} {'Build s':>9} {'Size MB':>9}")
    print("  " + "─" * 76)
    for r in rows:
        print(
            f"  {r['method']!s:<28} {100 * float(r['recall']):>9.1f}% "
            f"{float(r['p50_ms']):>10.2f} {float(r['p95_ms']):>10.2f} "
            f"{float(r['qps']):>8.0f} {float(r['build_s']):>9.2f} {float(r['size_mb']):>9.2f}"
        )
    print("══════════════════════════════════════════════════════════════════════════════")
    print(
        "Recall@K = mean overlap with exact top-K (inner product / cosine on normalized vectors).\n"
        "FAISS FlatIP should be ~100% recall; DCEE/HNSW/IVF show approximate tradeoffs.\n"
    )


if __name__ == "__main__":
    main()
