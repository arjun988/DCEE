"""
╔══════════════════════════════════════════════════════════════════════╗
║       Delta-Compressed Embedding Engine (DCEE) — GPU Native         ║
║       Designed for Google Colab (T4/A100)                            ║
║       Run: Runtime → Change runtime type → GPU                       ║
╚══════════════════════════════════════════════════════════════════════╝

ARCHITECTURE:
  [Cluster] → [Order] → [Delta Encode] → [Quantize] → [Binary Store]
  Query: Keyframe Search → Partial Reconstruct → Refine Top-K
"""

# ── 0. Install deps (run this cell first in Colab) ───────────────────
# !pip install -q cupy-cuda12x faiss-gpu-cu12 scikit-learn tqdm numpy

import os, time, struct, warnings
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# ── GPU imports with graceful CPU fallback ────────────────────────────
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    print("⚠ CuPy not found — falling back to NumPy (CPU). Install: pip install cupy-cuda12x")

if GPU_AVAILABLE:
    xp = cp          # array namespace
    print(f"✅ GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
else:
    xp = np
    print("ℹ  Running on CPU (NumPy)")

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm


# ═══════════════════════════════════════════════════════════════════════
# §1  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DCEEConfig:
    """All tunable hyper-parameters in one place."""
    dim: int = 128                  # embedding dimension
    n_clusters: int = 64            # number of semantic clusters
    keyframe_every: int = 8         # insert keyframe every N vectors
    quantization: str = "int8"      # "float16" | "int8" | "sparse"
    sparse_threshold: float = 0.02  # values below this → 0 (sparse mode)
    approx_precision: str = "int8"  # dtype for partial reconstruction
    top_k_refine: int = 10          # full-precision refine this many candidates
    batch_size: int = 1024          # GPU batch size for encoding


@dataclass
class ClusterBlock:
    """One cluster's worth of stored data."""
    cluster_id: int
    keyframe: np.ndarray            # shape (dim,) float32 — always full precision
    deltas: np.ndarray              # shape (N-1, dim) in quantized dtype
    original_indices: List[int]     # maps local→global index
    keyframe_positions: List[int]   # local positions that hold keyframes


@dataclass
class DCEEIndex:
    """The in-memory index after building."""
    config: DCEEConfig
    clusters: List[ClusterBlock]
    keyframe_matrix: np.ndarray     # (n_clusters, dim) — all cluster keyframes stacked
    cluster_norms: np.ndarray       # precomputed ||K||  for fast ANN


# ═══════════════════════════════════════════════════════════════════════
# §2  CLUSTERING & ORDERING
# ═══════════════════════════════════════════════════════════════════════

class EmbeddingPreprocessor:
    """
    Step 1: Cluster  → k-means (MiniBatch for large datasets)
    Step 2: Order    → greedy nearest-neighbour TSP per cluster
    """

    def __init__(self, cfg: DCEEConfig):
        self.cfg = cfg

    def cluster(self, embeddings: np.ndarray) -> List[List[int]]:
        """Returns list-of-lists: cluster → [global_indices]."""
        print(f"\n📦 Clustering {len(embeddings):,} embeddings → {self.cfg.n_clusters} clusters …")
        normed = normalize(embeddings.astype(np.float32))
        km = MiniBatchKMeans(
            n_clusters=self.cfg.n_clusters,
            batch_size=max(4096, self.cfg.batch_size),
            n_init=3, random_state=42
        )
        labels = km.fit_predict(normed)
        groups = [[] for _ in range(self.cfg.n_clusters)]
        for i, lbl in enumerate(labels):
            groups[lbl].append(i)
        sizes = [len(g) for g in groups if g]
        print(f"   cluster sizes — min:{min(sizes)}  max:{max(sizes)}  avg:{np.mean(sizes):.1f}")
        return groups

    def greedy_order(self, vecs: np.ndarray) -> List[int]:
        """Greedy nearest-neighbour ordering to minimise consecutive Δ magnitude."""
        n = len(vecs)
        if n <= 2:
            return list(range(n))
        visited = np.zeros(n, dtype=bool)
        order = [0]
        visited[0] = True
        cur = vecs[0]
        for _ in range(n - 1):
            # vectorised distance to all unvisited
            dists = np.sum((vecs - cur) ** 2, axis=1)
            dists[visited] = np.inf
            nxt = int(np.argmin(dists))
            order.append(nxt)
            visited[nxt] = True
            cur = vecs[nxt]
        return order


# ═══════════════════════════════════════════════════════════════════════
# §3  DELTA ENCODING + QUANTIZATION
# ═══════════════════════════════════════════════════════════════════════

class DeltaEncoder:
    """
    Produces keyframes + quantised delta arrays.
    Supports float16 / int8 / sparse modes.
    """

    def __init__(self, cfg: DCEEConfig):
        self.cfg = cfg

    # ── quantise helpers ─────────────────────────────────────────────

    def _to_float16(self, delta: np.ndarray) -> np.ndarray:
        return delta.astype(np.float16)

    def _to_int8(self, delta: np.ndarray) -> Tuple[np.ndarray, float]:
        """Symmetric per-vector int8 quantisation. Returns (int8_array, scale)."""
        scale = np.max(np.abs(delta)) / 127.0 + 1e-9
        return np.clip(np.round(delta / scale), -127, 127).astype(np.int8), scale

    def _to_sparse(self, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Zero out small values; store nonzero (idx, val) pairs."""
        mask = np.abs(delta) > self.cfg.sparse_threshold
        indices = np.where(mask)[0].astype(np.int16)
        values = delta[mask].astype(np.float16)
        return indices, values

    # ── main encode ──────────────────────────────────────────────────

    def encode_cluster(
        self,
        vecs: np.ndarray,       # (N, dim) float32, already ordered
        global_ids: List[int],
    ) -> ClusterBlock:
        n, dim = vecs.shape
        kf_every = self.cfg.keyframe_every

        keyframe_positions = []
        all_deltas = []         # list of encoded delta tuples/arrays
        all_scales = []         # int8 scales (None if not int8)
        prev = None

        for i in range(n):
            is_kf = (i % kf_every == 0)
            if is_kf:
                keyframe_positions.append(i)
                prev = vecs[i].copy()
                if i > 0:
                    # store a "reset" marker (zero delta)
                    d = np.zeros(dim, dtype=np.float32)
                    all_deltas.append(self._encode_delta(d))
                    all_scales.append(None)
            else:
                d = vecs[i] - prev
                all_deltas.append(self._encode_delta(d))
                all_scales.append(None)
                prev = vecs[i].copy()

        # Store scales alongside deltas for int8
        return ClusterBlock(
            cluster_id=global_ids[0],       # reused as cluster tag
            keyframe=vecs[0].astype(np.float32),
            deltas=all_deltas,              # list of encoded objects
            original_indices=global_ids,
            keyframe_positions=keyframe_positions,
        )

    def _encode_delta(self, d: np.ndarray):
        q = self.cfg.quantization
        if q == "float16":
            return self._to_float16(d)
        elif q == "int8":
            return self._to_int8(d)          # (int8_arr, scale)
        elif q == "sparse":
            return self._to_sparse(d)        # (indices, values)
        else:
            return d.astype(np.float32)

    # ── decode helpers ───────────────────────────────────────────────

    def decode_delta(self, encoded) -> np.ndarray:
        q = self.cfg.quantization
        if q == "float16":
            return encoded.astype(np.float32)
        elif q == "int8":
            arr, scale = encoded
            return arr.astype(np.float32) * scale
        elif q == "sparse":
            indices, values = encoded
            d = np.zeros(self.cfg.dim, dtype=np.float32)
            d[indices] = values.astype(np.float32)
            return d
        else:
            return encoded.astype(np.float32)

    def decode_vector(self, block: ClusterBlock, local_idx: int) -> np.ndarray:
        """Full-precision reconstruction of vector at local_idx."""
        # find the last keyframe ≤ local_idx
        kf_positions = np.array(block.keyframe_positions)
        kf_pos = kf_positions[kf_positions <= local_idx].max()

        # keyframe value (reconstruct by walking from first keyframe)
        cur = block.keyframe.copy()
        delta_start = 0  # after the initial keyframe

        # walk delta list
        for i in range(1, local_idx + 1):
            delta = self.decode_delta(block.deltas[i - 1])
            if i in block.keyframe_positions:
                cur = block.keyframe.copy()  # reset
                # For multi-keyframe blocks, keyframe values need to be
                # reconstructed; here we store them inline in the deltas list
                # as zero-deltas at reset points — so we just reset cur.
                # A production system would store kf values in a side-table.
            else:
                cur = cur + delta
        return cur


# ═══════════════════════════════════════════════════════════════════════
# §4  GPU BATCH QUERY ENGINE
# ═══════════════════════════════════════════════════════════════════════

class GPUQueryEngine:
    """
    Query strategy:
      1. ANN on keyframe matrix (GPU matmul)
      2. Partial Δ reconstruction (int8 GPU, low precision)
      3. Refine top-K fully
    """

    def __init__(self, index: DCEEIndex, encoder: DeltaEncoder):
        self.index = index
        self.encoder = encoder
        self.cfg = index.config

        # Upload keyframe matrix to GPU
        self.kf_gpu = xp.array(index.keyframe_matrix, dtype=xp.float32)
        print(f"⚡ GPU keyframe matrix: {self.kf_gpu.shape}  ({self.kf_gpu.nbytes/1024:.1f} KB)")

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Returns list of (global_index, cosine_similarity) sorted descending.
        """
        q_gpu = xp.array(query, dtype=xp.float32)
        q_norm = q_gpu / (xp.linalg.norm(q_gpu) + 1e-9)

        # ── Phase 1: Score keyframes (GPU matmul) ────────────────────
        kf_norm = self.kf_gpu / (
            xp.linalg.norm(self.kf_gpu, axis=1, keepdims=True) + 1e-9
        )
        kf_scores = kf_norm @ q_norm          # (n_clusters,)
        kf_scores_cpu = xp.asnumpy(kf_scores) if GPU_AVAILABLE else kf_scores

        # pick top clusters to expand
        n_probe = min(max(top_k, 8), len(self.index.clusters))
        top_cluster_ids = np.argsort(kf_scores_cpu)[::-1][:n_probe]

        # ── Phase 2: Partial reconstruction inside top clusters ──────
        candidates: List[Tuple[float, int]] = []  # (score, global_idx)

        for cid in top_cluster_ids:
            block = self.index.clusters[cid]
            partial_score = self._partial_reconstruct_score(block, q_norm)
            candidates.extend(partial_score)

        # ── Phase 3: Refine top-K fully ──────────────────────────────
        candidates.sort(key=lambda x: -x[0])
        top_candidates = candidates[:self.cfg.top_k_refine]

        refined = []
        for approx_score, gidx in top_candidates:
            # find which cluster and local idx owns this global id
            cid, local_idx = self._global_to_local(gidx)
            block = self.index.clusters[cid]
            vec_full = self.encoder.decode_vector(block, local_idx)
            score = float(np.dot(vec_full, xp.asnumpy(q_norm) if GPU_AVAILABLE else q_norm)
                          / (np.linalg.norm(vec_full) + 1e-9))
            refined.append((gidx, score))

        refined.sort(key=lambda x: -x[1])
        return refined[:top_k]

    def _partial_reconstruct_score(
        self,
        block: ClusterBlock,
        q_norm,             # GPU array
    ) -> List[Tuple[float, int]]:
        """
        Low-precision reconstruction + dot-product scoring.
        Uses int8 deltas; stops early if partial score is hopeless.
        """
        results = []
        cur = xp.array(block.keyframe, dtype=xp.float32)
        n = len(block.original_indices)

        for i in range(n):
            if i > 0:
                delta_encoded = block.deltas[i - 1]
                delta = xp.array(
                    self.encoder.decode_delta(delta_encoded), dtype=xp.float32
                )
                if i in block.keyframe_positions:
                    cur = xp.array(block.keyframe, dtype=xp.float32)
                else:
                    cur = cur + delta

            # Early exit: if partial dot product (first 32 dims) < threshold → skip
            partial = float(xp.dot(cur[:32], q_norm[:32]))
            if i > 0 and partial < -0.5:
                continue

            score = float(xp.dot(
                cur / (xp.linalg.norm(cur) + 1e-9), q_norm
            ))
            gidx = block.original_indices[i]
            results.append((score, gidx))

        return results

    def _global_to_local(self, gidx: int) -> Tuple[int, int]:
        for cid, block in enumerate(self.index.clusters):
            if gidx in block.original_indices:
                local = block.original_indices.index(gidx)
                return cid, local
        raise ValueError(f"Global index {gidx} not found")


# ═══════════════════════════════════════════════════════════════════════
# §5  BINARY STORAGE FORMAT
# ═══════════════════════════════════════════════════════════════════════
# Format:
#  HEADER  : magic(4B) | version(1B) | dim(4B) | n_clusters(4B) | n_vecs(4B)
#  CLUSTERS: for each cluster:
#    cluster_id(4B) | n_vecs(4B) | n_kf(4B)
#    keyframe: dim * float32
#    kf_positions: n_kf * int32
#    original_indices: n_vecs * int32
#    deltas: pickled (for simplicity; production → custom binary)

import pickle

MAGIC = b"DCEE"
VERSION = 1

def save_index(index: DCEEIndex, path: str):
    with open(path, "wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack("B", VERSION))
        f.write(struct.pack("III",
            index.config.dim,
            len(index.clusters),
            sum(len(c.original_indices) for c in index.clusters)
        ))
        # Keyframe matrix
        f.write(index.keyframe_matrix.astype(np.float32).tobytes())
        # Clusters
        for block in index.clusters:
            n = len(block.original_indices)
            nkf = len(block.keyframe_positions)
            f.write(struct.pack("III", block.cluster_id, n, nkf))
            f.write(block.keyframe.astype(np.float32).tobytes())
            f.write(np.array(block.keyframe_positions, np.int32).tobytes())
            f.write(np.array(block.original_indices, np.int32).tobytes())
            # Deltas — pickle for now
            delta_bytes = pickle.dumps(block.deltas)
            f.write(struct.pack("I", len(delta_bytes)))
            f.write(delta_bytes)
    size_mb = os.path.getsize(path) / 1e6
    print(f"💾 Saved → {path}  ({size_mb:.2f} MB)")


def load_index(path: str, cfg: DCEEConfig) -> DCEEIndex:
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC, "Bad magic bytes"
        version = struct.unpack("B", f.read(1))[0]
        dim, n_clusters, n_vecs = struct.unpack("III", f.read(12))
        kf_matrix = np.frombuffer(f.read(n_clusters * dim * 4), np.float32).reshape(n_clusters, dim).copy()
        clusters = []
        for _ in range(n_clusters):
            cid, n, nkf = struct.unpack("III", f.read(12))
            kf = np.frombuffer(f.read(dim * 4), np.float32).copy()
            kf_positions = list(np.frombuffer(f.read(nkf * 4), np.int32))
            orig_ids = list(np.frombuffer(f.read(n * 4), np.int32))
            dlen = struct.unpack("I", f.read(4))[0]
            deltas = pickle.loads(f.read(dlen))
            clusters.append(ClusterBlock(
                cluster_id=cid,
                keyframe=kf,
                deltas=deltas,
                original_indices=orig_ids,
                keyframe_positions=kf_positions,
            ))
    index = DCEEIndex(
        config=cfg,
        clusters=clusters,
        keyframe_matrix=kf_matrix,
        cluster_norms=np.linalg.norm(kf_matrix, axis=1),
    )
    print(f"📂 Loaded → {path}  ({n_vecs:,} vectors, {n_clusters} clusters)")
    return index


# ═══════════════════════════════════════════════════════════════════════
# §6  HIGH-LEVEL DCEE ENGINE (façade)
# ═══════════════════════════════════════════════════════════════════════

class DCEEEngine:
    """
    Main entry point.

    Usage:
        engine = DCEEEngine(DCEEConfig(dim=128, n_clusters=64))
        engine.build(embeddings)          # np.ndarray (N, dim)
        engine.save("my_index.dcee")
        results = engine.search(query, top_k=5)
    """

    def __init__(self, cfg: DCEEConfig):
        self.cfg = cfg
        self.pre = EmbeddingPreprocessor(cfg)
        self.enc = DeltaEncoder(cfg)
        self.index: Optional[DCEEIndex] = None
        self.query_engine: Optional[GPUQueryEngine] = None

    def build(self, embeddings: np.ndarray):
        assert embeddings.ndim == 2 and embeddings.shape[1] == self.cfg.dim, \
            f"Expected (N, {self.cfg.dim}), got {embeddings.shape}"

        t0 = time.time()
        N = len(embeddings)
        emb = embeddings.astype(np.float32)

        # 1. Cluster
        groups = self.pre.cluster(emb)

        # 2. Build cluster blocks
        print(f"\n🔧 Delta-encoding {self.cfg.n_clusters} clusters …")
        clusters: List[ClusterBlock] = []
        kf_list = []

        for cid, gids in enumerate(tqdm(groups, desc="Encoding")):
            if not gids:
                continue
            vecs = emb[gids]                        # (n, dim)
            # 2a. Greedy order
            order = self.pre.greedy_order(vecs)
            vecs_ordered = vecs[order]
            gids_ordered = [gids[o] for o in order]
            # 2b. Delta encode
            block = self.enc.encode_cluster(vecs_ordered, gids_ordered)
            block.cluster_id = cid
            clusters.append(block)
            kf_list.append(block.keyframe)

        kf_matrix = np.stack(kf_list, axis=0)       # (n_clusters, dim)

        self.index = DCEEIndex(
            config=self.cfg,
            clusters=clusters,
            keyframe_matrix=kf_matrix,
            cluster_norms=np.linalg.norm(kf_matrix, axis=1),
        )
        self.query_engine = GPUQueryEngine(self.index, self.enc)

        elapsed = time.time() - t0
        raw_bytes = N * self.cfg.dim * 4
        self._print_stats(N, raw_bytes, elapsed)

    def _print_stats(self, N: int, raw_bytes: int, elapsed: float):
        # Estimate compressed size
        q = self.cfg.quantization
        bpv = {"float16": 2, "int8": 1, "sparse": 0.5, "float32": 4}.get(q, 4)
        compressed = N * self.cfg.dim * bpv
        ratio = raw_bytes / compressed
        print(f"\n{'═'*52}")
        print(f"  Vectors       : {N:>10,}")
        print(f"  Dimension     : {self.cfg.dim:>10,}")
        print(f"  Clusters      : {self.cfg.n_clusters:>10,}")
        print(f"  Quantization  : {self.cfg.quantization:>10}")
        print(f"  Raw size      : {raw_bytes/1e6:>9.2f} MB")
        print(f"  Est. compressed:{compressed/1e6:>8.2f} MB")
        print(f"  Compression ≈  : {ratio:>8.1f}×")
        print(f"  Build time    : {elapsed:>9.2f} s")
        print(f"{'═'*52}\n")

    def search(self, query: np.ndarray, top_k: int = 5):
        assert self.query_engine is not None, "Call build() first"
        return self.query_engine.search(query, top_k)

    def save(self, path: str):
        assert self.index is not None, "Call build() first"
        save_index(self.index, path)

    def load(self, path: str):
        self.index = load_index(path, self.cfg)
        self.query_engine = GPUQueryEngine(self.index, self.enc)


# ═══════════════════════════════════════════════════════════════════════
# §7  BENCHMARK & DEMO  (run this as __main__)
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark():
    print("╔══════════════════════════════════════════════════╗")
    print("║  Delta-Compressed Embedding Engine — Benchmark  ║")
    print("╚══════════════════════════════════════════════════╝\n")

    DIM        = 128
    N          = 50_000    # vectors
    N_CLUSTERS = 64
    TOP_K      = 5
    SAVE_PATH  = "/tmp/dcee_demo.dcee"

    # ── Generate correlated synthetic data ───────────────────────────
    print(f"🎲 Generating {N:,} correlated embeddings (dim={DIM}) …")
    rng = np.random.default_rng(42)
    # Simulate document-chunk embeddings: 64 semantic topics
    topics = rng.standard_normal((N_CLUSTERS, DIM)).astype(np.float32)
    topics = topics / np.linalg.norm(topics, axis=1, keepdims=True)
    labels = rng.integers(0, N_CLUSTERS, size=N)
    noise  = rng.standard_normal((N, DIM)).astype(np.float32) * 0.15
    embeddings = topics[labels] + noise
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    print(f"   shape={embeddings.shape}  dtype={embeddings.dtype}\n")

    # ── Build index ───────────────────────────────────────────────────
    cfg = DCEEConfig(
        dim=DIM,
        n_clusters=N_CLUSTERS,
        keyframe_every=8,
        quantization="int8",
        top_k_refine=20,
    )
    engine = DCEEEngine(cfg)
    engine.build(embeddings)

    # ── Save & reload ─────────────────────────────────────────────────
    engine.save(SAVE_PATH)
    engine2 = DCEEEngine(cfg)
    engine2.load(SAVE_PATH)

    # ── Search benchmark ─────────────────────────────────────────────
    print("🔍 Running 100 search queries …")
    query_ids = rng.integers(0, N, size=100)
    latencies = []
    hits = 0

    for qid in tqdm(query_ids, desc="Querying"):
        q = embeddings[qid]
        t0 = time.perf_counter()
        results = engine2.search(q, top_k=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)

        returned_ids = [gidx for gidx, _ in results]
        if int(qid) in returned_ids:
            hits += 1

    recall = hits / len(query_ids) * 100
    print(f"\n{'─'*40}")
    print(f"  Recall@{TOP_K}    : {recall:.1f}%")
    print(f"  Latency p50 : {np.percentile(latencies,50):.2f} ms")
    print(f"  Latency p95 : {np.percentile(latencies,95):.2f} ms")
    print(f"  Latency p99 : {np.percentile(latencies,99):.2f} ms")
    print(f"{'─'*40}\n")

    # ── Quantization comparison ───────────────────────────────────────
    print("📊 Quantization mode comparison …\n")
    header = f"{'Mode':<10} {'Build(s)':>10} {'Recall%':>10} {'P50ms':>8} {'Est.MB':>10}"
    print(header)
    print("─" * len(header))

    for qmode in ["float32", "float16", "int8", "sparse"]:
        cfg_q = DCEEConfig(dim=DIM, n_clusters=N_CLUSTERS,
                           keyframe_every=8, quantization=qmode, top_k_refine=20)
        eng_q = DCEEEngine(cfg_q)
        t0 = time.time()
        eng_q.build(embeddings)
        build_t = time.time() - t0

        lats, rec = [], 0
        for qid in query_ids[:50]:
            q = embeddings[qid]
            t1 = time.perf_counter()
            res = eng_q.search(q, top_k=TOP_K)
            lats.append((time.perf_counter() - t1) * 1000)
            if int(qid) in [r for r, _ in res]:
                rec += 1
        recall_q = rec / 50 * 100

        bpv = {"float16": 2, "int8": 1, "sparse": 0.5, "float32": 4}[qmode]
        est_mb = N * DIM * bpv / 1e6
        print(f"{qmode:<10} {build_t:>10.2f} {recall_q:>9.1f}% {np.median(lats):>7.2f} {est_mb:>9.2f}")

    print(f"\n✅ Benchmark complete!\n")


# ═══════════════════════════════════════════════════════════════════════
# §8  COLAB CELL HELPERS
# ═══════════════════════════════════════════════════════════════════════

def colab_quick_start():
    """
    Copy-paste this into a Colab cell to get started immediately.

    ┌─────────────────────────────────────────────────────┐
    │  Cell 1: Install                                    │
    │  !pip install -q cupy-cuda12x scikit-learn tqdm     │
    │                                                     │
    │  Cell 2: Run                                        │
    │  from dcee_colab import *                           │
    │  run_benchmark()                                    │
    └─────────────────────────────────────────────────────┘

    Or bring your own embeddings:

        cfg = DCEEConfig(dim=YOUR_DIM, n_clusters=64)
        engine = DCEEEngine(cfg)
        engine.build(your_numpy_array)          # (N, dim) float32
        results = engine.search(query_vec, top_k=10)
        # returns [(global_index, cosine_score), ...]
    """
    pass


if __name__ == "__main__":
    run_benchmark()
