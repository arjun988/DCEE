# -*- coding: utf-8 -*-
"""Internal implementation of the Delta-Compressed Embedding Engine."""

from __future__ import annotations

import logging
import os
import struct
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    cp = None  # type: ignore
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    xp = cp
else:
    xp = np


def is_gpu_available() -> bool:
    """Return True if CuPy sees a CUDA device (computations use GPU when True)."""
    return bool(GPU_AVAILABLE)


def to_cpu(a: Any) -> np.ndarray:
    if GPU_AVAILABLE and cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def to_gpu(a: Any):
    if GPU_AVAILABLE and cp is not None:
        return cp.array(a, dtype=cp.float32)
    return np.asarray(a, dtype=np.float32)


@dataclass
class DCEEConfig:
    dim: int = 128
    n_clusters: int = 64
    keyframe_every: int = 16
    quantization: str = "int8"  # 'float32' | 'float16' | 'int8'
    top_k_refine: int = 20
    n_probe: int = 8
    batch_size: int = 4096
    n_probe_max: int = 32
    adaptive_probe: bool = True
    adaptive_probe_margin: float = 0.028
    verbose: bool = True

    @staticmethod
    def tuned_for(n_vectors: int, dim: int) -> "DCEEConfig":
        n_vectors = max(n_vectors, 1)
        nc = max(16, min(512, int(np.sqrt(n_vectors))))
        probe = max(8, min(48, max(nc // 6, 8)))
        probe_max = min(nc, max(probe * 2, probe + 8))
        refine = max(32, min(256, n_vectors // 200))
        kfe = max(8, min(24, 14 + int(np.log2(max(n_vectors // 1000, 1)))))
        return DCEEConfig(
            dim=dim,
            n_clusters=nc,
            keyframe_every=kfe,
            quantization="int8",
            top_k_refine=refine,
            n_probe=probe,
            n_probe_max=probe_max,
            adaptive_probe=True,
            adaptive_probe_margin=0.028,
            batch_size=max(4096, min(8192, n_vectors // 4)),
        )


@dataclass
class ClusterBlockV2:
    cluster_id: int
    global_indices: np.ndarray
    delta_q: np.ndarray
    scales: np.ndarray
    kf_mask: np.ndarray
    kf_values: np.ndarray
    _gpu_reconstructed: object = None


@dataclass
class DCEEIndex:
    config: DCEEConfig
    clusters: list
    keyframe_matrix: np.ndarray


class EmbeddingPreprocessor:
    def __init__(self, cfg: DCEEConfig):
        self.cfg = cfg

    def cluster(self, emb: np.ndarray):
        if self.cfg.verbose:
            logger.info(
                "Clustering %s embeddings into %s clusters",
                f"{len(emb):,}",
                self.cfg.n_clusters,
            )
        km = MiniBatchKMeans(
            n_clusters=self.cfg.n_clusters,
            batch_size=max(4096, self.cfg.batch_size),
            n_init=3,
            random_state=42,
        )
        labels = km.fit_predict(normalize(emb))
        groups: List[List[int]] = [[] for _ in range(self.cfg.n_clusters)]
        for i, l in enumerate(labels):
            groups[l].append(i)
        sizes = [len(g) for g in groups if g]
        if sizes and self.cfg.verbose:
            logger.info(
                "Cluster sizes — min:%s max:%s avg:%.0f",
                min(sizes),
                max(sizes),
                float(np.mean(sizes)),
            )
        return groups

    def greedy_order(self, vecs: np.ndarray) -> List[int]:
        n = len(vecs)
        if n <= 2:
            return list(range(n))

        from sklearn.decomposition import PCA

        try:
            proj = PCA(n_components=1, random_state=42).fit_transform(vecs).ravel()
            return np.argsort(proj).tolist()
        except ValueError:
            return list(range(n))


class VectorisedDeltaEncoder:
    def __init__(self, cfg: DCEEConfig):
        self.cfg = cfg

    def encode_cluster(self, vecs: np.ndarray, global_ids: Sequence[int]) -> ClusterBlockV2:
        N, D = vecs.shape
        kfe = self.cfg.keyframe_every
        delta_f32 = np.zeros((N, D), np.float32)
        kf_mask = np.zeros(N, bool)
        kf_values = np.zeros((N, D), np.float32)
        cur_kf = vecs[0].copy()
        for i in range(N):
            if i % kfe == 0:
                kf_mask[i] = True
                cur_kf = vecs[i].copy()
                delta_f32[i] = 0.0
            else:
                delta_f32[i] = vecs[i] - vecs[i - 1]
            kf_values[i] = cur_kf

        q = self.cfg.quantization
        if q == "int8":
            scales = np.max(np.abs(delta_f32), axis=1) / 127.0 + 1e-9
            delta_q = np.clip(np.round(delta_f32 / scales[:, None]), -127, 127).astype(np.int8)
        elif q == "float16":
            delta_q = delta_f32.astype(np.float16)
            scales = np.ones(N, np.float32)
        else:
            delta_q = delta_f32
            scales = np.ones(N, np.float32)

        return ClusterBlockV2(
            cluster_id=0,
            global_indices=np.array(global_ids, np.int32),
            delta_q=delta_q,
            scales=scales.astype(np.float32),
            kf_mask=kf_mask,
            kf_values=kf_values,
        )

    def decode_cluster_gpu(self, block: ClusterBlockV2):
        if block._gpu_reconstructed is not None:
            return block._gpu_reconstructed
        N, D = block.delta_q.shape

        if self.cfg.quantization == "int8":
            delta_gpu = xp.array(block.delta_q, xp.float32) * xp.array(block.scales, xp.float32)[:, None]
        else:
            delta_gpu = xp.array(block.delta_q, xp.float32)

        delta_gpu[xp.array(block.kf_mask)] = 0.0
        cumsum_gpu = xp.cumsum(delta_gpu, axis=0)

        kf_indices = np.where(block.kf_mask)[0]
        kf_cumsum = cumsum_gpu[xp.array(kf_indices)]
        baseline = xp.zeros((N, D), xp.float32)
        for seg_i, kf_pos in enumerate(kf_indices):
            nxt = kf_indices[seg_i + 1] if seg_i + 1 < len(kf_indices) else N
            baseline[kf_pos:nxt] = kf_cumsum[seg_i]

        recon = xp.array(block.kf_values, xp.float32) + (cumsum_gpu - baseline)
        block._gpu_reconstructed = recon
        return recon


class GPUQueryEngineV2:
    def __init__(self, index: DCEEIndex, encoder: VectorisedDeltaEncoder):
        self.index = index
        self.encoder = encoder
        self.cfg = index.config
        kfm = index.keyframe_matrix.astype(np.float32)
        self.kf_gpu = to_gpu(kfm / (np.linalg.norm(kfm, axis=1, keepdims=True) + 1e-9))
        if self.cfg.verbose:
            logger.info("Pre-building cluster tensors (%s)", "GPU" if GPU_AVAILABLE else "CPU")
        t0 = time.time()
        for b in index.clusters:
            encoder.decode_cluster_gpu(b)
        if self.cfg.verbose:
            logger.info("Cluster tensor prep finished in %.2fs", time.time() - t0)

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        q_gpu = to_gpu(query)
        q_norm = q_gpu / (xp.linalg.norm(q_gpu) + 1e-9)

        kf_scores = to_cpu(self.kf_gpu @ q_norm)
        top_cids = self._select_probe_clusters(kf_scores)

        all_scores: List[np.ndarray] = []
        all_gidx: List[np.ndarray] = []
        for cid in top_cids:
            block = self.index.clusters[int(cid)]
            recon = block._gpu_reconstructed
            norms = xp.linalg.norm(recon, axis=1, keepdims=True) + 1e-9
            scores = to_cpu((recon / norms) @ q_norm)
            all_scores.append(scores)
            all_gidx.append(block.global_indices)

        cs = np.concatenate(all_scores)
        cg = np.concatenate(all_gidx)
        rk = min(self.cfg.top_k_refine, len(cs))
        top_li = np.argpartition(cs, -rk)[-rk:]
        top_li = top_li[np.argsort(cs[top_li])[::-1]]

        q_cpu = to_cpu(q_norm)
        refined: List[Tuple[int, float]] = []
        for li in top_li:
            gidx = int(cg[li])
            cid_r, local_r = self._rev_map[gidx]
            vec = to_cpu(self.index.clusters[cid_r]._gpu_reconstructed[local_r])
            score = float(np.dot(vec, q_cpu) / (np.linalg.norm(vec) + 1e-9))
            refined.append((gidx, score))
        refined.sort(key=lambda x: -x[1])
        return refined[:top_k]

    def _select_probe_clusters(self, kf_scores: np.ndarray) -> np.ndarray:
        nc = len(kf_scores)
        base = min(max(1, self.cfg.n_probe), nc)
        cap = min(max(base, self.cfg.n_probe_max), nc)
        sorted_idx = np.argsort(-kf_scores)
        selected: List[int] = list(sorted_idx[:base])
        if self.cfg.adaptive_probe and cap > base and self.cfg.adaptive_probe_margin > 0:
            margin = float(self.cfg.adaptive_probe_margin)
            for j in range(base, cap):
                if kf_scores[sorted_idx[j - 1]] - kf_scores[sorted_idx[j]] < margin:
                    selected.append(int(sorted_idx[j]))
                else:
                    break
        return np.asarray(selected, dtype=np.int64)

    def _build_reverse_map(self) -> None:
        self._rev_map: dict = {}
        for cid, b in enumerate(self.index.clusters):
            for li, gi in enumerate(b.global_indices):
                self._rev_map[int(gi)] = (cid, li)


MAGIC = b"DCE2"
# v3: AMP block used native "BIf" (often 12 bytes with padding — loaders must not assume 9).
# v4: AMP block uses "=BIf" (exactly 9 bytes, portable).
FORMAT_VERSION = 4
AMP_PACK_LEGACY = "BIf"  # v3 only
AMP_PACK = "=BIf"  # v4 — standard alignment, no padding


def save_index(index: DCEEIndex, path: Union[str, Path], *, verbose: bool = True) -> None:
    path = Path(path)
    cfg = index.config
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("B", FORMAT_VERSION))
        f.write(struct.pack("IIIII", cfg.dim, cfg.n_clusters, cfg.keyframe_every, cfg.top_k_refine, cfg.n_probe))
        f.write(struct.pack("B", ["float32", "float16", "int8"].index(cfg.quantization)))
        ap = 1 if getattr(cfg, "adaptive_probe", False) else 0
        npm = int(getattr(cfg, "n_probe_max", cfg.n_probe))
        margin = float(getattr(cfg, "adaptive_probe_margin", 0.0))
        f.write(struct.pack(AMP_PACK, ap, npm, margin))
        f.write(index.keyframe_matrix.astype(np.float32).tobytes())
        for b in index.clusters:
            N = len(b.global_indices)
            f.write(struct.pack("II", b.cluster_id, N))
            f.write(b.global_indices.astype(np.int32).tobytes())
            f.write(b.kf_mask.astype(np.uint8).tobytes())
            f.write(b.kf_values.astype(np.float32).tobytes())
            f.write(b.scales.astype(np.float32).tobytes())
            dt = {"float32": 0, "float16": 1, "int8": 2}[cfg.quantization]
            f.write(struct.pack("B", dt))
            f.write(b.delta_q.tobytes())
    if verbose:
        logger.info("Saved index → %s (%.2f MB)", path, path.stat().st_size / 1e6)


def load_index(path: Union[str, Path], *, verbose: bool = True) -> DCEEIndex:
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Not a DCEE index file (bad magic): {path}")
        ver = struct.unpack("B", f.read(1))[0]
        if ver > FORMAT_VERSION:
            raise ValueError(f"Unsupported index format version {ver} (max {FORMAT_VERSION})")
        dim, nc, kfe, topk, nprobe = struct.unpack("IIIII", f.read(20))
        qm = ["float32", "float16", "int8"][struct.unpack("B", f.read(1))[0]]
        if ver >= 4:
            amp_n = struct.calcsize(AMP_PACK)
            buf = f.read(amp_n)
            if len(buf) != amp_n:
                raise ValueError(f"Truncated AMP header in {path}")
            ap_b, npm, margin = struct.unpack(AMP_PACK, buf)
            adaptive_probe = bool(ap_b)
            n_probe_max = int(npm)
            adaptive_margin = float(margin)
        elif ver == 3:
            amp_n = struct.calcsize(AMP_PACK_LEGACY)
            buf = f.read(amp_n)
            if len(buf) != amp_n:
                raise ValueError(f"Truncated AMP header in {path}")
            ap_b, npm, margin = struct.unpack(AMP_PACK_LEGACY, buf)
            adaptive_probe = bool(ap_b)
            n_probe_max = int(npm)
            adaptive_margin = float(margin)
        else:
            adaptive_probe = False
            n_probe_max = nprobe
            adaptive_margin = 0.0
        cfg = DCEEConfig(
            dim=dim,
            n_clusters=nc,
            keyframe_every=kfe,
            quantization=qm,
            top_k_refine=topk,
            n_probe=nprobe,
            n_probe_max=n_probe_max,
            adaptive_probe=adaptive_probe,
            adaptive_probe_margin=adaptive_margin,
        )
        kfm = np.frombuffer(f.read(nc * dim * 4), np.float32).reshape(nc, dim).copy()
        clusters = []
        for _ in range(nc):
            cid, N = struct.unpack("II", f.read(8))
            gidx = np.frombuffer(f.read(N * 4), np.int32).copy()
            kf_mask = np.frombuffer(f.read(N), np.uint8).astype(bool).copy()
            kf_vals = np.frombuffer(f.read(N * dim * 4), np.float32).reshape(N, dim).copy()
            scales = np.frombuffer(f.read(N * 4), np.float32).copy()
            dt_tag = struct.unpack("B", f.read(1))[0]
            dt = [np.float32, np.float16, np.int8][dt_tag]
            delta_q = np.frombuffer(f.read(N * dim * np.dtype(dt).itemsize), dt).reshape(N, dim).copy()
            clusters.append(ClusterBlockV2(cid, gidx, delta_q, scales, kf_mask, kf_vals))
        total = sum(len(b.global_indices) for b in clusters)
        if verbose:
            logger.info("Loaded index ← %s (%s vectors, %s clusters)", path, f"{total:,}", nc)
    return DCEEIndex(cfg, clusters, kfm)


class DCEEEngine:
    """High-level API: build a compressed index, search, persist."""

    def __init__(self, cfg: DCEEConfig):
        self.cfg = cfg
        self.pre = EmbeddingPreprocessor(cfg)
        self.enc = VectorisedDeltaEncoder(cfg)
        self.index: DCEEIndex | None = None
        self.qe: GPUQueryEngineV2 | None = None

    @classmethod
    def from_file(cls, path: Union[str, Path], *, verbose: bool | None = None) -> "DCEEEngine":
        """
        Load a persisted index from disk. Does not require a placeholder config:
        dimensions and hyperparameters are read from the file.
        """
        path = Path(path)
        idx = load_index(path, verbose=verbose if verbose is not None else True)
        eng = cls(idx.config)
        if verbose is not None:
            eng.cfg.verbose = verbose
        eng.index = idx
        eng.enc = VectorisedDeltaEncoder(eng.cfg)
        eng.qe = GPUQueryEngineV2(idx, eng.enc)
        eng.qe._build_reverse_map()
        return eng

    def build(self, emb: np.ndarray) -> None:
        emb = emb.astype(np.float32)
        N = len(emb)
        t0 = time.time()
        groups = self.pre.cluster(emb)
        if self.cfg.verbose:
            logger.info("Delta-encoding %s clusters", self.cfg.n_clusters)
        clusters: List[ClusterBlockV2] = []
        kf_list: List[np.ndarray] = []
        for cid, gids in enumerate(
            tqdm(groups, desc="Encoding", disable=not self.cfg.verbose),
        ):
            if not gids:
                continue
            vecs = emb[gids]
            order = self.pre.greedy_order(vecs)
            vecs_o = vecs[order]
            gids_o = [gids[o] for o in order]
            block = self.enc.encode_cluster(vecs_o, gids_o)
            block.cluster_id = cid
            clusters.append(block)
            kf_list.append(vecs_o[0])
        kfm = np.stack(kf_list)
        self.index = DCEEIndex(self.cfg, clusters, kfm)
        self.qe = GPUQueryEngineV2(self.index, self.enc)
        self.qe._build_reverse_map()
        bpv = {"float32": 4, "float16": 2, "int8": 1}.get(self.cfg.quantization, 4)
        raw = N * self.cfg.dim * 4 / 1e6
        comp = N * self.cfg.dim * bpv / 1e6
        elapsed = time.time() - t0
        if self.cfg.verbose:
            logger.info(
                "Build summary: vectors=%s raw_MB=%.2f compressed_MB=%.2f ratio=%.1fx time=%.2fs",
                f"{N:,}",
                raw,
                comp,
                raw / comp if comp > 0 else 0.0,
                elapsed,
            )

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.qe is None:
            raise RuntimeError("Call build() or from_file() before search()")
        return self.qe.search(query, top_k)

    def save(self, path: Union[str, Path]) -> None:
        if self.index is None:
            raise RuntimeError("No index to save; call build() first")
        save_index(self.index, path, verbose=self.cfg.verbose)

    def load(self, path: Union[str, Path]) -> None:
        loaded = type(self).from_file(path, verbose=self.cfg.verbose)
        self.cfg = loaded.cfg
        self.pre = loaded.pre
        self.enc = loaded.enc
        self.index = loaded.index
        self.qe = loaded.qe
