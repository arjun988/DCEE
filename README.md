# DCEE — Delta-Compressed Embedding Engine

## Introduction

**DCEE** (Delta-Compressed Embedding Engine) targets embeddings that sit together in semantic space—**document chunks, chats, logs, or clustered corpora**—where sequential **delta coding** plus **quantization** shrinks storage versus raw float32 vectors. The pipeline **clusters** vectors (MiniBatch k-means), **orders** points inside each cluster to keep deltas small, stores **keyframes + deltas**, and at query time uses **keyframe routing** with optional **Adaptive Margin Probing (AMP)** to widen cluster search when scores are ambiguous. Math runs on **CuPy when available**, otherwise **NumPy**.

On **correlated synthetic benchmarks** in this repo (`benchmark_dcee.py`, **50,000** normalized vectors, **Recall@5** vs exact inner-product neighbors), a **tuned DCEE+AMP** configuration achieved **~96% recall** with **~4× smaller** on-disk payload than storing uncompressed float32 norms (see compressed size column below). Latency and recall depend on **hardware, `n_probe`, quantization, and dataset**; treat these as **example numbers**, not guarantees for every workload.

### Example benchmark snapshot (internal script, same queries for all methods)

| Method | Recall@5 | P50 (ms) | P95 (ms) | QPS (approx.) | Build (s) | Size (MB) |
|--------|----------|----------|----------|----------------|-----------|-----------|
| **DCEE+AMP (tuned)** | 96.4% | 1.37 | 1.95 | 422 | 12.57 | **6.40** |
| FAISS `IndexFlatIP` | 100.0% | 0.53 | 0.79 | 1897 | 0.01 | 25.60 |
| FAISS HNSW (`M=32`, `ef=64`) | 100.0% | 0.09 | 0.11 | 10689 | 0.63 | 39.21 |
| FAISS IVF-Flat (`nprobe=8`) | 90.6% | 0.03 | 0.03 | 36364 | 0.48 | 26.47 |

**Takeaway:** DCEE trades some recall versus exact flat search for **much smaller index bytes**; graph/IVF methods can be faster but use **different memory/compute tradeoffs**. Reproduce or tune with `benchmark_dcee.py` (and `tune_dcee.py`) on your own data.

## Install

From [PyPI](https://pypi.org/project/dcee/) (recommended):

```bash
pip install dcee
```

Install a specific release:

```bash
pip install "dcee>=0.1.0"
```

**Dependencies** (pulled in automatically): `numpy`, `scikit-learn`, `tqdm`. Python **3.10+**.

**Optional GPU acceleration:** install a [CuPy](https://docs.cupy.dev/) wheel that matches your CUDA toolkit (e.g. `cupy-cuda12x`). If CuPy is not installed, DCEE runs on **NumPy** (CPU).

**Development** (editable install from a clone):

```bash
git clone https://github.com/arjun988/DCEE.git
cd DCEE
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from dcee import DCEEConfig, DCEEEngine, is_gpu_available

print("GPU:", is_gpu_available())

emb = np.random.randn(10_000, 128).astype(np.float32)
emb /= np.linalg.norm(emb, axis=1, keepdims=True)

cfg = DCEEConfig.tuned_for(len(emb), emb.shape[1])
engine = DCEEEngine(cfg)
engine.build(emb)

q = emb[0]
for idx, score in engine.search(q, top_k=5):
    print(idx, score)

engine.save("index.dce2")

loaded = DCEEEngine.from_file("index.dce2")
print(loaded.search(q, top_k=3))
```

## Configuration

- **`DCEEConfig`**: defaults for `dim`, `n_clusters`, `keyframe_every`, `quantization`, `n_probe`, `n_probe_max`, AMP (`adaptive_probe`, `adaptive_probe_margin`), `top_k_refine`, `verbose`.
- **`DCEEConfig.tuned_for(n_vectors, dim)`**: heuristic scale-aware defaults.

Set `verbose=False` for quiet builds and loads.

## License

See `LICENSE` in the repository.
