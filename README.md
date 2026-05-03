# DCEE — Delta-Compressed Embedding Engine

Compressed approximate similarity search for **correlated** embedding sequences (e.g. chunks from one document, adjacent logs). Uses k-means routing, delta coding inside clusters, optional **Adaptive Margin Probing (AMP)** at query time, and optional **CuPy** for GPU math (falls back to NumPy).

## Install

```bash
pip install .
```

Optional GPU: install a matching [CuPy](https://docs.cupy.dev/) wheel for your CUDA version (e.g. `cupy-cuda12x`).

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
