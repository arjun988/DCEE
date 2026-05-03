# -*- coding: utf-8 -*-
"""
DCEE — Delta-Compressed Embedding Engine.

Approximate nearest-neighbor search over delta-compressed embeddings (optional CuPy GPU).
"""

from dcee.core import (
    ClusterBlockV2,
    DCEEConfig,
    DCEEEngine,
    DCEEIndex,
    is_gpu_available,
    load_index,
    save_index,
)

__version__ = "1.0.0"

__all__ = [
    "ClusterBlockV2",
    "DCEEConfig",
    "DCEEEngine",
    "DCEEIndex",
    "__version__",
    "is_gpu_available",
    "load_index",
    "save_index",
]
