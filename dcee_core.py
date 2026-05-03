# -*- coding: utf-8 -*-
# pyright: reportUnusedImport=false
"""
Compatibility shim: import from the ``dcee`` package instead.

    from dcee import DCEEConfig, DCEEEngine, ...
"""

from dcee import (
    ClusterBlockV2,
    DCEEConfig,
    DCEEEngine,
    DCEEIndex,
    __version__,
    is_gpu_available,
    load_index,
    save_index,
)

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
