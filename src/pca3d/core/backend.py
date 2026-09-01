"""NumPy/CuPy dispatch.

Design: functions that manipulate state arrays derive their array module from the
*input array*, so the same code path serves both backends and the choice is made once,
where the ensemble is allocated. Nothing imports cupy at module load; a missing or
broken CUDA stack degrades to numpy with an explicit reason available.
"""

from __future__ import annotations

import numpy as np

_cupy = None
_cupy_reason = "not attempted"


def _try_cupy():
    global _cupy, _cupy_reason
    if _cupy is not None or _cupy_reason not in ("not attempted",):
        return _cupy
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        cp.arange(4).sum()  # force a kernel launch so failures surface here
        _cupy = cp
        _cupy_reason = "ok"
    except Exception as exc:  # pragma: no cover - host dependent
        _cupy = None
        _cupy_reason = f"unavailable: {exc}"
    return _cupy


def gpu_available() -> bool:
    return _try_cupy() is not None


def gpu_status() -> str:
    _try_cupy()
    return _cupy_reason


def array_module(a):
    """The array module (numpy or cupy) that owns array ``a``."""
    cp = _try_cupy()
    if cp is not None and isinstance(a, cp.ndarray):
        return cp
    return np


def to_device(a: np.ndarray, use_gpu: bool):
    """Move a numpy array to the GPU (or return it unchanged)."""
    if not use_gpu:
        return a
    cp = _try_cupy()
    if cp is None:
        raise RuntimeError(f"GPU requested but {gpu_status()}")
    return cp.asarray(a)


def to_numpy(a) -> np.ndarray:
    cp = _try_cupy()
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)
