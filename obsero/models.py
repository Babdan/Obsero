"""
obsero.models — Weight resolution with heterogeneous multi-GPU support.

Search order for a given stem + gpu_id:
    If TensorRT bindings are available:
        1) models/{stem}_sm{XX}.engine (arch-specific TRT)
        2) models/{stem}.engine        (generic TRT)
    Always:
        3) models/{stem}.pt            (PyTorch fallback)
"""

from __future__ import annotations
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def _tensorrt_available() -> bool:
    """
    Return True only when TensorRT Python bindings are importable.

    Env override:
      OBSERO_DISABLE_TRT=1  -> force .pt fallback even if TensorRT exists.
    """
    if os.environ.get("OBSERO_DISABLE_TRT", "0") == "1":
        return False
    return importlib.util.find_spec("tensorrt") is not None


def _compute_cap_str(gpu_id: int) -> str | None:
    """Return e.g. 'sm86' for the GPU, or None if CUDA unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(gpu_id)
        return f"sm{major}{minor}"
    except Exception:
        return None


def resolve_weight_path_for_gpu(stem: str, gpu_id: int) -> tuple[Path | None, bool]:
    """
    Resolve best weight file for *stem* on *gpu_id*.

    Returns (path, is_trt).  (None, False) if nothing found.
    """
    cap = _compute_cap_str(gpu_id)
    # TRT engines are only viable when TRT bindings and CUDA are both available.
    trt_ok = _tensorrt_available() and (cap is not None)

    # 1) arch-specific engine (only if TensorRT runtime is available)
    if trt_ok and cap:
        arch_engine = MODELS_DIR / f"{stem}_{cap}.engine"
        if arch_engine.exists():
            return arch_engine, True

    # 2) generic engine (only if TensorRT runtime is available)
    if trt_ok:
        generic_engine = MODELS_DIR / f"{stem}.engine"
        if generic_engine.exists():
            return generic_engine, True

    # 3) fallback to PyTorch
    pt = MODELS_DIR / f"{stem}.pt"
    if pt.exists():
        return pt, False

    return None, False
