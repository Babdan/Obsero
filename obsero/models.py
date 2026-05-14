"""
obsero.models — Weight resolution with heterogeneous multi-GPU support.

Search order for a given YOLO stem + gpu_id:
    If TensorRT bindings are available:
        1) models/yolo/{stem}_sm{XX}.engine (arch-specific TRT)
        2) models/yolo/{stem}.engine        (generic TRT)
    Always:
        3) models/yolo/{stem}.pt            (PyTorch fallback)

Root-level models/{stem}.* paths are kept as a compatibility fallback.
"""

from __future__ import annotations
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
YOLO_MODELS_DIR = MODELS_DIR / "yolo"


def _yolo_model_dirs() -> tuple[Path, Path]:
    """Prefer the organized YOLO folder, then old root-level model paths."""
    return YOLO_MODELS_DIR, MODELS_DIR


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
        for model_dir in _yolo_model_dirs():
            arch_engine = model_dir / f"{stem}_{cap}.engine"
            if arch_engine.exists():
                return arch_engine, True

    # 2) generic engine (only if TensorRT runtime is available)
    if trt_ok:
        for model_dir in _yolo_model_dirs():
            generic_engine = model_dir / f"{stem}.engine"
            if generic_engine.exists():
                return generic_engine, True

    # 3) fallback to PyTorch
    pt = resolve_pytorch_weight_path(stem)
    if pt is not None:
        return pt, False

    return None, False


def resolve_pytorch_weight_path(stem: str) -> Path | None:
    """Resolve the PyTorch fallback for a YOLO model stem."""
    for model_dir in _yolo_model_dirs():
        pt = model_dir / f"{stem}.pt"
        if pt.exists():
            return pt
    return None
