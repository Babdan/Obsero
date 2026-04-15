#!/usr/bin/env python3
"""
install.py — Obsero dependency installer

Automatically:
  1. Detects installed CUDA version via nvidia-smi
  2. Installs torch + torchvision from the matching PyTorch CUDA wheel index
  3. Installs all requirements.txt packages
  4. Installs TensorRT (nvidia-pyindex + tensorrt) when a CUDA GPU is present

Usage:
    python install.py            # normal install
    python install.py --cpu      # force CPU-only (skip CUDA + TensorRT)
    python install.py --dry-run  # print commands without running them
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# ── PyTorch wheel tags ordered newest → oldest ─────────────────────────────
# Each entry: (min_cuda_major, min_cuda_minor, wheel_tag)
TORCH_CUDA_MAP = [
    (12, 4, "cu124"),
    (12, 1, "cu121"),
    (11, 8, "cu118"),
]
TORCH_INDEX_BASE = "https://download.pytorch.org/whl"

HERE = Path(__file__).parent


# ── Helpers ─────────────────────────────────────────────────────────────────

def pip(*args: str, dry_run: bool = False) -> None:
    cmd = [sys.executable, "-m", "pip", *args]
    print("  $", " ".join(cmd))
    if not dry_run:
        subprocess.check_call(cmd)


def detect_cuda() -> tuple[int, int] | None:
    """Return (major, minor) from nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.DEVNULL, text=True
        )
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return None


def cuda_wheel_tag(major: int, minor: int) -> str:
    """Map a CUDA version to the closest supported PyTorch wheel tag."""
    for req_major, req_minor, tag in TORCH_CUDA_MAP:
        if (major, minor) >= (req_major, req_minor):
            return tag
    return "cpu"  # CUDA too old for any supported wheel


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Obsero installer")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only install (skip CUDA + TensorRT)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    dry = args.dry_run
    print("=" * 50)
    print("  Obsero Installer")
    print("=" * 50)

    # ── Step 1: Detect CUDA ──────────────────────────────────────────────────
    cuda = None if args.cpu else detect_cuda()

    if args.cpu:
        print("\nMode: CPU-only (--cpu flag)")
        tag = "cpu"
    elif cuda:
        major, minor = cuda
        tag = cuda_wheel_tag(major, minor)
        if tag == "cpu":
            print(f"\nDetected CUDA {major}.{minor} — too old for supported PyTorch wheels.")
            print("Minimum supported: CUDA 11.8")
            print("Falling back to CPU-only PyTorch.")
        else:
            print(f"\nDetected CUDA {major}.{minor} → PyTorch wheel: {tag}")
    else:
        print("\nNo NVIDIA GPU / nvidia-smi not found — installing CPU-only PyTorch.")
        tag = "cpu"

    # ── Step 2: torch + torchvision ─────────────────────────────────────────
    index_url = f"{TORCH_INDEX_BASE}/{tag}"
    print(f"\n[1/3] Installing torch + torchvision ({tag}) ...")
    pip("install",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "--index-url", index_url,
        dry_run=dry)

    # ── Step 3: requirements.txt ─────────────────────────────────────────────
    req = HERE / "requirements.txt"
    print(f"\n[2/3] Installing {req.name} ...")
    pip("install", "-r", str(req), dry_run=dry)

    # ── Step 4: TensorRT (CUDA only) ─────────────────────────────────────────
    if cuda and tag != "cpu":
        print("\n[3/3] Installing TensorRT (CUDA GPU detected) ...")
        try:
            pip("install", "nvidia-pyindex", dry_run=dry)
            pip("install", "tensorrt>=8.6.0", dry_run=dry)
            print("  TensorRT installed.")
        except subprocess.CalledProcessError as e:
            print(f"\n  WARNING: TensorRT install failed ({e}).")
            print("  Obsero will fall back to .pt models — TensorRT is optional.")
    else:
        print("\n[3/3] Skipping TensorRT (no CUDA GPU).")

    print("\n" + "=" * 50)
    print("  Install complete.")
    if tag != "cpu":
        print(f"  CUDA build: {tag}")
    else:
        print("  Build: CPU-only")
    print("=" * 50)


if __name__ == "__main__":
    main()
