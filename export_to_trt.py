from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODELS_DIR = Path("models") / "yolo"
IMG_SIZE = 640


def export_weight(path: Path, device: int, img_size: int) -> None:
    from ultralytics import YOLO

    print(f"Exporting {path} -> TensorRT engine (FP16, dynamic)")
    model = YOLO(str(path))
    model.export(
        format="engine",
        device=device,
        half=True,
        imgsz=img_size,
        dynamic=True,
        workspace=4096,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO .pt weights to TensorRT engines.")
    parser.add_argument("--weights", type=Path, help="Specific .pt file to export.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                        help="Directory of .pt files to export when --weights is omitted.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Export image size.")
    args = parser.parse_args()

    weights = [args.weights] if args.weights else sorted(args.models_dir.glob("*.pt"))
    if not weights:
        print(f"No .pt files found in {args.models_dir}")
        return

    for path in weights:
        export_weight(path, args.device, args.imgsz)
    print("Done.")


if __name__ == "__main__":
    main()
