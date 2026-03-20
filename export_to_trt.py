# export_to_trt.py
from ultralytics import YOLO
from pathlib import Path

MODELS = Path("models")
IMG_SIZE = 640  # must match your runtime size

for pt in MODELS.glob("*.pt"):
    print(f"Exporting {pt} -> TensorRT engine (FP16, dynamic)")
    model = YOLO(str(pt))
    # creates models/<name>.engine next to the .pt
    model.export(
        format="engine",
        device=0,        # your RTX 5090
        half=True,       # FP16
        imgsz=IMG_SIZE,
        dynamic=True,    # dynamic shapes
        workspace=4096   # MB, adjust if needed
    )
print("Done.")
