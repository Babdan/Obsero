# run_webcam.py  (optimized: PPE helmet+vest only, add fall, higher conf, FP16, cadence)
import json, datetime, time, threading, queue
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
import supervision as sv

# ----------------- Settings -----------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
INCIDENTS_DIR = ROOT / "incidents"
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640  # lower = faster
# Confidence thresholds (>= 0.55 as requested)
CONF_PPE   = 0.60   # covers hat/nohat/vest/novest
CONF_SMOKE = 0.60   # cigarette
CONF_PHONE = 0.60
CONF_FIRE  = 0.60   # fire/smoke (scene)
CONF_FALL  = 0.60

# Model cadence (run every N frames to reduce total passes)
CADENCE = {
    "ppe": 1,          # every frame
    "smoke": 2,        # every 2nd frame
    "phone": 2,        # every 2nd frame
    "fire_smoke": 3,   # every 3rd frame
    "fall": 2          # every 2nd frame
}

# Debounce per TAG to avoid spamming (seconds)
DEFAULT_COOLDOWN = 5.0
PER_TAG_COOLDOWN = {"FIRE_SMOKE": 8.0}
LAST_EMIT = {}

# Thread-safe Display Queue
DISPLAY_QUEUE = queue.Queue(maxsize=1)
STOP_EVENT = threading.Event()

# Device
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == 0:
    print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print("PyTorch GPU optimizations enabled.")

# Weights (helmet-only removed)
VEST_HELMET_WEIGHTS = MODELS_DIR / "helmet_vest_best.pt"   # wesjos
SMOKE_WEIGHTS       = MODELS_DIR / "smoke_best.pt"         # cigarette
PHONE_WEIGHTS       = MODELS_DIR / "phone_best.pt"         # phone
FIRE_SMOKE_WEIGHTS  = MODELS_DIR / "fire_smoke_best.pt"    # fire+smoke
FALL_WEIGHTS        = MODELS_DIR / "fall_best.pt"          # fall

def load_model(path_or_name: Path):
    try:
        if isinstance(path_or_name, Path) and not path_or_name.exists():
            return None
        m = YOLO(str(path_or_name))
        print(f"Loaded model {path_or_name.name}")
        return m
    except Exception as e:
        print(f"[WARN] Could not load {path_or_name}: {e}")
        return None

def save_incident(frame, label, conf, xyxy):
    tag = label.split(':', 1)[0]
    now = time.time()
    cooldown = PER_TAG_COOLDOWN.get(tag, DEFAULT_COOLDOWN)
    if now - LAST_EMIT.get(tag, 0.0) < cooldown:
        return
    LAST_EMIT[tag] = now

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    x1, y1, x2, y2 = map(int, xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
    crop = frame[y1:y2, x1:x2]
    safe_label = label.replace(':', '_')
    img_name = f"{ts.replace(':','-').replace(' ','_')}_{safe_label}_{int(conf*100)}.jpg"
    cv2.imwrite(str(INCIDENTS_DIR / img_name), crop)
    event = {"ts": ts, "label": label, "conf": round(float(conf), 3), "bbox": [x1, y1, x2, y2], "image": img_name}
    with open(INCIDENTS_DIR / "events.jsonl", "a", encoding="utf-8") as f:
        print(json.dumps(event, ensure_ascii=False), file=f, flush=True)

def inference_thread(cap, rules, display_queue, stop_event):
    print("Inference thread started.")
    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3)

    frame_idx = 0
    fps = 0.0
    frame_count = 0
    t0 = time.time()

    with torch.inference_mode():
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed. Stopping inference thread.")
                break

            annotated = frame.copy()
            all_dets, all_labels = [], []

            for key, entry in rules.items():
                model, (tag, conf_thr, keywords), cadence = entry
                if cadence > 1 and (frame_idx % cadence) != 0:
                    continue  # skip this model this frame

                # FP16 inference via half=True
                results = model.predict(
                    source=frame, conf=conf_thr, device=DEVICE, imgsz=IMG_SIZE,
                    half=True, verbose=False
                )
                res  = results[0]
                dets = sv.Detections.from_ultralytics(res)

                labels = []
                for i, xyxy in enumerate(dets.xyxy):
                    cls_id   = int(dets.class_id[i])
                    cls_name = res.names.get(cls_id, str(cls_id))
                    score    = float(dets.confidence[i]) if dets.confidence is not None else 0.0
                    labels.append(f"{tag}:{cls_name} {score:.2f}")
                    if any(k.lower() in cls_name.lower() for k in keywords):
                        save_incident(frame, f"{tag}:{cls_name}", score, xyxy)

                if len(dets) > 0:
                    all_dets.append(dets)
                    all_labels.extend(labels)

            if all_dets:
                combined = sv.Detections.merge(all_dets)
                annotated = box_annotator.annotate(annotated, combined)
                annotated = label_annotator.annotate(annotated, combined, labels=all_labels)

            # FPS
            frame_count += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frame_count / dt
                frame_count = 0
                t0 = time.time()
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if not display_queue.full():
                display_queue.put(annotated)

            frame_idx += 1

    print("Inference thread finished.")

def main():
    # Load models
    ppe_model   = load_model(VEST_HELMET_WEIGHTS)   # hat/nohat + vest/novest
    smoke_model = load_model(SMOKE_WEIGHTS)         # cigarette
    phone_model = load_model(PHONE_WEIGHTS)         # smartphone
    fire_model  = load_model(FIRE_SMOKE_WEIGHTS)    # fire + smoke
    fall_model  = load_model(FALL_WEIGHTS)          # fall

    if not any([ppe_model, smoke_model, phone_model, fire_model, fall_model]):
        raise SystemExit("No models found. Place weights in ./models and rerun.")

    # Rules: when to raise incidents based on class names
    model_rules = {}
    if ppe_model:
        # Trigger on violations: 'nohat' and 'novest'; you can also log 'hat'/'vest' as OK events if needed
        model_rules["ppe"] = (ppe_model, ("PPE",   CONF_PPE,   ["nohat", "novest"]), CADENCE["ppe"])
    if smoke_model:
        model_rules["smoke"] = (smoke_model, ("SMOKE", CONF_SMOKE, ["cigarette", "smoke"]), CADENCE["smoke"])
    if phone_model:
        model_rules["phone"] = (phone_model, ("PHONE", CONF_PHONE, ["phone", "mobile", "smartphone"]), CADENCE["phone"])
    if fire_model:
        model_rules["fire_smoke"] = (fire_model, ("FIRE_SMOKE", CONF_FIRE, ["fire", "smoke"]), CADENCE["fire_smoke"])
    if fall_model:
        model_rules["fall"] = (fall_model, ("FALL", CONF_FALL, ["fall", "fallen", "falldown"]), CADENCE["fall"])

    # Webcam
    print("Searching for camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("❌ Could not open webcam. Check connection and drivers.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)
    print("✓ Camera opened successfully.")

    # Inference thread
    t = threading.Thread(target=inference_thread, args=(cap, model_rules, DISPLAY_QUEUE, STOP_EVENT), daemon=True)
    t.start()

    # Display loop
    window_name = "Webcam Safety Pilot - Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    print("Display loop started. Press 'Q' to quit.")

    while not STOP_EVENT.is_set():
        try:
            annotated = DISPLAY_QUEUE.get_nowait()
            cv2.imshow(window_name, annotated)
        except queue.Empty:
            time.sleep(0.001)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            STOP_EVENT.set()
            break

    t.join()
    cap.release()
    cv2.destroyAllWindows()
    print("Application finished.")

if __name__ == "__main__":
    main()
