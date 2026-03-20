# run_webcam.py
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

IMG_SIZE = 640
CONF_HELMET = 0.40
CONF_SMOKE  = 0.40
CONF_PHONE  = 0.45

# --- Thread-safe Queue for Display ---
DISPLAY_QUEUE = queue.Queue(maxsize=1)
STOP_EVENT = threading.Event()

# Prefer GPU if available
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == 0:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Running on GPU: {gpu_name}")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print("PyTorch GPU optimizations enabled.")

# Model file paths
HELMET_WEIGHTS = MODELS_DIR / "helmet_best.pt"
SMOKE_WEIGHTS  = MODELS_DIR / "smoke_best.pt"
PHONE_WEIGHTS  = MODELS_DIR / "phone_best.pt"

def load_model(path_or_name):
    try:
        if isinstance(path_or_name, Path) and not path_or_name.exists():
            return None
        model = YOLO(str(path_or_name))
        model.to(DEVICE)
        print(f"Loaded model {path_or_name.name} to {DEVICE}")
        return model
    except Exception as e:
        print(f"[WARN] Could not load {path_or_name}: {e}")
        return None

def save_incident(frame, label, conf, xyxy):
    """Saves a cropped image of the incident and logs the event."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    x1, y1, x2, y2 = map(int, xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
    crop = frame[y1:y2, x1:x2]
    
    img_name = f"{ts.replace(':','-').replace(' ','_')}_{label.replace(':', '_')}_{int(conf*100)}.jpg"
    cv2.imwrite(str(INCIDENTS_DIR / img_name), crop)
    
    event = {"ts": ts, "label": label, "conf": round(float(conf), 3), "bbox": [x1, y1, x2, y2], "image": img_name}
    with open(INCIDENTS_DIR / "events.jsonl", "a", encoding="utf-8") as f:
        print(json.dumps(event, ensure_ascii=False), file=f, flush=True)

def inference_thread(cap, models, rules, display_queue, stop_event):
    """
    Main processing thread: grabs frames, runs inference, annotates, and queues for display.
    """
    print("Inference thread started.")
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3)
    
    frame_count = 0
    start_time = time.time()
    fps = 0

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed. Stopping inference thread.")
            break

        annotated_frame = frame.copy()
        
        all_dets = []
        all_labels = []

        # --- Run Inference on all models ---
        for model, (tag, conf_thr, keywords) in rules.items():
            results = model.predict(source=frame, conf=conf_thr, device=DEVICE, imgsz=IMG_SIZE, verbose=False)
            res = results[0]
            dets = sv.Detections.from_ultralytics(res)
            
            labels = []
            for i, xyxy in enumerate(dets.xyxy):
                cls_id = int(dets.class_id[i])
                cls_name = res.names.get(cls_id, str(cls_id))
                score = float(dets.confidence[i]) if dets.confidence is not None else 0.0
                labels.append(f"{tag}:{cls_name} {score:.2f}")

                if any(k.lower() in cls_name.lower() for k in keywords):
                    save_incident(frame, f"{tag}:{cls_name}", score, xyxy)
            
            if len(dets) > 0:
                all_dets.append(dets)
                all_labels.extend(labels)

        # --- Annotate Frame ---
        if all_dets:
            combined_dets = sv.Detections.merge(all_dets)
            annotated_frame = box_annotator.annotate(annotated_frame, combined_dets)
            annotated_frame = label_annotator.annotate(annotated_frame, combined_dets, labels=all_labels)

        # --- Calculate and Add FPS ---
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            start_time = time.time()
            frame_count = 0
        
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # --- Queue for Display ---
        if not display_queue.full():
            display_queue.put(annotated_frame)

    print("Inference thread finished.")

def main():
    # --- Load Models ---
    helmet_model = load_model(HELMET_WEIGHTS)
    smoke_model = load_model(SMOKE_WEIGHTS)
    phone_model = load_model(PHONE_WEIGHTS)
    
    models = {'helmet': helmet_model, 'smoke': smoke_model, 'phone': phone_model}
    if not any(models.values()):
        raise SystemExit("No models found. Place weights in ./models and rerun.")

    model_rules = {}
    if models['helmet']: model_rules[models['helmet']] = ("HELMET", CONF_HELMET, ["NO-Hardhat", "no-helmet"])
    if models['smoke']: model_rules[models['smoke']] = ("SMOKE", CONF_SMOKE, ["cigarette", "smoke"])
    if models['phone']: model_rules[models['phone']] = ("PHONE", CONF_PHONE, ["phone", "mobile"])

    # --- Open Camera ---
    print("Searching for camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("❌ Could not open webcam. Please check connection and drivers.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("✓ Camera opened successfully.")

    # --- Start Inference Thread ---
    proc_thread = threading.Thread(target=inference_thread, args=(cap, models, model_rules, DISPLAY_QUEUE, STOP_EVENT))
    proc_thread.start()

    # --- Main Display Loop ---
    window_name = "Webcam Safety Pilot - Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    print("Display loop started. Press 'Q' to quit.")

    while not STOP_EVENT.is_set():
        try:
            # Get the latest annotated frame, non-blocking
            annotated_frame = DISPLAY_QUEUE.get_nowait()
            cv2.imshow(window_name, annotated_frame)
        except queue.Empty:
            # No new frame, just process UI events
            time.sleep(0.001) 

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("Quit signal received.")
            STOP_EVENT.set()
            break
            
    # --- Cleanup ---
    print("Stopping threads...")
    proc_thread.join()
    print("Inference thread joined.")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Application finished.")

if __name__ == "__main__":
    main()
