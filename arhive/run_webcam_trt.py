# run_webcam_trt.py
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
CONF_HELMET_VEST = 0.40
CONF_SMOKE = 0.40
CONF_PHONE = 0.45
CONF_FIRE_SMOKE = 0.40
CONF_FALL = 0.40

# --- Threading and Queues ---
RAW_FRAME_QUEUE = queue.Queue(maxsize=2)
INFERENCE_QUEUES = {} # One queue per model
DISPLAY_QUEUE = queue.Queue(maxsize=1)
RESULTS_QUEUE = queue.Queue(maxsize=100) # For annotated frames
STOP_EVENT = threading.Event()

# --- Device Configuration ---
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == 0:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Running on GPU: {gpu_name}")
    torch.set_float32_matmul_precision("high")

# --- Model Definitions ---
MODEL_DEFINITIONS = {
    "helmet_vest": {
        "weights": MODELS_DIR / "helmet_vest_best.engine",
        "conf": CONF_HELMET_VEST,
        "keywords": ["NO-Hardhat", "no-helmet", "vest", "NO-Vest"]
    },
    "smoke": {
        "weights": MODELS_DIR / "smoke_best.engine",
        "conf": CONF_SMOKE,
        "keywords": ["cigarette", "smoke"]
    },
    "phone": {
        "weights": MODELS_DIR / "phone_best.engine",
        "conf": CONF_PHONE,
        "keywords": ["phone", "mobile"]
    },
    "fire_smoke": {
        "weights": MODELS_DIR / "fire_smoke_best.engine",
        "conf": CONF_FIRE_SMOKE,
        "keywords": ["fire", "smoke"]
    },
    "fall": {
        "weights": MODELS_DIR / "fall_best.engine",
        "conf": CONF_FALL,
        "keywords": ["fall", "falling", "man_down"]
    }
}

def load_model(model_name, model_info):
    weights = model_info["weights"]
    if not weights.exists():
        print(f"[WARN] Weights file not found, skipping: {weights}")
        return None
    try:
        # Specify task='detect' to suppress the warning
        model = YOLO(str(weights), task='detect')
        print(f"Loaded TRT model: {weights.name}")
        return model
    except Exception as e:
        print(f"[ERROR] Could not load {model_name}: {e}")
        return None

def save_incident(frame, label, conf, xyxy):
    ts_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    img_name = f"{ts_str}_{label.replace(':', '_')}_{int(conf*100)}.jpg"
    x1, y1, x2, y2 = map(int, xyxy)
    crop = frame[y1:y2, x1:x2]
    if crop.size > 0:
        cv2.imwrite(str(INCIDENTS_DIR / img_name), crop)

# --- THREADING FUNCTIONS ---

def camera_thread(cap, queue, stop_event):
    print("Camera thread started.")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed. Stopping.")
            break
        if not queue.full():
            queue.put(frame)
        else:
            queue.get() # Discard oldest frame to make space for the new one
            queue.put(frame)
    print("Camera thread finished.")

def frame_distributor_thread(raw_queue, inference_queues, display_queue, stop_event):
    print("Frame distributor thread started.")
    while not stop_event.is_set():
        try:
            frame = raw_queue.get(timeout=1)
            
            # Push to display queue
            if not display_queue.full():
                display_queue.put(frame)
            else:
                display_queue.get()
                display_queue.put(frame)

            # Push to all inference queues
            for q in inference_queues.values():
                if not q.full():
                    q.put(frame)
                else:
                    q.get()
                    q.put(frame)

        except queue.Empty:
            continue
    print("Frame distributor thread finished.")

def inference_worker(model_name, model, model_info, in_queue, out_queue, stop_event):
    tag = model_name.upper()
    conf_thr = model_info["conf"]
    keywords = model_info["keywords"]
    print(f"Inference worker for {tag} started.")

    while not stop_event.is_set():
        try:
            frame = in_queue.get(timeout=1)
            
            results = model.predict(source=frame, conf=conf_thr, device=DEVICE, verbose=False)
            res = results[0]
            dets = sv.Detections.from_ultralytics(res)
            
            labels = []
            for i in range(len(dets)):
                xyxy = dets.xyxy[i]
                conf = dets.confidence[i]
                cls_id = dets.class_id[i]

                cls_name = res.names.get(int(cls_id), f"ID:{cls_id}")
                label = f"{tag}:{cls_name} {conf:.2f}"
                labels.append(label)
                
                if any(k.lower() in cls_name.lower() for k in keywords):
                    save_incident(frame.copy(), f"{tag}-{cls_name}", conf, xyxy)
            
            if len(dets) > 0:
                out_queue.put((dets, labels))

        except queue.Empty:
            continue
    print(f"Inference worker for {tag} finished.")

# --- MAIN ---
def main():
    # --- Load Models ---
    models = {}
    for name, info in MODEL_DEFINITIONS.items():
        model = load_model(name, info)
        if model:
            models[name] = model
            INFERENCE_QUEUES[name] = queue.Queue(maxsize=1)

    if not models:
        raise SystemExit("No models could be loaded. Exiting.")

    # --- Open Camera ---
    print("Searching for camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("❌ Could not open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("✓ Camera opened successfully.")

    # --- Start Threads ---
    threads = []
    cam_thread = threading.Thread(target=camera_thread, args=(cap, RAW_FRAME_QUEUE, STOP_EVENT))
    threads.append(cam_thread)

    dist_thread = threading.Thread(target=frame_distributor_thread, args=(RAW_FRAME_QUEUE, INFERENCE_QUEUES, DISPLAY_QUEUE, STOP_EVENT))
    threads.append(dist_thread)

    for name, model in models.items():
        worker = threading.Thread(
            target=inference_worker,
            args=(name, model, MODEL_DEFINITIONS[name], INFERENCE_QUEUES[name], RESULTS_QUEUE, STOP_EVENT)
        )
        threads.append(worker)

    for t in threads:
        t.start()

    # --- Main Display Loop ---
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3)
    
    window_name = "Webcam Safety Pilot (TRT) - Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    fps_start_time = time.time()
    fps_frame_count = 0
    display_fps = 0

    print("Display loop started. Press 'Q' to quit.")
    while not STOP_EVENT.is_set():
        try:
            display_frame = DISPLAY_QUEUE.get(timeout=1)
            
            all_dets = []
            all_labels = []
            while not RESULTS_QUEUE.empty():
                try:
                    dets, labels = RESULTS_QUEUE.get_nowait()
                    all_dets.append(dets)
                    all_labels.extend(labels)
                except queue.Empty:
                    break
            
            annotated_frame = display_frame
            if all_dets:
                combined_dets = sv.Detections.merge(all_dets)
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=combined_dets)
                # The supervision annotator needs a list of labels that matches the combined detections
                # We must regenerate the labels list here for correct association
                final_labels = [item for sublist in [d[1] for d in zip(all_dets, [[l] * len(d[0]) for d, l in zip(all_dets, all_labels)])] for item in sublist]
                
                # Recreate labels based on the final merged detections to ensure correct mapping
                final_labels = []
                temp_label_idx = 0
                for dets in all_dets:
                    final_labels.extend(all_labels[temp_label_idx : temp_label_idx + len(dets)])
                    temp_label_idx += len(dets)

                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=combined_dets, labels=final_labels)

            # FPS Calculation
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                display_fps = fps_frame_count / (time.time() - fps_start_time)
                fps_frame_count = 0
                fps_start_time = time.time()

            fps_text = f"FPS: {display_fps:.1f}"
            cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(window_name, annotated_frame)

        except queue.Empty:
            time.sleep(0.01)
            continue

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            print("Quit signal received.")
            STOP_EVENT.set()
            break

    # --- Cleanup ---
    print("Stopping threads...")
    for t in threads:
        t.join()
    
    cap.release()
    cv2.destroyAllWindows()
    print("Application finished.")

if __name__ == "__main__":
    main()
