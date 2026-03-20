# run_webcam_trt.py
# ---- Single-camera TRT demo (Dahua RTSP) to verify connectivity + run all algorithms ----
# - Hard-coded RTSP (change SOURCE below)
# - One capture thread, one worker per model, draw everything in a single OpenCV window
# - Saves cropped incident images to ./incidents when keywords match (cooled down per tag)

import os, time, json, datetime, threading, queue
from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv

# ===================== USER: set your Dahua RTSP here =====================
# Works the same as your successful ffplay command.
SOURCE = "rtsp://admin:Petra123!@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0&unicast=true"
# Optional: test other channels quickly by changing channel=X and/or subtype=1 (substream).
# ==========================================================================

# Force OpenCV/FFmpeg to use TCP for RTSP (helps on some networks)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ----------------- Paths & constants -----------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
INCIDENTS_DIR = ROOT / "incidents"
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640
CONF_PPE   = 0.40
CONF_SMOKE = 0.40
CONF_PHONE = 0.45
CONF_FIRE  = 0.40
CONF_FALL  = 0.40

# Per-model cooldown (seconds) for saving incidents
DEFAULT_COOLDOWN = 4.0
PER_TAG_COOLDOWN = {"FIRE_SMOKE": 6.0}

# --- Threading and Queues ---
RAW_FRAME_QUEUE   = queue.Queue(maxsize=2)
INFERENCE_QUEUES  = {}                 # one queue per model
DISPLAY_QUEUE     = queue.Queue(maxsize=1)
RESULTS_QUEUE     = queue.Queue(maxsize=200)  # holds (sv.Detections, labels)
STOP_EVENT        = threading.Event()

# --- Device Configuration ---
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == 0:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# --- Model set (TRT-first, .pt fallback) ---
def w(stem):  # helper to prefer .engine then .pt
    e = MODELS_DIR / f"{stem}.engine"
    p = MODELS_DIR / f"{stem}.pt"
    return e if e.exists() else (p if p.exists() else None)

MODEL_DEFS = {
    "ppe":        {"weights": w("helmet_vest_best"), "conf": CONF_PPE,   "keywords": ["nohat","novest","NO-Hardhat","NO-Vest"]},
    "smoke":      {"weights": w("smoke_best"),       "conf": CONF_SMOKE, "keywords": ["cigarette","smoke"]},
    "phone":      {"weights": w("phone_best"),       "conf": CONF_PHONE, "keywords": ["phone","mobile","smartphone"]},
    "fire_smoke": {"weights": w("fire_smoke_best"),  "conf": CONF_FIRE,  "keywords": ["fire","smoke"]},
    "fall":       {"weights": w("fall_best"),        "conf": CONF_FALL,  "keywords": ["fall","fallen","falldown","man_down"]},
}

# --- Load a model (TRT or PT) ---
def load_model(tag, info):
    path = info["weights"]
    if path is None:
        print(f"[{tag}] ⚠️ weights not found (.engine or .pt) under ./models – skipping")
        return None, False
    try:
        m = YOLO(str(path), task="detect")
        is_trt = path.suffix.lower() == ".engine"
        print(f"[{tag}] loaded {'TRT' if is_trt else 'PT'}: {path.name}")
        return m, is_trt
    except Exception as e:
        print(f"[{tag}] ❌ load error: {e}")
        return None, False

# --- Incident saving with cooldown per (tag) ---
_LAST_EMIT = {}  # tag -> t_last
def save_incident(frame, tag, name, conf, xyxy):
    now = time.time()
    cd = PER_TAG_COOLDOWN.get(tag, DEFAULT_COOLDOWN)
    if now - _LAST_EMIT.get(tag, 0.0) < cd:
        return
    _LAST_EMIT[tag] = now
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    x1, y1, x2, y2 = map(int, xyxy)
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = max(x1+1,x2), max(y1+1,y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size > 0:
        safe = f"{tag}_{name}".replace(":","_").replace("/","_")
        fn = INCIDENTS_DIR / f"{ts}_{safe}_{int(conf*100)}.jpg"
        cv2.imwrite(str(fn), crop)
        print(f"[incident] saved {fn.name}")

# ---------------- Threads ----------------
def camera_thread(source, raw_q, stop_event):
    print("[cam] thread starting…")
    backoff = 0.5
    while not stop_event.is_set():
        # Use FFMPEG backend for RTSP
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print(f"[cam] open failed, retry in {backoff:.1f}s")
            time.sleep(backoff); backoff = min(5.0, backoff*1.5)
            continue
        print("[cam] opened OK")
        backoff = 0.5
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[cam] read failed – reopening…")
                cap.release()
                break
            # push the newest frame (drop oldest)
            if raw_q.full():
                try: raw_q.get_nowait()
                except queue.Empty: pass
            raw_q.put(frame)
    print("[cam] thread exit")

def distributor_thread(raw_q, inf_qs, display_q, stop_event):
    print("[dist] thread starting…")
    while not stop_event.is_set():
        try:
            frame = raw_q.get(timeout=1)
        except queue.Empty:
            continue
        # display queue
        if display_q.full():
            try: display_q.get_nowait()
            except queue.Empty: pass
        display_q.put(frame)
        # fan out to each model queue
        for q in inf_qs.values():
            if q.full():
                try: q.get_nowait()
                except queue.Empty: pass
            q.put(frame)
    print("[dist] thread exit")

def inference_worker(tag, model, is_trt, conf_thr, keywords, in_q, out_q, stop_event):
    TAG = tag.upper()
    print(f"[{TAG}] worker started (TRT={is_trt})")
    with torch.inference_mode():
        while not stop_event.is_set():
            try:
                frame = in_q.get(timeout=1)
            except queue.Empty:
                continue
            try:
                reslist = model.predict(source=frame, conf=conf_thr, device=DEVICE,
                                        imgsz=IMG_SIZE, half=is_trt, verbose=False)
                r0 = reslist[0]
                dets = sv.Detections.from_ultralytics(r0)

                # create labels + save incidents on keyword match
                labels = []
                for i, xyxy in enumerate(dets.xyxy):
                    cls_id = int(dets.class_id[i])
                    name   = r0.names.get(cls_id, str(cls_id))
                    conf   = float(dets.confidence[i]) if dets.confidence is not None else 0.0
                    labels.append(f"{TAG}:{name} {conf:.2f}")
                    if any(k.lower() in name.lower() for k in keywords):
                        save_incident(frame, TAG, name, conf, xyxy)
                if len(dets) > 0:
                    try: out_q.put_nowait((dets, labels))
                    except queue.Full: pass
            except Exception as e:
                print(f"[{TAG}] error: {e}")
    print(f"[{TAG}] worker exit")

# ---------------- Main ----------------
def main():
    # 1) Load models
    models = {}
    for tag, info in MODEL_DEFS.items():
        m, is_trt = load_model(tag, info)
        if m is None: 
            continue
        models[tag] = (m, is_trt, info["conf"], info["keywords"])
        INFERENCE_QUEUES[tag] = queue.Queue(maxsize=1)

    if not models:
        raise SystemExit("No models available. Put *.engine (or *.pt) into ./models")

    # 2) Start threads
    t_cam  = threading.Thread(target=camera_thread,     args=(SOURCE, RAW_FRAME_QUEUE, STOP_EVENT), daemon=True)
    t_dist = threading.Thread(target=distributor_thread, args=(RAW_FRAME_QUEUE, INFERENCE_QUEUES, DISPLAY_QUEUE, STOP_EVENT), daemon=True)
    t_cam.start(); t_dist.start()

    workers = []
    for tag, (m, is_trt, conf_thr, keywords) in models.items():
        tw = threading.Thread(target=inference_worker,
                              args=(tag, m, is_trt, conf_thr, keywords, INFERENCE_QUEUES[tag], RESULTS_QUEUE, STOP_EVENT),
                              daemon=True)
        workers.append(tw); tw.start()

    # 3) Display loop
    box_annotator   = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3)
    win = "RTSP Single-Cam TRT Demo (press Q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    fps_t0, fps_n, fps_val = time.time(), 0, 0.0
    print("[ui] running – waiting for frames…")
    while not STOP_EVENT.is_set():
        try:
            frame = DISPLAY_QUEUE.get(timeout=2)
        except queue.Empty:
            continue

        # drain results; merge
        det_list, label_list = [], []
        while True:
            try:
                dets, labels = RESULTS_QUEUE.get_nowait()
                det_list.append(dets); label_list.extend(labels)
            except queue.Empty:
                break

        if det_list:
            merged = sv.Detections.merge(det_list)
            frame  = box_annotator.annotate(frame, merged)
            # Ensure label count == number of boxes
            if len(label_list) != len(merged):
                # best-effort: trim or pad with class ids
                if len(label_list) > len(merged):
                    label_list = label_list[:len(merged)]
                else:
                    label_list += ["obj"] * (len(merged) - len(label_list))
            frame  = label_annotator.annotate(frame, merged, labels=label_list)

        # FPS overlay
        fps_n += 1
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps_val = fps_n / dt
            fps_n = 0; fps_t0 = time.time()
        cv2.putText(frame, f"FPS: {fps_val:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,140,0), 2)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            STOP_EVENT.set()
            break
        elif key in (ord('s'), ord('S')):
            # Save a full-frame snapshot for debugging
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fn = INCIDENTS_DIR / f"snapshot_{ts}.jpg"
            cv2.imwrite(str(fn), frame)
            print(f"[ui] snapshot saved: {fn.name}")

    cv2.destroyAllWindows()
    print("[main] exiting…")

if __name__ == "__main__":
    main()
