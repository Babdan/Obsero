"""
obsero.workers — Camera processes, model workers, router, composer, FanOut.

Key changes from baseline:
  • mp_infer_worker accepts gpu_id; calls torch.cuda.set_device.
  • FanOut cadence is per (camera_id, model_key) — not global.
  • Output tuples include gpu_id and model_key for DB/debug.
  • results_collector_thread integrates TemporalGate via GateManager.
  • Evidence: saves both full-frame snapshot and crop.
  • Camera offline detection via last-seen timestamps.
"""

from __future__ import annotations

import datetime, json, os, queue, signal, sys, threading, time, traceback
from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import multiprocessing as mp

# ── paths ──
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
INCIDENTS_DIR = ROOT / "incidents"
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640

# ═══════════════════ Helpers ═════════════════════════════════════════════════

def encode_jpeg(img, q=82) -> bytes | None:
    ok, b = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return b.tobytes() if ok else None


def make_no_signal_frame(w=640, h=360, label="NO SIGNAL"):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (18, 18, 24)
    for i in range(-h, w, 16):
        cv2.line(img, (i, 0), (i + h, h), (30, 30, 44), 1, cv2.LINE_AA)
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), (70, 70, 95), 2, cv2.LINE_AA)
    (tw, th2), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(img, label, ((w - tw) // 2, (h + th2) // 2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 140, 0), 2, cv2.LINE_AA)
    return img


# ═══════════════════ Model inference worker (multiprocessing) ════════════════

def mp_infer_worker(proc_name: str, stem: str, conf_thr: float,
                    keywords: list[str], gpu_id: int,
                    input_q: mp.Queue, output_q: mp.Queue):
    """
    One model worker per (model_key, gpu_id).
    Output tuple: (camera_id, tag, dets, names, gpu_id, model_key)
    """
    import torch
    from ultralytics import YOLO
    from obsero.models import resolve_weight_path_for_gpu

    # Prevent Ctrl+C from hard-aborting child processes on Windows.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    tag = proc_name.upper()
    model_key = proc_name.lower()

    # select device
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device_sel = gpu_id
        print(f"[{proc_name}] GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}", flush=True)
    else:
        device_sel = "cpu"
        print(f"[{proc_name}] CUDA unavailable, using CPU", flush=True)

    # resolve & load weights — TRT first, fallback to PT
    path, is_trt = resolve_weight_path_for_gpu(stem, gpu_id)
    if path is None:
        print(f"[{proc_name}] ERROR: no weights found for stem={stem}", flush=True)
        return

    try:
        model = YOLO(str(path), task="detect")
        print(f"[{proc_name}] loaded {'TRT' if is_trt else 'PT'}: {path.name}", flush=True)
    except Exception as e:
        # TRT engine may fail on incompatible GPU — fall back to .pt
        print(f"[{proc_name}] TRT load failed ({e}), trying .pt fallback …", flush=True)
        pt_path = MODELS_DIR / f"{stem}.pt"
        if pt_path.exists():
            try:
                model = YOLO(str(pt_path), task="detect")
                is_trt = False
                path = pt_path
                print(f"[{proc_name}] PT fallback OK: {pt_path.name}", flush=True)
            except Exception as e2:
                print(f"[{proc_name}] PT fallback also failed: {e2}", flush=True)
                return
        else:
            print(f"[{proc_name}] no .pt fallback available", flush=True)
            return

    with torch.inference_mode():
        while True:
            item = input_q.get()
            if item is None:
                break
            try:
                camera_id, blob = item
                arr = np.frombuffer(blob, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                reslist = model.predict(source=frame, conf=conf_thr,
                                        device=device_sel, imgsz=IMG_SIZE,
                                        half=is_trt, verbose=False)
                r0 = reslist[0]
                dets = []
                names = r0.names
                if r0.boxes is not None and r0.boxes.xyxy is not None:
                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    conf = r0.boxes.conf.cpu().numpy()
                    cls = r0.boxes.cls.cpu().numpy()
                    for i in range(xyxy.shape[0]):
                        dets.append([float(xyxy[i, 0]), float(xyxy[i, 1]),
                                     float(xyxy[i, 2]), float(xyxy[i, 3]),
                                     float(conf[i]), int(cls[i])])
                output_q.put((camera_id, tag, dets, names, gpu_id, model_key))
            except Exception as e:
                print(f"[{proc_name}] inference error: {e}", flush=True)
                traceback.print_exc(file=sys.stdout)


# ═══════════════════ FanOut — per-camera cadence ════════════════════════════

class FanOut:
    """
    Routes JPEG blobs from cameras to model input queues.
    Cadence is tracked per (camera_id, model_key) — not per global tick.

    register(model_key, gpu_id, q)  — model workers register their queues.
    camera_gpu_map: dict[camera_id -> gpu_id]  — set by bootstrap.
    """

    def __init__(self, cadence_map: dict[str, int]):
        self.cadence = cadence_map
        # key -> list of (gpu_id, mp.Queue)
        self._queues: dict[str, list[tuple[int, mp.Queue]]] = {}
        # (camera_id, model_key) -> counter
        self._counters: dict[tuple[int, str], int] = {}
        # camera_id -> gpu_id
        self.camera_gpu: dict[int, int] = {}

    def register(self, model_key: str, gpu_id: int, q: mp.Queue):
        self._queues.setdefault(model_key, []).append((gpu_id, q))

    def set_camera_gpu(self, cam_gpu: dict[int, int]):
        self.camera_gpu = cam_gpu

    def send(self, jpeg_bytes: bytes, camera_id: int):
        cam_gpu = self.camera_gpu.get(camera_id)
        for model_key, gpu_q_list in self._queues.items():
            ckey = (camera_id, model_key)
            c = self._counters.get(ckey, 0) + 1
            every = max(1, self.cadence.get(model_key, 1))
            if (c % every) == 0:
                # send ONLY to the worker on this camera's GPU
                for gid, q in gpu_q_list:
                    if cam_gpu is not None and gid != cam_gpu:
                        continue
                    try:
                        q.put_nowait((camera_id, jpeg_bytes))
                    except Exception:
                        pass  # bounded queue — drop frame on backpressure
            self._counters[ckey] = c if c < 10_000_000 else 0


# ═══════════════════ Camera process (multiprocessing) ════════════════════════

def camera_proc(camera_id: int, source, out_q: mp.Queue,
                max_fps=12, target_side=640):
    """One process per camera. Sends (camera_id, jpeg_bytes) to out_q."""

    # Main process handles Ctrl+C and then terminates camera processes cleanly.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                          "rtsp_transport;tcp|stimeout;5000000")

    def put_placeholder():
        h = int(target_side * 9 / 16)
        f = make_no_signal_frame(target_side, h)
        b = encode_jpeg(f, 82)
        if b:
            try:
                out_q.put_nowait((camera_id, b))
            except Exception:
                pass

    def open_capture(src):
        # Webcam indices should use DirectShow/ANY, not FFmpeg.
        src_is_index = isinstance(src, int) or (isinstance(src, str) and src.isdigit())
        if src_is_index:
            idx = int(src)
            if os.name == "nt":
                cap_local = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap_local.isOpened():
                    cap_local.release()
                    cap_local = cv2.VideoCapture(idx, cv2.CAP_MSMF)
                    if not cap_local.isOpened():
                        cap_local.release()
                        cap_local = cv2.VideoCapture(idx, cv2.CAP_ANY)
            else:
                cap_local = cv2.VideoCapture(idx, cv2.CAP_ANY)
            return cap_local
        cap_local = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        cap_local.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap_local

    backoff = 0.5
    while True:
        cap = open_capture(source)
        if not cap.isOpened():
            put_placeholder()
            time.sleep(min(3.0, backoff))
            backoff = min(5.0, backoff * 1.5)
            continue

        backoff = 0.5
        last = time.time()
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                put_placeholder()
                cap.release()
                time.sleep(0.5)
                break

            h, w = frame.shape[:2]
            scale = target_side / max(h, w)
            if scale < 1.0:
                frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                         interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame
            b = encode_jpeg(frame_small, 82)
            if b:
                try:
                    out_q.put_nowait((camera_id, b))
                except Exception:
                    pass  # drop frame — backpressure

            dt = time.time() - last
            tgt = 1.0 / max(1, max_fps)
            if dt < tgt:
                time.sleep(tgt - dt)
            last = time.time()


# ═══════════════════ Camera process manager ══════════════════════════════════

class MultiCamProcManager:
    def __init__(self, out_queue: mp.Queue):
        self._lock = threading.Lock()
        self._procs: dict[int, mp.Process] = {}
        self.sources: dict[int, object] = {}
        self.active_camera_id: int | None = None
        self.out_queue = out_queue

    def start_all(self, cam_map: dict[int, tuple], active_camera_id: int | None):
        """cam_map: {camera_id: (url, max_fps, target_side)}"""
        with self._lock:
            self.active_camera_id = active_camera_id
            for cid, (src, fps, side) in cam_map.items():
                if cid in self._procs and self._procs[cid].is_alive():
                    continue
                p = mp.Process(target=camera_proc,
                               args=(cid, src, self.out_queue, fps, side),
                               daemon=True)
                self._procs[cid] = p
                p.start()
            print(f"[cams] started {len(self._procs)} camera process(es)", flush=True)

    def switch_active(self, camera_id: int):
        with self._lock:
            self.active_camera_id = camera_id

    def restart_with(self, cam_map: dict[int, tuple], active_camera_id: int | None):
        self.stop_all()
        self.start_all(cam_map, active_camera_id)

    def stop_all(self):
        with self._lock:
            for cid, p in self._procs.items():
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1.5)
            self._procs.clear()


# ═══════════════════ Router thread ═══════════════════════════════════════════

def camera_router_thread(out_q: mp.Queue, fanout: FanOut,
                         raw_snaps: dict, raw_lock: threading.Lock,
                         display_q: queue.Queue,
                         get_active: Callable[[], int | None],
                         stop: threading.Event,
                         set_online_cb: Callable[[int], None],
                         last_seen: dict[int, float]):
    """
    Reads (camera_id, jpeg) from camera processes, stores raw snapshot,
    feeds fanout, pushes active camera frames to display queue.
    """
    seen_online: set[int] = set()
    while not stop.is_set():
        try:
            camera_id, jpeg_small = out_q.get(timeout=1)
        except Exception:
            continue

        # raw snapshot (unannotated)
        with raw_lock:
            raw_snaps[camera_id] = jpeg_small

        last_seen[camera_id] = time.time()

        # fan out to model workers
        fanout.send(jpeg_small, camera_id)

        # mark camera online (once)
        if camera_id not in seen_online:
            try:
                set_online_cb(camera_id)
            except Exception:
                pass
            seen_online.add(camera_id)

        # push raw frame to display queue for active camera
        active = get_active()
        if active == camera_id:
            arr = np.frombuffer(jpeg_small, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                if not display_q.empty():
                    try:
                        display_q.get_nowait()
                    except queue.Empty:
                        pass
                display_q.put(frame)


# ═══════════════════ Composer thread ════════════════════════════════════════

def composer_thread(stop: threading.Event,
                    get_active: Callable[[], int | None],
                    display_q: queue.Queue,
                    results_q: queue.Queue,
                    annotated_jpeg: dict,  # {"jpeg": bytes|None}
                    annotated_lock: threading.Lock,
                    annotated_snaps: dict[int, bytes],
                    snap_lock: threading.Lock):
    """Overlays detections on the active camera frame → annotated JPEG."""
    import supervision as sv
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3)
    fps_start = time.time()
    fps_count = 0
    fps_val = 0.0

    while not stop.is_set():
        try:
            frame = display_q.get(timeout=1)
        except queue.Empty:
            continue

        active_id = get_active()
        while True:
            try:
                cam_id, xyxy, conf, cls_ids, names, tag = results_q.get_nowait()
            except queue.Empty:
                break
            if cam_id != active_id or xyxy.shape[0] == 0:
                continue
            dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls_ids)
            frame = box_annotator.annotate(frame, dets)
            labels = [f"{tag}:{names.get(int(cid), str(int(cid)))} {float(cf):.2f}"
                      for cid, cf in zip(cls_ids, conf)]
            frame = label_annotator.annotate(frame, dets, labels=labels)

        # FPS overlay (top-right to avoid camera timestamp)
        fps_count += 1
        dt = time.time() - fps_start
        if dt >= 1.0:
            fps_val = fps_count / dt
            fps_count = 0
            fps_start = time.time()
        h, w = frame.shape[:2]
        fps_text = f"FPS: {fps_val:.1f}"
        (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(frame, fps_text, (w - tw - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 140, 0), 2, cv2.LINE_AA)

        b = encode_jpeg(frame, 80)
        if b:
            with annotated_lock:
                annotated_jpeg["jpeg"] = b
            aid = get_active()
            if aid is not None:
                with snap_lock:
                    annotated_snaps[aid] = b


# ═══════════════════ Results collector thread ════════════════════════════════

def results_collector_thread(mp_out_q: mp.Queue,
                             get_active: Callable[[], int | None],
                             raw_snaps: dict, raw_lock: threading.Lock,
                             results_q: queue.Queue,
                             gate_mgr,  # GateManager
                             model_keywords: dict[str, list[str]],
                             per_tag_cooldown: dict[str, float],
                             default_cooldown: float,
                             incidents_ring: deque,
                             stop: threading.Event):
    """
    Reads model outputs, applies temporal gating, saves evidence (crop + full).
    """
    from obsero.db import alert_insert

    last_emit: dict[tuple[str, int], float] = {}

    print("[results] collector started", flush=True)
    while not stop.is_set():
        try:
            camera_id, tag, dets, names, w_gpu_id, w_model_key = mp_out_q.get(timeout=1)
        except Exception:
            continue

        # Workers may emit names like "ppe_gpu0"; normalize to config key "ppe".
        model_key = (w_model_key or tag).lower()
        if "_gpu" in model_key:
            model_key = model_key.split("_gpu", 1)[0]
        event_tag = model_key.upper()

        # --- get raw snapshot for this camera ---
        with raw_lock:
            jpeg_bytes = raw_snaps.get(camera_id)
        if not jpeg_bytes:
            continue
        img_arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        keywords = model_keywords.get(model_key, [])
        positive_for_gate = False

        xyxy_list, conf_list, cls_list = [], [], []
        best_det = None  # (name, conf, xyxy) for incident saving

        for x1, y1, x2, y2, conf_val, clsid in dets:
            xyxy_list.append([x1, y1, x2, y2])
            conf_list.append(conf_val)
            cls_list.append(int(clsid))
            name = names.get(int(clsid), str(int(clsid)))
            if any(k.lower() in name.lower() for k in keywords):
                positive_for_gate = True
                if best_det is None or conf_val > best_det[1]:
                    best_det = (name, conf_val, [x1, y1, x2, y2])

        # --- temporal gate ---
        fired = gate_mgr.feed(camera_id, model_key, positive_for_gate)

        if fired and best_det is not None:
            det_name, det_conf, det_xyxy = best_det
            # additional per-tag cooldown check
            now = time.time()
            cd = per_tag_cooldown.get(event_tag, default_cooldown)
            emit_key = (event_tag, camera_id)
            if now - last_emit.get(emit_key, 0.0) >= cd:
                last_emit[emit_key] = now
                _save_incident_with_evidence(
                    frame, event_tag, det_name, det_conf, det_xyxy,
                    camera_id, w_gpu_id, model_key, gate_mgr, incidents_ring
                )

        # --- push detections for live view overlay ---
        active_id = get_active()
        if active_id == camera_id and xyxy_list:
            try:
                results_q.put_nowait((
                    camera_id,
                    np.array(xyxy_list, dtype=np.float32),
                    np.array(conf_list, dtype=np.float32),
                    np.array(cls_list, dtype=np.int32),
                    names, event_tag
                ))
            except queue.Full:
                pass


def _save_incident_with_evidence(frame, tag, name, score, xyxy,
                                 camera_id, gpu_id, model_key,
                                 gate_mgr, incidents_ring):
    """Save crop + full-frame images, insert alert into DB."""
    from obsero.db import alert_insert

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    x1, y1, x2, y2 = map(int, xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)

    safe_label = f"{tag}_{name}".replace(":", "_").replace(" ", "_")
    ts_safe = ts.replace(":", "-").replace(" ", "_")

    # crop image
    crop = frame[y1:y2, x1:x2]
    crop_name = f"{ts_safe}_{safe_label}_{int(score * 100)}_crop.jpg"
    if crop.size > 0:
        cv2.imwrite(str(INCIDENTS_DIR / crop_name), crop)

    # full-frame image (unannotated)
    full_name = f"{ts_safe}_{safe_label}_{int(score * 100)}_full.jpg"
    cv2.imwrite(str(INCIDENTS_DIR / full_name), frame)

    # gate snapshot for rule_json
    gate_snap = gate_mgr.snapshot(camera_id, tag.lower())
    rule_json = json.dumps(gate_snap)

    event = {"ts": ts, "label": f"{tag}:{name}", "conf": round(float(score), 3),
             "bbox": [x1, y1, x2, y2], "image": crop_name, "full_image": full_name}
    incidents_ring.appendleft(event)

    alert_insert(ts, camera_id, "medium", tag, f"{tag}:{name}",
                 float(score), [x1, y1, x2, y2], crop_name,
                 site="Site-A", status="new",
                 model_key=model_key, gpu_id=gpu_id,
                 rule_json=rule_json, full_image=full_name)


# ═══════════════════ Camera offline detection ════════════════════════════════

def camera_offline_monitor(last_seen: dict[int, float],
                           timeout_sec: float,
                           stop: threading.Event,
                           set_offline_cb: Callable[[int], None]):
    """Marks cameras offline if no frame received within timeout_sec."""
    known_offline: set[int] = set()
    while not stop.is_set():
        time.sleep(5.0)
        now = time.time()
        for cid, ts in list(last_seen.items()):
            if (now - ts) > timeout_sec:
                if cid not in known_offline:
                    print(f"[offline] camera {cid} offline (no frame for {timeout_sec:.0f}s)",
                          flush=True)
                    try:
                        set_offline_cb(cid)
                    except Exception:
                        pass
                    known_offline.add(cid)
            else:
                known_offline.discard(cid)
