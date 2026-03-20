# multicam_panel_trt.py — TRT-first multi-camera inference + FastAPI control-room panel
# Product: Obsero Safety Panel
#
# Stable on Windows:
# - Uvicorn runs in MAIN thread (no signal issues)
# - Backend (models/cameras/router/composer) boots in a worker thread
# - One process per camera; one process per model
# - Robust camera reopen on unplug; online/offline in DB
# - i18n + tabs (Live / Multi-cam) + favicon route
# - GPU% / VRAM% / GPU temp / CPU temp KPIs with safe fallbacks
#
# Run:
#   python multicam_panel_trt.py --source 0 --port 9009

import argparse, json, datetime, time, threading, queue, sqlite3, sys, traceback, os, signal
from collections import deque
from pathlib import Path
import webbrowser
import multiprocessing as mp

import cv2
from ultralytics import YOLO
import supervision as sv
import psutil
import numpy as np

# ---- NVML (GPU stats) ----
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

from fastapi import FastAPI, Response, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ----------------- Paths & constants -----------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
INCIDENTS_DIR = ROOT / "incidents"
DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "static"
DATA_DIR.mkdir(exist_ok=True)
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "panel.db"

IMG_SIZE = 640
CONF_PPE   = 0.60
CONF_SMOKE = 0.60
CONF_PHONE = 0.60
CONF_FIRE  = 0.75
CONF_FALL  = 0.85

CADENCE = {"ppe": 1, "smoke": 2, "phone": 2, "fire_smoke": 3, "fall": 2}
DEFAULT_COOLDOWN = 5.0
PER_TAG_COOLDOWN = {"FIRE_SMOKE": 8.0}

STOP_EVENT = threading.Event()

LAST_JPEG = None
LAST_JPEG_LOCK = threading.Lock()

LAST_SNAPSHOTS = {}          # camera_id -> jpeg bytes
LAST_SNAPSHOTS_LOCK = threading.Lock()

DISPLAY_FRAME_QUEUE = queue.Queue(maxsize=1)
RESULTS_QUEUE = queue.Queue(maxsize=800)

INCIDENTS_RING = deque(maxlen=200)

# Multiprocessing queues/handles (created in main before spawn)
CAMERA_OUT = None            # mp.Queue: camera processes -> main (camera_id, jpeg_small)
MP_INPUTS = {}               # model_key -> (mp.Queue, mp.Process)
MP_OUTPUT = None             # model procs -> main (camera_id, tag, dets, names)

MODEL_SET = {
    "ppe":         ("helmet_vest_best", CONF_PPE,   ["nohat", "novest"]),
    "smoke":       ("smoke_best",       CONF_SMOKE, ["cigarette", "smoke"]),
    "phone":       ("phone_best",       CONF_PHONE, ["phone", "mobile", "smartphone"]),
    "fire_smoke":  ("fire_smoke_best",  CONF_FIRE,  ["fire", "smoke"]),
    "fall":        ("fall_best",        CONF_FALL,  ["fall", "fallen", "falldown", "man_down"]),
}

# ------------ Database ------------
def db_conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def db_init():
    con = db_conn(); cur = con.cursor()
    cur.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS cameras (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        code TEXT UNIQUE,
        url TEXT NOT NULL,
        online INTEGER DEFAULT 0,
        ptz_protocol TEXT DEFAULT 'onvif',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        camera_id INTEGER,
        level TEXT DEFAULT 'medium',
        type TEXT,
        label TEXT,
        conf REAL,
        bbox TEXT,
        image TEXT,
        site TEXT,
        status TEXT DEFAULT 'new',
        reviewer TEXT,
        reviewed_at TEXT,
        FOREIGN KEY(camera_id) REFERENCES cameras(id)
    );
    CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts);
    CREATE INDEX IF NOT EXISTS idx_alerts_cam ON alerts(camera_id);
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        role TEXT
    );
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT DEFAULT CURRENT_TIMESTAMP,
        actor TEXT,
        action TEXT,
        details TEXT
    );
    CREATE TABLE IF NOT EXISTS patrol_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        camera_ids TEXT NOT NULL,
        patrol_type TEXT,
        frequency TEXT,
        next_run TEXT,
        last_report TEXT,
        active INTEGER DEFAULT 1
    );
    CREATE TABLE IF NOT EXISTS alarm_levels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT UNIQUE,
        level TEXT
    );
    """)
    con.commit(); con.close()

db_init()

def audit(actor, action, details=""):
    con = db_conn(); con.execute(
        "INSERT INTO audit_logs(actor,action,details) VALUES (?,?,?)",
        (actor, action, details))
    con.commit(); con.close()

# ------------ Cameras ------------
def cameras_all():
    con = db_conn(); rows = con.execute("SELECT * FROM cameras ORDER BY id").fetchall(); con.close()
    return [dict(r) for r in rows]

def camera_by_id(cid):
    con = db_conn(); r = con.execute("SELECT * FROM cameras WHERE id=?", (cid,)).fetchone(); con.close()
    return dict(r) if r else None

def camera_upsert(name, url, code=None, ptz_protocol='onvif', cid=None):
    con = db_conn()
    if cid is None:
        con.execute("INSERT INTO cameras(name, code, url, ptz_protocol) VALUES (?,?,?,?)",
                    (name, code, url, ptz_protocol))
    else:
        con.execute("UPDATE cameras SET name=?, code=?, url=?, ptz_protocol=? WHERE id=?",
                    (name, code, url, ptz_protocol, cid))
    con.commit(); con.close()

def camera_set_online(cid, online: bool):
    con = db_conn()
    con.execute("UPDATE cameras SET online=? WHERE id=?", (1 if online else 0, cid))
    con.commit(); con.close()

# ------------ Alerts ------------
def alert_insert(ts, camera_id, level, etype, label, conf, bbox, image, site=None, status='new'):
    con = db_conn()
    con.execute("""INSERT INTO alerts(ts,camera_id,level,type,label,conf,bbox,image,site,status)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (ts, camera_id, level, etype, label, conf, json.dumps(bbox), image, site, status))
    con.commit(); con.close()

def alert_query(filters):
    sql = "SELECT * FROM alerts WHERE 1=1"
    args = []
    if filters.get('camera_id'):
        sql += " AND camera_id=?"; args.append(int(filters['camera_id']))
    if filters.get('level'):
        sql += " AND level=?"; args.append(filters['level'])
    if filters.get('type'):
        sql += " AND type=?"; args.append(filters['type'])
    if filters.get('t_from'):
        sql += " AND ts>=?"; args.append(filters['t_from'])
    if filters.get('t_to'):
        sql += " AND ts<=?"; args.append(filters['t_to'])
    sql += " ORDER BY ts DESC LIMIT 500"
    con = db_conn(); rows = con.execute(sql, args).fetchall(); con.close()
    return [dict(r) for r in rows]

def alert_update_status(aid, status, reviewer=None):
    con = db_conn()
    con.execute("UPDATE alerts SET status=?, reviewer=?, reviewed_at=CURRENT_TIMESTAMP WHERE id=?",
                (status, reviewer, aid))
    con.commit(); con.close()

# ------------ Health (CPU/GPU) ------------
def _cpu_temp_c():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        candidates = []
        for _, entries in temps.items():
            for ent in entries:
                label = (ent.label or "").lower()
                if any(k in label for k in ["package", "tdie", "tctl", "cpu"]):
                    if ent.current is not None:
                        candidates.append(ent.current)
                elif ent.current is not None:
                    candidates.append(ent.current)
        if not candidates:
            return None
        return float(max(candidates))
    except Exception:
        return None

def get_health_snapshot():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    mem_percent = mem.percent
    cpu_temp = _cpu_temp_c()

    gpu_name = "N/A"
    gpu_util = None
    vram_percent = None
    vram_used_mib = None
    vram_total_mib = None
    gpu_temp = None

    if NVML_OK:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(h).decode("utf-8") if hasattr(pynvml.nvmlDeviceGetName(h), 'decode') else str(pynvml.nvmlDeviceGetName(h))
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpu_util = int(u.gpu)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(h)
            vram_used_mib = int(meminfo.used / (1024*1024))
            vram_total_mib = int(meminfo.total / (1024*1024))
            if vram_total_mib > 0:
                vram_percent = round(100.0 * vram_used_mib / vram_total_mib, 1)
            gpu_temp = int(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass

    return dict(
        cpu=cpu,
        mem=mem_percent,
        cpu_temp=cpu_temp,
        gpu=gpu_name,
        gpu_util=gpu_util,
        vram_percent=vram_percent,
        vram_used_mib=vram_used_mib,
        vram_total_mib=vram_total_mib,
        gpu_temp=gpu_temp
    )

# ------------ Weights ------------
def resolve_weight_path(stem: str):
    e = MODELS_DIR / f"{stem}.engine"
    p = MODELS_DIR / f"{stem}.pt"
    if e.exists(): return e, True
    if p.exists(): return p, False
    return None, None

# ------------ Incidents ------------
LAST_EMIT_TAGCAM = {}  # (tag,camera_id)->last_time

def save_incident_and_alert(frame, tag, name, score, xyxy, camera_id=None, site=None, level="medium"):
    now = time.time()
    cd = PER_TAG_COOLDOWN.get(tag, DEFAULT_COOLDOWN)
    k = (tag, camera_id)
    if now - LAST_EMIT_TAGCAM.get(k, 0.0) < cd:
        return
    LAST_EMIT_TAGCAM[k] = now

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    x1, y1, x2, y2 = map(int, xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
    crop = frame[y1:y2, x1:x2]
    safe_label = f"{tag}:{name}".replace(':','_')
    img_name = f"{ts.replace(':','-').replace(' ','_')}_{safe_label}_{int(score*100)}.jpg"
    if crop.size > 0:
        cv2.imwrite(str(INCIDENTS_DIR / img_name), crop)
    event = {"ts": ts, "label": f"{tag}:{name}", "conf": round(float(score),3), "bbox": [x1,y1,x2,y2], "image": img_name}
    INCIDENTS_RING.appendleft(event)
    alert_insert(ts, camera_id, level, tag, f"{tag}:{name}", float(score), [x1,y1,x2,y2], img_name, site=site, status='new')

# ------------ MP: model workers ------------
def mp_infer_worker(proc_name, stem, conf_thr, keywords, input_q: mp.Queue, output_q: mp.Queue):
    import torch
    tag = proc_name.upper()
    device_sel = 0 if torch.cuda.is_available() else "cpu"
    try:
        path, is_trt = resolve_weight_path(stem)
        if path is None:
            print(f"[{proc_name}] missing weights: {stem}", flush=True)
            return
        model = YOLO(str(path), task="detect")
        print(f"[{proc_name}] loaded {'TRT' if is_trt else 'PT'}: {path.name}", flush=True)
    except Exception as e:
        print(f"[{proc_name}] load error: {e}", flush=True)
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
                reslist = model.predict(source=frame, conf=conf_thr, device=device_sel, imgsz=IMG_SIZE,
                                        half=(path.suffix=='.engine'), verbose=False)
                r0 = reslist[0]
                dets = []
                names = r0.names
                if r0.boxes is not None and r0.boxes.xyxy is not None:
                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    conf = r0.boxes.conf.cpu().numpy()
                    cls  = r0.boxes.cls.cpu().numpy()
                    for i in range(xyxy.shape[0]):
                        dets.append([float(xyxy[i,0]), float(xyxy[i,1]), float(xyxy[i,2]), float(xyxy[i,3]),
                                     float(conf[i]), int(cls[i])])
                output_q.put((camera_id, tag, dets, names))
            except Exception as e:
                print(f"[{proc_name}] inference error: {e}", flush=True)
                traceback.print_exc(file=sys.stdout)

class FanOut:
    def __init__(self, cadence_map: dict[str, int]):
        self.inputs: dict[str, tuple[mp.Queue, int]] = {}
        self.cadence = cadence_map

    def register(self, key: str, q: mp.Queue):
        self.inputs[key] = (q, 0)

    def send(self, jpeg_bytes: bytes, camera_id: int):
        for key, (q, c) in list(self.inputs.items()):
            c += 1
            every = max(1, self.cadence.get(key, 1))
            if (c % every) == 0:
                try: q.put_nowait((camera_id, jpeg_bytes))
                except Exception: pass
                c = 0 if c > 10_000_000 else c
            self.inputs[key] = (q, c)

FANOUT = None

def results_collector_thread(mp_out_q: mp.Queue, get_active_cam_id):
    print("[results] collector started", flush=True)
    while not STOP_EVENT.is_set():
        try:
            camera_id, tag, dets, names = mp_out_q.get(timeout=1)
        except Exception:
            continue
        if not dets:
            continue

        with LAST_SNAPSHOTS_LOCK:
            jpeg_bytes = LAST_SNAPSHOTS.get(camera_id)
        if not jpeg_bytes:
            continue
        img_arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        xyxy_list, conf_list, cls_list = [], [], []
        for x1, y1, x2, y2, conf, clsid in dets:
            xyxy_list.append([x1, y1, x2, y2])
            conf_list.append(conf)
            cls_list.append(int(clsid))
            try:
                name = names.get(int(clsid), str(int(clsid)))
                keywords = dict(MODEL_SET).get(tag.lower(), (None,None,[]))[2]
                if any(k in name.lower() for k in [k.lower() for k in keywords]):
                    save_incident_and_alert(frame, tag, name, conf, [x1,y1,x2,y2], camera_id=camera_id, site="Site-A")
            except Exception:
                pass

        active_id = get_active_cam_id()
        if active_id == camera_id and xyxy_list:
            try:
                RESULTS_QUEUE.put_nowait((camera_id,
                                          np.array(xyxy_list, dtype=np.float32),
                                          np.array(conf_list, dtype=np.float32),
                                          np.array(cls_list, dtype=np.int32),
                                          names, tag))
            except queue.Full:
                pass

# ------------ Camera process ------------
def camera_proc(camera_id: int, source, out_q: mp.Queue, max_fps=12, target_side=640, usb_mjpg=True):
    def try_open_backends(idx):
        print(f"[cam{camera_id}] opening index {idx} (MSMF→DSHOW→AUTO)", flush=True)
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if cap.isOpened(): return cap
        cap.release()
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened(): return cap
        cap.release()
        return cv2.VideoCapture(idx)

    def open_cap(src):
        if isinstance(src, int):
            cap = try_open_backends(src)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, max_fps)
            if usb_mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        else:
            print(f"[cam{camera_id}] opening stream {src}", flush=True)
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    backoff = 0.5
    online_ping = False
    while True:
        cap = open_cap(source)
        if not cap.isOpened():
            print(f"[cam{camera_id}] open failed, retry in {backoff:.1f}s", flush=True)
            time.sleep(backoff)
            backoff = min(5.0, backoff * 1.5)
            continue

        print(f"[cam{camera_id}] opened OK", flush=True)
        last = time.time()
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[cam{camera_id}] read failed (unplug/stream hiccup). Reopening...", flush=True)
                cap.release()
                time.sleep(backoff)
                backoff = min(5.0, backoff * 1.5)
                online_ping = False
                break

            h, w = frame.shape[:2]
            scale = target_side / max(h, w)
            if scale < 1.0:
                frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame
            ok2, blob = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok2:
                try:
                    out_q.put_nowait((camera_id, blob.tobytes()))
                except Exception:
                    pass
                if not online_ping:
                    # ensure router sees at least one snapshot promptly
                    try: out_q.put_nowait((camera_id, blob.tobytes()))
                    except Exception: pass
                    online_ping = True

            now = time.time()
            dt = now - last
            tgt = 1.0 / max(1, max_fps)
            if dt < tgt:
                time.sleep(tgt - dt)
            last = time.time()

# ------------ Camera manager ------------
class MultiCamProcManager:
    def __init__(self, out_queue: mp.Queue):
        self._lock = threading.Lock()
        self._procs: dict[int, mp.Process] = {}
        self.sources: dict[int, object] = {}
        self.active_camera_id: int | None = None
        self.out_queue = out_queue
        self.max_fps = 12
        self.target_side = 640
        self.usb_mjpg = True

    def start_all(self, cam_map: dict[int, object], active_camera_id: int | None):
        with self._lock:
            self.sources = cam_map.copy()
            self.active_camera_id = active_camera_id
            for cid, src in cam_map.items():
                if cid in self._procs and self._procs[cid].is_alive():
                    continue
                p = mp.Process(target=camera_proc,
                               args=(cid, src, self.out_queue, self.max_fps, self.target_side, self.usb_mjpg),
                               daemon=True)
                self._procs[cid] = p
                p.start()
            print(f"[cams] started {len(self._procs)} camera process(es)", flush=True)

    def switch_active(self, camera_id: int):
        with self._lock:
            self.active_camera_id = camera_id

    def stop_all(self):
        with self._lock:
            for cid, p in self._procs.items():
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1.5)
            self._procs.clear()

CAM_MGR: MultiCamProcManager | None = None

# ------------ Router & composer ------------
def camera_router_thread(out_q: mp.Queue, get_active_cam_id):
    print("[router] camera router started", flush=True)
    seen_online = set()
    while not STOP_EVENT.is_set():
        try:
            camera_id, jpeg_small = out_q.get(timeout=1)
        except Exception:
            continue

        with LAST_SNAPSHOTS_LOCK:
            LAST_SNAPSHOTS[camera_id] = jpeg_small

        if FANOUT is not None:
            FANOUT.send(jpeg_small, camera_id)

        if camera_id not in seen_online:
            try: camera_set_online(camera_id, True)
            except Exception: pass
            seen_online.add(camera_id)

        active = get_active_cam_id()
        if active == camera_id:
            arr = np.frombuffer(jpeg_small, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                if not DISPLAY_FRAME_QUEUE.empty():
                    try: DISPLAY_FRAME_QUEUE.get_nowait()
                    except queue.Empty: pass
                DISPLAY_FRAME_QUEUE.put(frame)

def composer_thread(stop: threading.Event, get_active_cam_id):
    print("[composer] started", flush=True)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=3)
    fps_start = time.time(); fps_count = 0; fps_val = 0.0

    while not stop.is_set():
        try:
            frame = DISPLAY_FRAME_QUEUE.get(timeout=1)
        except queue.Empty:
            continue

        active_id = get_active_cam_id()
        while True:
            try:
                cam_id, xyxy, conf, cls_ids, names, tag = RESULTS_QUEUE.get_nowait()
            except queue.Empty:
                break
            if cam_id != active_id or xyxy.shape[0] == 0:
                continue
            dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls_ids)
            frame = box_annotator.annotate(frame, dets)
            labels = [f"{tag}:{names.get(int(cid), str(int(cid)))} {float(cf):.2f}"
                      for cid, cf in zip(cls_ids, conf)]
            frame = label_annotator.annotate(frame, dets, labels=labels)

        fps_count += 1
        dt = time.time() - fps_start
        if dt >= 1.0:
            fps_val = fps_count / dt; fps_count = 0; fps_start = time.time()
        cv2.putText(frame, f"FPS: {fps_val:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,140,0),2)
        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with LAST_JPEG_LOCK:
                global LAST_JPEG
                LAST_JPEG = jpeg.tobytes()
            aid = get_active_cam_id()
            if aid is not None:
                with LAST_SNAPSHOTS_LOCK:
                    LAST_SNAPSHOTS[aid] = LAST_JPEG
    print("[composer] exit", flush=True)

# --------- FastAPI app & static ---------
app = FastAPI()
app.mount("/incidents", StaticFiles(directory=str(INCIDENTS_DIR)), name="incidents")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

FALLBACK_FAVICON_SVG = b"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
<rect width='64' height='64' fill='#0b0f14'/><path d='M36 6L14 36h14l-6 22 28-36H36z' fill='#ff7a1a'/>
</svg>"""

@app.get("./static/favicon.ico")
def favicon():
    ico_path = STATIC_DIR / "favicon.ico"
    if ico_path.exists():
        return FileResponse(str(ico_path), media_type="image/x-icon")
    return Response(content=FALLBACK_FAVICON_SVG, media_type="image/svg+xml")

# ---------- i18n ----------
I18N = {
    "en": {
        "title": "Obsero Safety Panel",
        "tab_live": "Live View",
        "tab_multi": "Multi-cam",
        "switch": "Switch to Camera",
        "overall": "Overview",
        "sites": "Sites",
        "hosts": "Analyzer Hosts",
        "devices": "Devices",
        "online": "online",
        "offline": "offline",
        "cpu": "CPU",
        "ram": "RAM",
        "gpu": "GPU",
        "gpu_util": "GPU Util",
        "vram": "VRAM",
        "gpu_temp": "GPU Temp",
        "cpu_temp": "CPU Temp",
        "realtime_alerts": "Real-time Alerts",
        "grid_title": "Multi-view (snapshots, refresh every 2s)",
        "current_cam": "Active camera",
        "view": "view",
        "time": "Time",
        "event": "Event",
        "level": "Level",
        "image": "Image",
        "lang": "Language",
        "english": "English",
        "chinese": "中文",
        "turkish": "Türkçe",
        "default_cam": "Default Camera"
    },
    "zh": {
        "title": "Obsero 安全面板",
        "tab_live": "实时画面",
        "tab_multi": "多路画面",
        "switch": "切换到该摄像头",
        "overall": "总体状态",
        "sites": "场站",
        "hosts": "分析主机",
        "devices": "设备",
        "online": "在线",
        "offline": "离线",
        "cpu": "CPU",
        "ram": "内存",
        "gpu": "显卡",
        "gpu_util": "显卡占用",
        "vram": "显存",
        "gpu_temp": "显卡温度",
        "cpu_temp": "CPU温度",
        "realtime_alerts": "实时告警",
        "grid_title": "分屏（快照，2秒刷新）",
        "current_cam": "当前摄像头",
        "view": "查看",
        "time": "时间",
        "event": "事件",
        "level": "级别",
        "image": "图像",
        "lang": "语言",
        "english": "English",
        "chinese": "中文",
        "turkish": "Türkçe",
        "default_cam": "默认相机"
    },
    "tr": {
        "title": "Obsero Güvenlik Paneli",
        "tab_live": "Canlı Görüntü",
        "tab_multi": "Çoklu Kamera",
        "switch": "Kameraya Geç",
        "overall": "Genel Durum",
        "sites": "Saha",
        "hosts": "Analiz Sunucuları",
        "devices": "Cihazlar",
        "online": "çevrimiçi",
        "offline": "çevrimdışı",
        "cpu": "CPU",
        "ram": "RAM",
        "gpu": "GPU",
        "gpu_util": "GPU Kullanımı",
        "vram": "VRAM",
        "gpu_temp": "GPU Sıcaklığı",
        "cpu_temp": "CPU Sıcaklığı",
        "realtime_alerts": "Anlık Alarmlar",
        "grid_title": "Çoklu Görünüm (anlık görüntü, 2 sn)",
        "current_cam": "Aktif kamera",
        "view": "gör",
        "time": "Zaman",
        "event": "Olay",
        "level": "Seviye",
        "image": "Görüntü",
        "lang": "Dil",
        "english": "English",
        "chinese": "中文",
        "turkish": "Türkçe",
        "default_cam": "Varsayılan Kamera"
    }
}

@app.get("/api/i18n")
def api_i18n(lang: str = "en"):
    return JSONResponse(I18N.get(lang, I18N["en"]))

# ---------- Home (tabs) ----------
HOME_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title id="t-title">Obsero Safety Panel</title>
  <link rel="icon" href="./static/favicon.ico" />
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root{
      --bg:#0b0f14; --bg-2:#101820; --card:#111821; --line:#1f2b37;
      --text:#e9eef5; --muted:#9fb3c8;
      --accent:#ff7a1a; --accent-2:#ff9b4e; --accent-3:#ff6a00;
    }
    *{box-sizing:border-box}
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:0;background:var(--bg);color:var(--text)}
    header{padding:12px 16px;background:linear-gradient(90deg,var(--bg-2),#0e1218 60%,var(--bg-2));
            border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px}
    .logo{display:inline-flex;align-items:center;gap:8px}
    .logo i{display:inline-block;width:18px;height:18px;background:var(--accent);clip-path:polygon(45% 0,20% 45%,45% 45%,35% 100%,85% 35%,55% 35%);}
    .brand{font-weight:700;letter-spacing:.3px}
    .lang{margin-left:auto; display:flex; gap:8px; align-items:center}
    select{background:#0f1620;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:6px 8px}
    main{padding:16px}
    .tabs{display:flex;gap:8px;margin-bottom:12px}
    .tab{padding:8px 12px;border-radius:999px;border:1px solid #5a2c00;background:linear-gradient(180deg,var(--accent),var(--accent-3));color:#1a0f08;cursor:pointer;font-weight:700}
    .tab.ghost{background:#0f1620;border-color:#203041;color:#e9eef5}
    .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;box-shadow:0 0 0 1px rgba(255,122,26,0.04),0 10px 30px rgba(0,0,0,.25); margin-bottom:16px}
    .grid2{display:grid;grid-template-columns:2fr 1fr;gap:16px}
    .small{color:var(--muted)}
    table{width:100%;border-collapse:collapse}
    th,td{border-bottom:1px solid #223142;padding:6px 8px;font-size:14px}
    .pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#1d2a36;color:#ffd1ae;font-size:12px;border:1px solid #26384a}
    .row{display:flex;gap:8px;align-items:center}
    button{background:linear-gradient(180deg,var(--accent),var(--accent-3));
      border:1px solid #5a2c00;color:#1a0f08;border-radius:8px;padding:8px 12px;cursor:pointer;font-weight:600}
    button:hover{filter:brightness(1.05)}
    .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
    img.stream,img.snap{width:100%;height:auto;border-radius:8px;border:1px solid #233445}
    .badge{display:inline-block;background:#1f2b37;border:1px solid #324559;color:#ffd8bd;padding:2px 8px;border-radius:999px;font-size:12px}
    .loader{position:relative;min-height:120px;display:grid;place-items:center}
    .spinner{width:38px;height:38px;border-radius:50%;border:4px solid rgba(255,122,26,.25);border-top-color:var(--accent);animation:spin 1s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
    .overlay{position:fixed;inset:0;background:rgba(0,0,0,.35);display:none;align-items:center;justify-content:center;z-index:30}
    .overlay .spinner{width:54px;height:54px;border-top-color:var(--accent-2)}
    .kpi{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
    .kpi .box{background:#0f1620;border:1px solid #203041;border-radius:10px;padding:8px 10px}
    .hidden{display:none}
  </style>
</head>
<body>
<header>
  <div class="logo"><i></i><span class="brand" id="t-title-2">Obsero Safety Panel</span></div>
  <div class="lang">
    <span class="small" id="t-lang">Language</span>
    <select id="langSel" onchange="setLang(this.value)">
      <option value="en">English</option>
      <option value="zh">中文</option>
      <option value="tr">Türkçe</option>
    </select>
  </div>
</header>

<main>
  <div class="tabs">
    <button id="tabLive" class="tab" onclick="showTab('live')">Live View</button>
    <button id="tabMulti" class="tab ghost" onclick="showTab('multi')">Multi-cam</button>
  </div>

  <section id="panelLive">
    <div class="grid2">
      <div class="card">
        <h2 id="t-live">Live View</h2>
        <div class="row">
          <select id="camSelect"></select>
          <button id="t-switch" onclick="switchCam()">Switch to Camera</button>
        </div>
        <div class="loader"><div class="spinner" id="liveSpin"></div></div>
        <img id="live" class="stream" src="" style="display:none"/>
        <div class="small" id="statusLine"></div>
      </div>

      <div class="card">
        <h2 id="t-overall">Overview</h2>
        <div class="kpi" id="stats"></div>
        <h3 id="t-rt">Real-time Alerts</h3>
        <table>
          <thead><tr>
            <th id="th-time">Time</th><th id="th-event">Event</th><th id="th-level">Level</th><th id="th-image">Image</th>
          </tr></thead>
          <tbody id="rtAlerts"></tbody>
        </table>
      </div>
    </div>
  </section>

  <section id="panelMulti" class="hidden">
    <div class="card">
      <h2 id="t-grid">Multi-view (snapshots, refresh every 2s)</h2>
      <div class="grid" id="grid"></div>
    </div>
  </section>
</main>

<div class="overlay" id="overlay"><div class="spinner"></div></div>

<script>
let LANG = localStorage.getItem('lang') || 'en';
document.getElementById('langSel').value = LANG;

function showTab(which){
  const live = document.getElementById('panelLive');
  const multi = document.getElementById('panelMulti');
  const t1 = document.getElementById('tabLive');
  const t2 = document.getElementById('tabMulti');
  if(which==='live'){ live.classList.remove('hidden'); multi.classList.add('hidden'); t1.classList.remove('ghost'); t2.classList.add('ghost'); }
  else{ multi.classList.remove('hidden'); live.classList.add('hidden'); t2.classList.remove('ghost'); t1.classList.add('ghost'); }
}

function setLang(l){ localStorage.setItem('lang', l); LANG = l; applyLang(); }

async function applyLang(){
  const r = await fetch('/api/i18n?lang='+LANG); const t = await r.json();
  const map = {
    't-title': 'title','t-title-2':'title','t-live':'tab_live','t-grid':'grid_title','t-lang':'lang',
    't-overall':'overall','t-rt':'realtime_alerts','t-switch':'switch',
    'th-time':'time','th-event':'event','th-level':'level','th-image':'image'
  };
  for (const id in map){ const el=document.getElementById(id); if(el) el.textContent = t[map[id]]; }
  document.getElementById('tabLive').textContent = t.tab_live;
  document.getElementById('tabMulti').textContent = t.tab_multi;
  document.title = t.title;
  window._i18n = t;
}

function showOverlay(b){ document.getElementById('overlay').style.display = b ? 'flex' : 'none'; }

async function loadCams(){
  const r = await fetch('/api/cameras'); const cams = await r.json();
  const t = window._i18n || {};
  const sel = document.getElementById('camSelect'); sel.innerHTML='';
  for(const c of cams){
    const o=document.createElement('option'); o.value=c.id;
    const on = c.online ? (t.online||'online') : (t.offline||'offline');
    const name = c.name || (t.default_cam || 'Default Camera');
    o.textContent=`[${on}] ${name} (${c.code||c.id})`; sel.appendChild(o);
  }
}

async function switchCam(){
  showOverlay(true);
  const id = document.getElementById('camSelect').value;
  await fetch('/api/select_camera?camera_id='+id, {method:'POST'});
  setTimeout(()=>{ loadLive(); showOverlay(false); }, 600);
}

function loadLive(){
  const live = document.getElementById('live');
  const spin = document.getElementById('liveSpin');
  live.style.display='none'; spin.style.display='block';
  live.onload = ()=>{ spin.style.display='none'; live.style.display='block'; };
  live.onerror = ()=>{ spin.style.display='block'; live.style.display='none'; setTimeout(loadLive, 1000); };
  live.src = '/stream?t=' + Date.now();
}

function fmtTemp(v){ return (v===null || v===undefined) ? '—' : (Math.round(v)+'°C'); }
function fmtPct(v){ return (v===null || v===undefined) ? '—' : (v+'%'); }

async function refreshStats(){
  const r = await fetch('/api/status'); const s = await r.json();
  const t = window._i18n || {};
  const vram = (s.health.vram_percent!==null && s.health.vram_percent!==undefined)
      ? `${s.health.vram_percent}% (${s.health.vram_used_mib}/${s.health.vram_total_mib} MiB)` : '—';
  const kpi = `
    <div class="box">${t.sites||'Sites'}: <b>1</b></div>
    <div class="box">${t.hosts||'Analyzer Hosts'}: <b>1</b></div>
    <div class="box">${t.devices||'Devices'}: <b>${s.cameras_online}/${s.cameras_total}</b> ${(t.online||'online')}</div>
    <div class="box">${t.cpu||'CPU'}: ${s.health.cpu}%</div>
    <div class="box">${t.cpu_temp||'CPU Temp'}: ${fmtTemp(s.health.cpu_temp)}</div>
    <div class="box">${t.ram||'RAM'}: ${s.health.mem}%</div>
    <div class="box">${t.gpu||'GPU'}: ${s.health.gpu}</div>
    <div class="box">${t.gpu_util||'GPU Util'}: ${fmtPct(s.health.gpu_util)}</div>
    <div class="box">${t.vram||'VRAM'}: ${vram}</div>
    <div class="box">${t.gpu_temp||'GPU Temp'}: ${fmtTemp(s.health.gpu_temp)}</div>
  `;
  document.getElementById('stats').innerHTML = kpi;
  const ac = s.active_camera; const label = (t.current_cam||'Active camera');
  const name = (ac && ac.name) ? ac.name : (t.default_cam || 'Default Camera');
  document.getElementById('statusLine').textContent = `${label}: ${name} / ${ac?.url||''}`;
}

async function refreshRT(){
  const t = window._i18n || {};
  const r = await fetch('/api/alerts?limit=10'); const arr = await r.json();
  const tb = document.getElementById('rtAlerts'); tb.innerHTML='';
  for(const a of arr){
    const tr=document.createElement('tr');
    tr.innerHTML = `<td>${a.ts.split('.')[0]}</td>
      <td><span class="pill">${a.type}</span> ${a.label||''}</td>
      <td>${a.level}</td>
      <td>${a.image?('<a class="badge" href="/incidents/'+a.image+'" target="_blank">'+(t.view||'view')+'</a>'):''}</td>`;
    tb.appendChild(tr);
  }
}

async function refreshGrid(){
  const t = window._i18n || {};
  const r = await fetch('/api/cameras'); const cams = await r.json();
  const g = document.getElementById('grid'); g.innerHTML='';
  for(const c of cams){
    const on = c.online ? (t.online||'online') : (t.offline||'offline');
    const name = c.name || (t.default_cam || 'Default Camera');
    const wrap = document.createElement('div');
    wrap.innerHTML = `<div class="small">${name} (${on})</div>
      <div class="loader"><div class="spinner"></div></div>
      <img class="snap" style="display:none" src="/snapshot?camera_id=${c.id}&t=${Date.now()}">`;
    const img = wrap.querySelector('img'); const sp = wrap.querySelector('.spinner');
    img.onload = ()=>{ sp.style.display='none'; img.style.display='block'; };
    img.onerror = ()=>{ sp.style.display='block'; img.style.display='none'; };
    g.appendChild(wrap);
  }
}

setInterval(refreshStats, 2000);
setInterval(refreshRT, 2000);
setInterval(()=>{ if(!document.getElementById('panelMulti').classList.contains('hidden')) refreshGrid(); }, 2000);

(async ()=>{ showTab('live'); await applyLang(); await loadCams(); loadLive(); refreshStats(); refreshRT(); refreshGrid(); })();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HOME_HTML

@app.get("/api/ping")
def api_ping():
    return PlainTextResponse("pong")

# ---- Streaming MJPEG ----
@app.get("/stream")
def mjpeg_stream():
    return StreamingResponse(_gen_mjpeg(), media_type='multipart/x-mixed-replace; boundary=frame')

def _gen_mjpeg():
    boundary=b"--frame\r\n"; headers=b"Content-Type: image/jpeg\r\n\r\n"
    def gen():
        while not STOP_EVENT.is_set():
            with LAST_JPEG_LOCK:
                b = LAST_JPEG
            if b is not None:
                yield boundary + headers + b + b"\r\n"
            time.sleep(0.04)
    return gen()

# ============ APIs ============
@app.get("/api/status")
def api_status():
    cams = cameras_all()
    con = db_conn()
    total = con.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
    online = con.execute("SELECT COUNT(*) FROM cameras WHERE online=1").fetchone()[0]
    type_counts = {k: v for k,v in con.execute("SELECT type, COUNT(*) FROM alerts GROUP BY type").fetchall()}
    level_counts = {k: v for k,v in con.execute("SELECT level, COUNT(*) FROM alerts GROUP BY level").fetchall()}
    con.close()
    return JSONResponse({
        "sites": 1,
        "servers": 1,
        "cameras_total": total, "cameras_online": online,
        "health": get_health_snapshot(),
        "alert_stats": {"type": type_counts, "level": level_counts},
        "active_camera": {
            "id": CAM_MGR.active_camera_id if CAM_MGR else None,
            "url": next((c["url"] for c in cams if CAM_MGR and c["id"]==CAM_MGR.active_camera_id), None),
            "name": next((c["name"] for c in cams if CAM_MGR and c["id"]==CAM_MGR.active_camera_id), None)
        }
    })

@app.get("/api/cameras")
def api_cameras():
    return JSONResponse(cameras_all())

@app.post("/api/cameras")
def api_camera_add(name: str = Form(...), url: str = Form(...), code: str = Form(None), ptz_protocol: str = Form("onvif")):
    camera_upsert(name, url, code, ptz_protocol, cid=None)
    audit("system", "camera_add", f"{name}")
    return PlainTextResponse("ok")

@app.put("/api/cameras/{cid}")
def api_camera_update(cid: int, name: str = Form(...), url: str = Form(...), code: str = Form(None), ptz_protocol: str = Form("onvif")):
    camera_upsert(name, url, code, ptz_protocol, cid=cid)
    audit("system", "camera_update", f"{cid}")
    return PlainTextResponse("ok")

@app.delete("/api/cameras/{cid}")
def api_camera_delete(cid: int):
    con = db_conn(); con.execute("DELETE FROM cameras WHERE id=?", (cid,)); con.commit(); con.close()
    audit("system","camera_delete", str(cid))
    return PlainTextResponse("ok")

@app.post("/api/select_camera")
def api_select_camera(camera_id: int):
    cam = camera_by_id(camera_id)
    if not cam: return JSONResponse({"error":"not found"}, status_code=404)
    if CAM_MGR:
        CAM_MGR.switch_active(cam["id"])
    audit("system","select_camera", str(cam["id"]))
    return PlainTextResponse("ok")

@app.get("/snapshot")
def api_snapshot(camera_id: int):
    with LAST_SNAPSHOTS_LOCK:
        b = LAST_SNAPSHOTS.get(camera_id)
    if b:
        return Response(content=b, media_type="image/jpeg")
    return Response(status_code=204)

@app.get("/api/alerts")
def api_alerts(camera_id: int | None = None, level: str | None = None, type: str | None = None,
               t_from: str | None = None, t_to: str | None = None, limit: int = 50):
    rows = alert_query({"camera_id":camera_id, "level":level, "type":type, "t_from":t_from, "t_to":t_to})
    return JSONResponse(rows[:max(1,min(limit,500))])

@app.post("/api/alerts/confirm")
def api_alert_confirm(alert_id: int, status: str = Form("ack"), reviewer: str = Form("operator")):
    alert_update_status(alert_id, status, reviewer)
    audit(reviewer, "alert_confirm", f"{alert_id}:{status}")
    return PlainTextResponse("ok")

@app.post("/api/alerts/upload")
def api_alert_upload(camera_id: int = Form(None), level: str = Form("manual"),
                     type: str = Form("MANUAL"), site: str = Form("Site-A"),
                     file: UploadFile = File(...)):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    name = f"{ts.replace(':','-').replace(' ','_')}_manual.jpg"
    data = file.file.read()
    (INCIDENTS_DIR / name).write_bytes(data)
    alert_insert(ts, camera_id, level, type, "manual_upload", 1.0, [0,0,0,0], name, site, 'new')
    audit("operator", "alert_upload", name)
    return PlainTextResponse("ok")

@app.get("/api/alarm_levels")
def api_levels_get():
    con = db_conn(); rows = con.execute("SELECT event_type, level FROM alarm_levels").fetchall(); con.close()
    return JSONResponse([{ "event_type": r["event_type"], "level": r["level"] } for r in rows])

@app.post("/api/alarm_levels")
def api_levels_set(event_type: str = Form(...), level: str = Form(...)):
    con = db_conn()
    con.execute("INSERT INTO alarm_levels(event_type,level) VALUES(?,?) ON CONFLICT(event_type) DO UPDATE SET level=excluded.level",
                (event_type, level))
    con.commit(); con.close()
    audit("admin","alarm_level_set", f"{event_type}={level}")
    return PlainTextResponse("ok")

@app.get("/api/logs")
def api_logs(limit: int = 200):
    con = db_conn(); rows = con.execute("SELECT * FROM audit_logs ORDER BY ts DESC LIMIT ?", (limit,)).fetchall(); con.close()
    return JSONResponse([dict(r) for r in rows])

# ---------- bootstrap helpers ----------
def parse_source(source_str: str):
    try:
        if str(source_str).isdigit(): return int(source_str)
    except: pass
    return source_str

def backend_bootstrap(initial_active_id: int):
    global MP_INPUTS, MP_OUTPUT, FANOUT, CAM_MGR
    try:
        print("[boot] backend bootstrap starting ...", flush=True)

        # 1) model workers + fanout
        MP_INPUTS = {}
        MP_OUTPUT = mp.Queue(maxsize=200)
        fan = FanOut(CADENCE)
        print("[boot] spawning model workers ...", flush=True)
        for key, (stem, conf, keywords) in MODEL_SET.items():
            q = mp.Queue(maxsize=120)
            p = mp.Process(target=mp_infer_worker, args=(key, stem, conf, keywords, q, MP_OUTPUT), daemon=True)
            p.start()
            MP_INPUTS[key] = (q, p)
            fan.register(key, q)
        print(f"[boot] models up: {len(MP_INPUTS)}", flush=True)
        threading.Thread(target=results_collector_thread,
                         args=(MP_OUTPUT, lambda: CAM_MGR.active_camera_id if CAM_MGR else None),
                         daemon=True).start()
        globals()['FANOUT'] = fan

        # 2) cameras + router + composer
        cams = cameras_all()
        cam_map = {c["id"]: parse_source(c["url"]) for c in cams}
        print(f"[boot] found {len(cam_map)} camera(s) in DB", flush=True)

        globals()['CAM_MGR'] = MultiCamProcManager(CAMERA_OUT)
        CAM_MGR.start_all(cam_map, initial_active_id)

        threading.Thread(target=camera_router_thread,
                         args=(CAMERA_OUT, lambda: CAM_MGR.active_camera_id if CAM_MGR else None),
                         daemon=True).start()
        threading.Thread(target=composer_thread,
                         args=(STOP_EVENT, lambda: CAM_MGR.active_camera_id if CAM_MGR else None),
                         daemon=True).start()

        print("[boot] backend bootstrap complete", flush=True)
    except Exception as e:
        print(f"[boot] ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)

def install_signal_handlers():
    def handle_sig(sig, frame):
        print(f"[main] signal {sig} -> shutting down ...", flush=True)
        STOP_EVENT.set()
    for s in (signal.SIGINT, signal.SIGTERM):
        try: signal.signal(s, handle_sig)
        except Exception: pass

# -------------- main --------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    install_signal_handlers()

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Webcam index or RTSP/HTTP URL")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--camera-name", type=str, default="Default Camera")
    parser.add_argument("--camera-code", type=str, default="CAM-001")
    args = parser.parse_args()

    # seed DB if empty
    if not cameras_all():
        camera_upsert(args.camera_name, args.source, args.camera_code)

    cams = cameras_all()
    active = next((c for c in cams if (c["url"]==args.source or str(c["id"])==args.source or c["code"]==args.camera_code)), cams[0])

    # IPC queues FIRST (so children inherit handles correctly)
    CAMERA_OUT = mp.Queue(maxsize=200)

    # Start backend in a worker thread
    threading.Thread(target=backend_bootstrap, args=(active["id"],), daemon=True).start()

    # Start web server in MAIN THREAD (Windows-friendly)
    print(f"[web] starting uvicorn on http://{args.host}:{args.port}", flush=True)
    try: webbrowser.open(f"http://127.0.0.1:{args.port}/")
    except: pass
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    # Teardown (after uvicorn exits)
    STOP_EVENT.set()
    if CAM_MGR: CAM_MGR.stop_all()
    for key, (q, p) in MP_INPUTS.items():
        try: q.put_nowait(None)
        except: pass
        if p.is_alive():
            p.join(timeout=2.0)
    print("[main] bye", flush=True)
