"""
obsero.api â€” FastAPI application, all routes, HTML UI.

Routes preserved:
  /  /stream  /snapshot  /api/cameras  /api/alerts  /api/status  /incidents/*

New routes:
  POST /api/alerts/label          â€” confirm / false_positive / ignore / close
  GET  /api/metrics/precision     â€” operational precision measurement
"""

from __future__ import annotations

import csv, datetime, json, queue, signal, threading, time, traceback, sys, os, warnings
from collections import deque
from pathlib import Path
from typing import Any

# Allow running this file directly for debugging by ensuring repo root is on sys.path.
if __name__ == "__main__" and __package__ in (None, ""):
  _ROOT = Path(__file__).resolve().parent.parent
  if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import psutil

from fastapi import FastAPI, Response, UploadFile, File, Form
from fastapi.responses import (HTMLResponse, StreamingResponse,
                                JSONResponse, PlainTextResponse, FileResponse,
                                RedirectResponse)
from fastapi.staticfiles import StaticFiles
from obsero.config import CameraCfg, CAMERAS_JSON_PATH, CONFIG_PATH
import yaml

# â”€â”€ paths â”€â”€
ROOT = Path(__file__).resolve().parent.parent
INCIDENTS_DIR = ROOT / "incidents"
STATIC_DIR = ROOT / "static"
REPORTS_DIR = ROOT / "data" / "reports"
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# â”€â”€ GPU monitoring â”€â”€
try:
  with warnings.catch_warnings():
    warnings.filterwarnings(
      "ignore",
      message="The pynvml package is deprecated.*",
      category=FutureWarning,
    )
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# App-level state â€” set by bootstrap (run.py) before uvicorn starts.
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
class _AppState:
    stop_event: threading.Event = threading.Event()
    cam_mgr: Any = None                       # MultiCamProcManager
    cfg: Any = None                           # SystemConfig
    annotated_jpeg: dict = {"jpeg": None}     # live stream jpeg
    annotated_jpeg_lock: threading.Lock = threading.Lock()
    raw_snaps: dict = {}                      # camera_id -> raw jpeg
    raw_snaps_lock: threading.Lock = threading.Lock()
    annotated_snaps: dict = {}                # camera_id -> annotated jpeg
    annotated_snaps_lock: threading.Lock = threading.Lock()
    incidents_ring: deque = deque(maxlen=200)
    _mp_inputs: Any = None

S = _AppState()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Helpers
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _encode_jpeg(img, q=82) -> bytes | None:
    ok, b = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return b.tobytes() if ok else None


def _no_signal(w=640, h=360):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (18, 18, 24)
    for i in range(-h, w, 16):
        cv2.line(img, (i, 0), (i + h, h), (30, 30, 44), 1, cv2.LINE_AA)
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), (70, 70, 95), 2, cv2.LINE_AA)
    txt = "NO SIGNAL"
    (tw, th2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(img, txt, ((w - tw) // 2, (h + th2) // 2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 140, 0), 2, cv2.LINE_AA)
    return _encode_jpeg(img, 80)


def _cpu_temp_c():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        cands = []
        for _, entries in temps.items():
            for e in entries:
                lab = (e.label or "").lower()
                if any(k in lab for k in ["package", "tdie", "tctl", "cpu"]):
                    if e.current is not None:
                        cands.append(e.current)
                elif e.current is not None:
                    cands.append(e.current)
        return float(max(cands)) if cands else None
    except Exception:
        return None


def get_health_snapshot():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    cpu_temp = _cpu_temp_c()

    gpus_info: list[dict] = []
    if NVML_OK:
        try:
            count = pynvml.nvmlDeviceGetCount()
        except Exception:
            count = 0
        for idx in range(count):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                name_raw = pynvml.nvmlDeviceGetName(h)
                gpu_name = name_raw.decode("utf-8") if hasattr(name_raw, "decode") else str(name_raw)
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                mi = pynvml.nvmlDeviceGetMemoryInfo(h)
                gt = int(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
                gpus_info.append(dict(
                    id=idx, name=gpu_name, util=int(u.gpu),
                    vram_used_mib=int(mi.used // (1024 * 1024)),
                    vram_total_mib=int(mi.total // (1024 * 1024)),
                    vram_percent=round(100.0 * mi.used / mi.total, 1) if mi.total else None,
                    temp=gt,
                ))
            except Exception:
                pass

    # backward-compat single-GPU summary
    g0 = gpus_info[0] if gpus_info else {}
    return dict(
        cpu=cpu, mem=mem, cpu_temp=cpu_temp,
        gpu=g0.get("name", "N/A"),
        gpu_util=g0.get("util"),
        vram_percent=g0.get("vram_percent"),
        vram_used_mib=g0.get("vram_used_mib"),
        vram_total_mib=g0.get("vram_total_mib"),
        gpu_temp=g0.get("temp"),
        gpus=gpus_info,
    )


def _is_configured() -> bool:
    try:
        con = db_conn()
        total = int(con.execute("SELECT COUNT(*) FROM cameras").fetchone()[0])
        con.close()
        return total > 0
    except Exception:
        return False


def _write_empty_camera_config():
    data = {
        "discovered_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "nvr": None,
        "cameras": [],
    }
    CAMERAS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    CAMERAS_JSON_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _parse_db_ts(ts: str | None) -> datetime.datetime | None:
    if not ts:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.datetime.strptime(ts, fmt)
        except Exception:
            continue
    return None


def _bbox_from_json(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    try:
        b = json.loads(raw)
        if isinstance(b, list) and len(b) == 4:
            return [float(x) for x in b]
    except Exception:
        return None
    return None


def _bbox_iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = a_area + b_area - inter
    return (inter / union) if union > 0 else 0.0


SLA_MINUTES_BY_LEVEL = {
    "high": 5,
    "medium": 15,
    "low": 30,
}


def _sla_worker_loop():
    while not S.stop_event.is_set():
        try:
            con = db_conn()
            rows = con.execute(
                """
                SELECT id, ts, level, status, assignee, due_at
                  FROM alerts
                 WHERE status <> 'closed'
                """
            ).fetchall()
            con.close()

            now = datetime.datetime.now()
            for r in rows:
                aid = int(r["id"])
                ts = _parse_db_ts(r["ts"])
                level = (r["level"] or "medium").lower()
                status = (r["status"] or "new").lower()
                assignee = (r["assignee"] or "").strip().lower()
                due_at = _parse_db_ts(r["due_at"])

                if due_at is None and ts is not None:
                    mins = SLA_MINUTES_BY_LEVEL.get(level, 15)
                    due = ts + datetime.timedelta(minutes=mins)
                    due_str = due.strftime("%Y-%m-%d %H:%M:%S")
                    con2 = db_conn()
                    con2.execute("UPDATE alerts SET due_at=? WHERE id=?", (due_str, aid))
                    con2.commit()
                    con2.close()
                    due_at = due

                if due_at is None or due_at >= now:
                    continue

                # Escalation policy: unhandled early states -> supervisor; late states -> manager.
                target = "supervisor" if status in {"new", "triage", "ack"} else "manager"
                if assignee != target:
                    try:
                        incident_assign(aid, assignee=target, actor="sla-worker")
                    except Exception:
                        pass
                if status in {"new", "triage", "ack"}:
                    try:
                        incident_transition(aid, "assigned", actor="sla-worker", note="sla_escalation")
                    except Exception:
                        pass
                audit("sla-worker", "incident_escalated", f"{aid}:{target}")
        except Exception:
            traceback.print_exc()

        S.stop_event.wait(30.0)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# FastAPI application
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
app = FastAPI(title="Obsero Safety Panel")
app.mount("/incidents", StaticFiles(directory=str(INCIDENTS_DIR)), name="incidents")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def _startup_workers():
    # Uvicorn can import app multiple times in some modes; guard against duplicate workers.
    if getattr(S, "_sla_worker_started", False):
        return
    t = threading.Thread(target=_sla_worker_loop, daemon=True)
    t.start()
    S._sla_worker_started = True

FALLBACK_FAVICON_SVG = b"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
<rect width='64' height='64' fill='#0b0f14'/><path d='M36 6L14 36h14l-6 22 28-36H36z' fill='#ff7a1a'/>
</svg>"""


@app.get("/favicon.ico")
def favicon():
    ico = STATIC_DIR / "favicon.ico"
    if ico.exists():
        return FileResponse(str(ico), media_type="image/x-icon")
    return Response(content=FALLBACK_FAVICON_SVG, media_type="image/svg+xml")


# ---------- i18n ----------------------------------------------------------
I18N = {
    "en": {"title":"Obsero Safety Panel","tab_live":"Live View","tab_multi":"Multi-cam","switch":"Switch to Camera",
           "overall":"Overview","sites":"Sites","hosts":"Analyzer Hosts","devices":"Devices","online":"online","offline":"offline",
           "cpu":"CPU","ram":"RAM","gpu":"GPU","gpu_util":"GPU Util","vram":"VRAM","gpu_temp":"GPU Temp","cpu_temp":"CPU Temp",
           "realtime_alerts":"Real-time Alerts","grid_title":"Multi-view (snapshots, refresh every 2s)",
           "current_cam":"Active camera","view":"view","time":"Time","event":"Event","level":"Level","image":"Image",
      "lang":"Language","english":"English","chinese":"中文","turkish":"Türkçe","default_cam":"Default Camera",
  "tab_settings":"Settings","tab_ops":"Incident Ops",
  "ops_title":"Incident Ops Queue","ops_status":"Status","ops_overdue":"overdue only",
  "ops_refresh":"Refresh","ops_group":"Group Duplicates","ops_export_json":"Export JSON","ops_export_csv":"Export CSV",
  "ops_report_ready":"Report ready:","ops_report_view":"View","ops_report_download":"Download","ops_report_preview":"Preview",
  "ops_assign":"Assign","ops_update":"Update","ops_close":"Close","ops_merge":"Merge","ops_open":"Open",
  "ops_camera":"Camera","ops_assignee":"Assignee","ops_due":"Due","ops_actions":"Actions",
  "ops_kpi_backlog":"Backlog","ops_kpi_mtta":"MTTA","ops_kpi_mttr":"MTTR","ops_kpi_fp":"False+ Rate",
  "ops_popup_title":"Incident","ops_popup_save":"Save","ops_popup_cancel":"Cancel","ops_popup_timeline":"Timeline",
  "ops_note":"Note","ops_type":"Type","ops_site":"Site","ops_risk":"Risk","ops_policy":"Policy","ops_root_cause":"Root Cause",
  "ops_corrective":"Corrective Action","ops_approver":"Approver"},
    "zh": {"title":"Obsero 安全面板","tab_live":"实时画面","tab_multi":"多路画面","switch":"切换到该摄像头",
           "overall":"总体状态","sites":"场站","hosts":"分析主机","devices":"设备","online":"在线","offline":"离线",
           "cpu":"CPU","ram":"内存","gpu":"显卡","gpu_util":"显卡占用","vram":"显存","gpu_temp":"显卡温度","cpu_temp":"CPU温度",
           "realtime_alerts":"实时告警","grid_title":"分屏（快照，2秒刷新）","current_cam":"当前摄像头","view":"查看",
      "time":"时间","event":"事件","level":"级别","image":"图像","lang":"语言","english":"English","chinese":"中文","turkish":"Türkçe","default_cam":"默认相机",
      "tab_settings":"设置","tab_ops":"事件运营",
      "ops_title":"事件队列","ops_status":"状态","ops_overdue":"仅逾期",
      "ops_refresh":"刷新","ops_group":"重复合并","ops_export_json":"导出 JSON","ops_export_csv":"导出 CSV",
      "ops_report_ready":"报告已生成：","ops_report_view":"查看","ops_report_download":"下载","ops_report_preview":"预览",
      "ops_assign":"分配","ops_update":"更新","ops_close":"关闭","ops_merge":"合并","ops_open":"打开",
      "ops_camera":"摄像头","ops_assignee":"负责人","ops_due":"截止时间","ops_actions":"操作",
      "ops_kpi_backlog":"待处理","ops_kpi_mtta":"平均响应","ops_kpi_mttr":"平均关闭","ops_kpi_fp":"误报率",
      "ops_popup_title":"事件","ops_popup_save":"保存","ops_popup_cancel":"取消","ops_popup_timeline":"时间线",
      "ops_note":"备注","ops_type":"类型","ops_site":"站点","ops_risk":"风险","ops_policy":"政策","ops_root_cause":"根因",
      "ops_corrective":"纠正措施","ops_approver":"审批人"},
    "tr": {"title":"Obsero Güvenlik Paneli","tab_live":"Canlı Görüntü","tab_multi":"Çoklu Kamera","switch":"Kameraya Geç",
           "overall":"Genel Durum","sites":"Saha","hosts":"Analiz Sunucuları","devices":"Cihazlar","online":"çevrimiçi","offline":"çevrimdışı",
           "cpu":"CPU","ram":"RAM","gpu":"GPU","gpu_util":"GPU Kullanımı","vram":"VRAM","gpu_temp":"GPU Sıcaklığı","cpu_temp":"CPU Sıcaklığı",
           "realtime_alerts":"Anlık Alarmlar","grid_title":"Çoklu Görünüm (anlık görüntü, 2 sn)","current_cam":"Aktif kamera","view":"gör",
      "time":"Zaman","event":"Olay","level":"Seviye","image":"Görüntü","lang":"Dil","english":"English","chinese":"中文","turkish":"Türkçe","default_cam":"Varsayılan Kamera",
      "tab_settings":"Ayarlar","tab_ops":"Olay Operasyonu",
      "ops_title":"Olay Kuyruğu","ops_status":"Durum","ops_overdue":"yalnızca geciken",
      "ops_refresh":"Yenile","ops_group":"Yinelenenleri Birleştir","ops_export_json":"JSON Dışa Aktar","ops_export_csv":"CSV Dışa Aktar",
      "ops_report_ready":"Rapor hazır:","ops_report_view":"Görüntüle","ops_report_download":"İndir","ops_report_preview":"Önizleme",
      "ops_assign":"Ata","ops_update":"Güncelle","ops_close":"Kapat","ops_merge":"Birleştir","ops_open":"Aç",
      "ops_camera":"Kamera","ops_assignee":"Sorumlu","ops_due":"Termin","ops_actions":"İşlemler",
      "ops_kpi_backlog":"Bekleyen","ops_kpi_mtta":"MTTA","ops_kpi_mttr":"MTTR","ops_kpi_fp":"Yanlış Pozitif Oranı",
      "ops_popup_title":"Olay","ops_popup_save":"Kaydet","ops_popup_cancel":"İptal","ops_popup_timeline":"Zaman Çizgisi",
      "ops_note":"Not","ops_type":"Tür","ops_site":"Saha","ops_risk":"Risk","ops_policy":"Politika","ops_root_cause":"Kök Neden",
      "ops_corrective":"Düzeltici Aksiyon","ops_approver":"Onaylayan"},
}

@app.get("/api/i18n")
def api_i18n(lang: str = "en"):
    return JSONResponse(I18N.get(lang, I18N["en"]))


# ---------- Home HTML ---------------------------------------------------------
SETUP_HTML = """
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\"/>
  <title>Obsero Setup</title>
  <link rel=\"icon\" href=\"./static/favicon.ico\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>
  <style>
    :root{ --bg:#0b0f14; --bg-2:#101820; --card:#111821; --line:#1f2b37; --text:#e9eef5; --muted:#9fb3c8; --accent:#ff7a1a; --accent-2:#ff9b4e; --accent-3:#ff6a00; }
    *{box-sizing:border-box}
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:0;background:radial-gradient(1200px 800px at 10% 0%,#132335 0%,#0b0f14 55%),var(--bg);color:var(--text)}
    header{padding:14px 18px;background:linear-gradient(90deg,var(--bg-2),#0e1218 60%,var(--bg-2)); border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px}
    .logo{display:inline-flex;align-items:center;gap:8px}
    .logo i{display:inline-block;width:18px;height:18px;background:var(--accent);clip-path:polygon(45% 0,20% 45%,45% 45%,35% 100%,85% 35%,55% 35%);}
    .brand{font-weight:800;letter-spacing:.3px}
    main{max-width:760px;margin:26px auto;padding:0 16px}
    .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 0 0 1px rgba(255,122,26,0.05),0 14px 34px rgba(0,0,0,.28)}
    h2{margin:0 0 8px}
    .small{color:var(--muted)}
    .row{display:flex;gap:8px;align-items:center;margin:10px 0}
    select{background:#0f1620;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:8px 10px;min-width:260px}
    button{background:linear-gradient(180deg,var(--accent),var(--accent-3)); border:1px solid #5a2c00;color:#1a0f08;border-radius:8px;padding:9px 13px;cursor:pointer;font-weight:700}
    button.ghost{background:#0f1620;border-color:#203041;color:#e9eef5}
    .ok{color:#92f6b7}
    .bad{color:#ffb8b8}
  </style>
</head>
<body>
<header><div class=\"logo\"><i></i><span class=\"brand\">Obsero Setup Wizard</span></div></header>
<main>
  <div class=\"card\">
    <h2>Camera Onboarding (Webcam First)</h2>
    <div class=\"small\">Use this once before entering the safety panel. RTSP/NVR onboarding can be added later.</div>
    <div class=\"row\">
      <button class=\"ghost\" onclick=\"scan()\">Scan Webcams</button>
      <select id=\"cams\"></select>
      <button onclick=\"applySetup()\">Apply And Start Panel</button>
    </div>
    <div id=\"msg\" class=\"small\"></div>
  </div>
</main>
<script>
async function scan(){
  const msg = document.getElementById('msg');
  msg.textContent = 'Scanning local webcams...';
  const r = await fetch('/api/setup/webcams');
  const d = await r.json();
  const sel = document.getElementById('cams');
  sel.innerHTML = '';
  if(!d.ok || !d.cameras || d.cameras.length===0){
    msg.className = 'small bad';
    msg.textContent = 'No webcam detected. Close apps using the camera and try again.';
    return;
  }
  for(const c of d.cameras){
    const o = document.createElement('option');
    o.value = c.url;
    o.textContent = `${c.name} (index ${c.url})`;
    sel.appendChild(o);
  }
  msg.className = 'small ok';
  msg.textContent = `Found ${d.cameras.length} webcam(s). Select one and apply.`;
}

async function applySetup(){
  const sel = document.getElementById('cams');
  const msg = document.getElementById('msg');
  if(!sel.value){
    msg.className = 'small bad';
    msg.textContent = 'Select a webcam first.';
    return;
  }
  msg.className = 'small';
  msg.textContent = 'Applying configuration...';
  const idx = encodeURIComponent(sel.value);
  const r = await fetch('/api/setup/apply?webcam_index='+idx, {method:'POST'});
  const d = await r.json();
  if(!r.ok || !d.ok){
    msg.className = 'small bad';
    msg.textContent = 'Setup failed: ' + (d.error || 'unknown error');
    return;
  }
  msg.className = 'small ok';
  msg.textContent = 'Setup complete. Opening panel...';
  setTimeout(()=>{ window.location.href='/'; }, 450);
}

document.getElementById('msg').textContent = 'Click Scan Webcams to detect local devices.';
</script>
</body>
</html>
"""

HOME_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title id="t-title">Obsero Safety Panel</title>
  <link rel="icon" href="./static/favicon.ico" />
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root{ --bg:#0b0f14; --bg-2:#101820; --card:#111821; --line:#1f2b37; --text:#e9eef5; --muted:#9fb3c8; --accent:#ff7a1a; --accent-2:#ff9b4e; --accent-3:#ff6a00; }
    *{box-sizing:border-box}
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:0;background:var(--bg);color:var(--text)}
    header{padding:12px 16px;background:linear-gradient(90deg,var(--bg-2),#0e1218 60%,var(--bg-2)); border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px}
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
    .stream-frame{position:relative;width:100%;aspect-ratio:16/9;max-height:72vh;border-radius:8px;border:1px solid #233445;background:#05080d;overflow:hidden}
    img.stream{width:100%;height:100%;object-fit:contain;object-position:center center;display:block}
    img.snap{width:100%;height:auto;border-radius:8px;border:1px solid #233445;object-fit:cover;object-position:center center}
    .badge{display:inline-block;background:#1f2b37;border:1px solid #324559;color:#ffd8bd;padding:2px 8px;border-radius:999px;font-size:12px}
    .loader{position:relative;min-height:120px;display:grid;place-items:center}
    .spinner{width:38px;height:38px;border-radius:50%;border:4px solid rgba(255,122,26,.25);border-top-color:var(--accent);animation:spin 1s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
    .overlay{position:fixed;inset:0;background:rgba(0,0,0,.35);display:none;align-items:center;justify-content:center;z-index:30}
    .overlay .spinner{width:54px;height:54px;border-top-color:var(--accent-2)}
    .kpi{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
    .kpi .box{background:#0f1620;border:1px solid #203041;border-radius:10px;padding:8px 10px}
    .report-view{background:#0b1118;border:1px solid #203041;border-radius:8px;max-height:60vh;overflow:auto;padding:10px;white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#d6e5f3}
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
      <option value="zh">ä¸­æ–‡</option>
      <option value="tr">TÃ¼rkÃ§e</option>
    </select>
  </div>
</header>
<main>
  <div class="tabs">
    <button id="tabLive" class="tab" onclick="showTab('live')">Live View</button>
    <button id="tabMulti" class="tab ghost" onclick="showTab('multi')">Multi-cam</button>
    <button id="tabSettings" class="tab ghost" onclick="showTab('settings')">Settings</button>
    <button id="tabOps" class="tab ghost" onclick="showTab('ops')">Incident Ops</button>
  </div>
  <section id="panelLive">
    <div class="grid2">
      <div class="card">
        <h2 id="t-live">Live View</h2>
        <div class="row">
          <select id="camSelect"></select>
          <button id="t-switch" onclick="switchCam()">Switch to Camera</button>
        </div>
        <div class="stream-frame">
          <div class="loader" id="liveLoader"><div class="spinner" id="liveSpin"></div></div>
          <img id="live" class="stream" src="" style="display:none"/>
        </div>
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
  <section id="panelSettings" class="hidden">
    <div class="card">
      <h2 id="t-settings">Settings</h2>
      <div class="small">Minimum re-detection duration suppresses duplicate alerts for the same camera/event.</div>
      <div class="row" style="margin-top:10px">
        <label for="repeatSec" class="small">Repeat Duration (sec)</label>
        <input id="repeatSec" type="number" min="1" step="1" value="8" style="width:120px;background:#0f1620;border:1px solid #203041;color:#e9eef5;border-radius:8px;padding:7px"/>
        <button class="ghost" onclick="saveRepeatDuration()">Save</button>
      </div>
      <div id="settingsMsg" class="small"></div>
      <div class="small">Factory reset removes current camera configuration and takes you back to setup.</div>
      <div class="row" style="margin-top:10px">
        <button onclick="factoryReset()">Factory Reset</button>
      </div>
    </div>
  </section>
  <section id="panelOps" class="hidden">
    <div class="card">
      <h2 id="t-ops">Incident Ops Queue</h2>
      <div class="row" style="margin-bottom:8px;flex-wrap:wrap">
        <label class="small" id="ops-status-label" for="opsStatus">Status</label>
        <select id="opsStatus" style="min-width:160px">
          <option value="">all</option>
          <option value="new">new</option>
          <option value="triage">triage</option>
          <option value="assigned">assigned</option>
          <option value="investigating">investigating</option>
          <option value="awaiting_approval">awaiting_approval</option>
          <option value="ack">ack</option>
          <option value="confirmed">confirmed</option>
          <option value="false_positive">false_positive</option>
          <option value="ignored">ignored</option>
          <option value="closed">closed</option>
        </select>
        <label class="small"><input id="opsOverdue" type="checkbox"/> <span id="ops-overdue-label">overdue only</span></label>
        <button class="ghost" id="ops-refresh-btn" onclick="refreshOpsQueue()">Refresh</button>
        <button class="ghost" id="ops-group-btn" onclick="groupDuplicates()">Group Duplicates</button>
        <button class="ghost" id="ops-export-json-btn" onclick="generateReport('json')">Export JSON</button>
        <button class="ghost" id="ops-export-csv-btn" onclick="generateReport('csv')">Export CSV</button>
      </div>
      <div id="opsKpi" class="small" style="margin-bottom:8px"></div>
      <div id="opsMsg" class="small"></div>
      <div id="opsReportActions" class="row hidden" style="margin:8px 0;flex-wrap:wrap">
        <span id="ops-report-ready-label" class="small">Report ready:</span>
        <a id="opsReportViewLink" class="badge" href="#" target="_blank">View</a>
        <a id="opsReportDownloadLink" class="badge" href="#">Download</a>
        <button class="ghost" id="opsReportPreviewBtn" onclick="previewLastReport()">Preview</button>
      </div>
      <table>
        <thead><tr>
          <th>ID</th><th id="ops-th-time">Time</th><th id="ops-th-camera">Camera</th><th id="ops-th-event">Event</th><th id="ops-th-level">Level</th><th id="ops-th-status">Status</th><th id="ops-th-assignee">Assignee</th><th id="ops-th-due">Due</th><th id="ops-th-image">Image</th><th id="ops-th-actions">Actions</th>
        </tr></thead>
        <tbody id="opsBody"></tbody>
      </table>
    </div>
  </section>
</main>
<div class="overlay" id="overlay"><div class="spinner"></div></div>
<div class="overlay" id="opsOverlay">
  <div class="card" style="width:min(1040px,95vw);max-height:90vh;overflow:auto">
    <div class="row" style="justify-content:space-between;align-items:center">
      <h3 id="ops-modal-title" style="margin:0">Incident</h3>
      <button class="ghost" id="ops-cancel-top" onclick="closeOpsModal()">Close</button>
    </div>
    <div class="grid2" style="grid-template-columns:1.2fr 1fr;gap:12px">
      <div>
        <div class="small" id="ops-img-label">Image</div>
        <a id="opsImageLink" href="#" target="_blank"><img id="opsImage" class="snap" src="" style="max-height:52vh;object-fit:contain;background:#05080d"/></a>
      </div>
      <div>
        <div class="row"><label class="small" for="opsFStatus" id="ops-f-status-label">Status</label><select id="opsFStatus" style="min-width:180px">
          <option value="new">new</option><option value="triage">triage</option><option value="assigned">assigned</option>
          <option value="investigating">investigating</option><option value="awaiting_approval">awaiting_approval</option>
          <option value="ack">ack</option><option value="confirmed">confirmed</option><option value="false_positive">false_positive</option>
          <option value="ignored">ignored</option><option value="closed">closed</option>
        </select></div>
        <div class="row"><label class="small" for="opsFLevel" id="ops-f-level-label">Level</label><select id="opsFLevel" style="min-width:180px"><option>low</option><option>medium</option><option>high</option></select></div>
        <div class="row"><label class="small" for="opsFAssignee" id="ops-f-assignee-label">Assignee</label><input id="opsFAssignee" type="text" style="flex:1;background:#0f1620;border:1px solid #203041;color:#e9eef5;border-radius:8px;padding:7px"/></div>
        <div class="row"><label class="small" for="opsFType" id="ops-f-type-label">Type</label><input id="opsFType" type="text" style="flex:1;background:#0f1620;border:1px solid #203041;color:#e9eef5;border-radius:8px;padding:7px"/></div>
        <div class="row"><label class="small" for="opsFSite" id="ops-f-site-label">Site</label><input id="opsFSite" type="text" style="flex:1;background:#0f1620;border:1px solid #203041;color:#e9eef5;border-radius:8px;padding:7px"/></div>
        <div class="row"><label class="small" for="opsFRisk" id="ops-f-risk-label">Risk</label><input id="opsFRisk" type="number" min="0" max="100" style="width:120px;background:#0f1620;border:1px solid #203041;color:#e9eef5;border-radius:8px;padding:7px"/></div>
        <div class="row"><label class="small" for="opsFPolicy" id="ops-f-policy-label">Policy</label><input id="opsFPolicy" type="text" style="flex:1;background:#0f1620;border:1px solid #203041;color:#e9eef5;border-radius:8px;padding:7px"/></div>
        <div class="row"><label class="small" for="opsFRootCause" id="ops-f-root-label">Root Cause</label><input id="opsFRootCause" type="text" style="flex:1;background:#0f1620;border:1px solid #203041;color:#e9eef5;border-radius:8px;padding:7px"/></div>
        <div class="row"><label class="small" for="opsFCorrective" id="ops-f-corrective-label">Corrective Action</label><input id="opsFCorrective" type="text" style="flex:1;background:#0f1620;border:1px solid #203041;color:#e9eef5;border-radius:8px;padding:7px"/></div>
        <div class="row"><label class="small" for="opsFApprover" id="ops-f-approver-label">Approver</label><input id="opsFApprover" type="text" style="flex:1;background:#0f1620;border:1px solid #203041;color:#e9eef5;border-radius:8px;padding:7px"/></div>
        <div class="row"><label class="small" for="opsFNote" id="ops-f-note-label">Note</label><input id="opsFNote" type="text" style="flex:1;background:#0f1620;border:1px solid #203041;color:#e9eef5;border-radius:8px;padding:7px"/></div>
        <div class="row" style="margin-top:10px">
          <button id="ops-save-btn" onclick="saveOpsModal()">Save</button>
          <button class="ghost" id="ops-cancel-btn" onclick="closeOpsModal()">Cancel</button>
        </div>
      </div>
    </div>
    <div class="card" style="margin-top:12px;padding:8px">
      <div class="small" id="ops-timeline-label">Timeline</div>
      <div id="opsTimeline" class="small" style="max-height:160px;overflow:auto;white-space:pre-wrap"></div>
    </div>
  </div>
</div>
<div class="overlay" id="reportOverlay">
  <div class="card" style="width:min(980px,95vw);max-height:90vh;overflow:auto">
    <div class="row" style="justify-content:space-between;align-items:center">
      <h3 id="reportTitle" style="margin:0">Report Preview</h3>
      <button class="ghost" id="report-close-btn" onclick="closeReportViewer()">Close</button>
    </div>
    <div id="reportMeta" class="small" style="margin:8px 0"></div>
    <pre id="reportContent" class="report-view"></pre>
  </div>
</div>
<script>
let LANG = localStorage.getItem('lang') || 'en';
let OPS_ACTIVE_ID = null;
let OPS_LAST_REPORT = null;
let ACTIVE_CAM_ID = null;
let LIVE_SNAPSHOT_TIMER = null;
document.getElementById('langSel').value = LANG;

function showTab(which){
  const live = document.getElementById('panelLive');
  const multi = document.getElementById('panelMulti');
  const settings = document.getElementById('panelSettings');
  const ops = document.getElementById('panelOps');
  const t1 = document.getElementById('tabLive');
  const t2 = document.getElementById('tabMulti');
  const t3 = document.getElementById('tabSettings');
  const t4 = document.getElementById('tabOps');
  live.classList.add('hidden');
  multi.classList.add('hidden');
  settings.classList.add('hidden');
  ops.classList.add('hidden');
  t1.classList.add('ghost');
  t2.classList.add('ghost');
  t3.classList.add('ghost');
  t4.classList.add('ghost');
  if(which==='live'){ live.classList.remove('hidden'); t1.classList.remove('ghost'); }
  else if(which==='multi'){ multi.classList.remove('hidden'); t2.classList.remove('ghost'); }
  else if(which==='settings'){ settings.classList.remove('hidden'); t3.classList.remove('ghost'); }
  else { ops.classList.remove('hidden'); t4.classList.remove('ghost'); refreshOpsQueue(); }
}
function setLang(l){ localStorage.setItem('lang', l); LANG = l; applyLang(); }
async function applyLang(){
  const r = await fetch('/api/i18n?lang='+LANG); const t = await r.json();
  const map = {'t-title':'title','t-title-2':'title','t-live':'tab_live','t-grid':'grid_title','t-lang':'lang',
    't-overall':'overall','t-rt':'realtime_alerts','t-switch':'switch','t-settings':'tab_settings','t-ops':'tab_ops',
    'th-time':'time','th-event':'event','th-level':'level','th-image':'image'};
  for (const id in map){ const el=document.getElementById(id); if(el) el.textContent = t[map[id]]; }
  document.getElementById('tabLive').textContent = t.tab_live;
  document.getElementById('tabMulti').textContent = t.tab_multi;
  document.getElementById('tabSettings').textContent = t.tab_settings || 'Settings';
  document.getElementById('tabOps').textContent = t.tab_ops || 'Incident Ops';
  const xmap = {
    'ops-status-label':'ops_status','ops-overdue-label':'ops_overdue','ops-refresh-btn':'ops_refresh',
    'ops-group-btn':'ops_group','ops-export-json-btn':'ops_export_json','ops-export-csv-btn':'ops_export_csv',
    'ops-report-ready-label':'ops_report_ready','opsReportViewLink':'ops_report_view',
    'opsReportDownloadLink':'ops_report_download','opsReportPreviewBtn':'ops_report_preview',
    'ops-th-time':'time','ops-th-camera':'ops_camera','ops-th-event':'event','ops-th-level':'level',
    'ops-th-status':'ops_status','ops-th-assignee':'ops_assignee','ops-th-due':'ops_due','ops-th-image':'image',
    'ops-th-actions':'ops_actions','ops-modal-title':'ops_popup_title','ops-save-btn':'ops_popup_save',
    'ops-cancel-btn':'ops_popup_cancel','ops-cancel-top':'ops_popup_cancel','ops-timeline-label':'ops_popup_timeline',
    'ops-f-status-label':'ops_status','ops-f-level-label':'level','ops-f-assignee-label':'ops_assignee',
    'ops-f-type-label':'ops_type','ops-f-site-label':'ops_site','ops-f-risk-label':'ops_risk',
    'ops-f-policy-label':'ops_policy','ops-f-root-label':'ops_root_cause','ops-f-corrective-label':'ops_corrective',
    'ops-f-approver-label':'ops_approver','ops-f-note-label':'ops_note','ops-img-label':'image'
  };
  for (const id in xmap){ const el=document.getElementById(id); if(el) el.textContent = t[xmap[id]] || el.textContent; }
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
async function factoryReset(){
  if(!confirm('Factory reset camera setup and return to setup wizard?')) return;
  showOverlay(true);
  try{
    await fetch('/api/setup/factory_reset', {method:'POST'});
    window.location.href = '/setup';
  } finally {
    showOverlay(false);
  }
}
async function loadRepeatDuration(){
  try{
    const r = await fetch('/api/settings/repeat_duration');
    const d = await r.json();
    if(r.ok && d.ok){
      document.getElementById('repeatSec').value = Math.max(1, Math.round(d.seconds));
    }
  } catch(_e){ }
}
async function saveRepeatDuration(){
  const msg = document.getElementById('settingsMsg');
  const seconds = Math.max(1, Math.round(Number(document.getElementById('repeatSec').value || 1)));
  msg.textContent = 'Saving...';
  const r = await fetch('/api/settings/repeat_duration?seconds='+encodeURIComponent(seconds), {method:'POST'});
  const d = await r.json();
  if(!r.ok || !d.ok){
    msg.textContent = 'Failed to save repeat duration.';
    return;
  }
  msg.textContent = `Saved: ${seconds}s`;
}
async function opsPost(url, data){
  const body = new URLSearchParams();
  for (const k in data){
    if(data[k] !== undefined && data[k] !== null){
      body.append(k, String(data[k]));
    }
  }
  return fetch(url, {method:'POST', body});
}
function opsSetMsg(msg, bad=false){
  const el = document.getElementById('opsMsg');
  el.textContent = msg;
  el.style.color = bad ? '#ffb8b8' : '#9fb3c8';
}
function setLatestReport(reportName){
  OPS_LAST_REPORT = reportName || null;
  const wrap = document.getElementById('opsReportActions');
  const viewLink = document.getElementById('opsReportViewLink');
  const dlLink = document.getElementById('opsReportDownloadLink');
  if(!OPS_LAST_REPORT){
    wrap.classList.add('hidden');
    viewLink.href = '#';
    dlLink.href = '#';
    return;
  }
  const base = '/api/reports/' + encodeURIComponent(OPS_LAST_REPORT);
  viewLink.href = base + '?disposition=inline';
  dlLink.href = base + '?disposition=attachment';
  dlLink.setAttribute('download', OPS_LAST_REPORT);
  wrap.classList.remove('hidden');
}
function closeReportViewer(){
  document.getElementById('reportOverlay').style.display = 'none';
}
async function openReportViewer(reportName){
  const safeName = (reportName || '').trim();
  if(!safeName){
    opsSetMsg('No report selected.', true);
    return;
  }
  const r = await fetch('/api/reports/' + encodeURIComponent(safeName) + '/content');
  const d = await r.json();
  if(!r.ok || !d.ok){
    opsSetMsg('Could not load report preview.', true);
    return;
  }
  document.getElementById('reportTitle').textContent = 'Report Preview: ' + d.report;
  document.getElementById('reportMeta').textContent = 'Format: ' + (d.format || '-') + ' | Size: ' + (d.size_bytes || 0) + ' bytes';
  document.getElementById('reportContent').textContent = d.content || '';
  document.getElementById('reportOverlay').style.display = 'flex';
}
async function previewLastReport(){
  if(!OPS_LAST_REPORT){
    opsSetMsg('Generate a report first.', true);
    return;
  }
  await openReportViewer(OPS_LAST_REPORT);
}
async function opsAssign(id){
  const assignee = prompt('Assign incident to (username/role):', 'supervisor');
  if(!assignee){ return; }
  const dueMin = Number(prompt('Due in minutes (default 5):', '5') || '5');
  let dueAt = null;
  if(Number.isFinite(dueMin) && dueMin > 0){
    dueAt = new Date(Date.now() + dueMin * 60000).toISOString().replace('T',' ').slice(0,19);
  }
  const r = await opsPost('/api/incidents/'+id+'/assign', {assignee:assignee, actor:'ui', due_at:dueAt});
  if(!r.ok){ opsSetMsg('Assign failed for incident '+id, true); return; }
  opsSetMsg('Assigned incident '+id+' to '+assignee);
  refreshOpsQueue();
}
function closeOpsModal(){
  document.getElementById('opsOverlay').style.display = 'none';
  OPS_ACTIVE_ID = null;
}
async function openIncidentModal(id){
  OPS_ACTIVE_ID = id;
  const r = await fetch('/api/incidents/'+id);
  if(!r.ok){ opsSetMsg('Failed to load incident '+id, true); return; }
  const d = await r.json();
  document.getElementById('ops-modal-title').textContent = (window._i18n?.ops_popup_title || 'Incident') + ' #' + id;
  document.getElementById('opsFStatus').value = d.status || 'new';
  document.getElementById('opsFLevel').value = d.level || 'medium';
  document.getElementById('opsFAssignee').value = d.assignee || '';
  document.getElementById('opsFType').value = d.type || '';
  document.getElementById('opsFSite').value = d.site || '';
  document.getElementById('opsFRisk').value = (d.risk_score ?? '');
  document.getElementById('opsFPolicy').value = d.policy_clause || '';
  document.getElementById('opsFRootCause').value = d.root_cause_code || '';
  document.getElementById('opsFCorrective').value = d.corrective_action_code || '';
  document.getElementById('opsFApprover').value = d.approver || '';
  document.getElementById('opsFNote').value = '';
  const imgPath = d.image ? '/incidents/' + d.image : '';
  const img = document.getElementById('opsImage');
  const lnk = document.getElementById('opsImageLink');
  img.src = imgPath || ('/snapshot?camera_id=' + (d.camera_id || 0) + '&t=' + Date.now());
  lnk.href = img.src;
  const tl = Array.isArray(d.timeline) ? d.timeline : [];
  document.getElementById('opsTimeline').textContent = tl.length
    ? tl.map(x => `${x.ts} | ${x.actor||'system'} | ${x.from_status||'-'} -> ${x.to_status} ${x.note?('| '+x.note):''}`).join('\\n')
    : '-';
  document.getElementById('opsOverlay').style.display = 'flex';
}
async function saveOpsModal(){
  if(!OPS_ACTIVE_ID){ return; }
  const id = OPS_ACTIVE_ID;
  const data = {
    status: document.getElementById('opsFStatus').value,
    actor: 'ui',
    reviewer: 'operator',
    note: document.getElementById('opsFNote').value || '',
    level: document.getElementById('opsFLevel').value,
    type: document.getElementById('opsFType').value || '',
    policy_clause: document.getElementById('opsFPolicy').value || '',
    root_cause_code: document.getElementById('opsFRootCause').value || '',
    corrective_action_code: document.getElementById('opsFCorrective').value || '',
    approver: document.getElementById('opsFApprover').value || '',
    risk_score: document.getElementById('opsFRisk').value || ''
  };
  const r1 = await opsPost('/api/incidents/'+id+'/label', data);
  if(!r1.ok){ opsSetMsg('Update failed for incident '+id, true); return; }

  const assignee = (document.getElementById('opsFAssignee').value || '').trim();
  if(assignee){
    const r2 = await opsPost('/api/incidents/'+id+'/assign', {assignee:assignee, actor:'ui'});
    if(!r2.ok){ opsSetMsg('Saved details, but assignee update failed.', true); }
  }

  closeOpsModal();
  opsSetMsg('Incident '+id+' updated.');
  refreshOpsQueue();
}
async function opsLabel(id){
  const status = prompt('New status (triage/assigned/investigating/awaiting_approval/confirmed/false_positive/ignored/closed):', 'triage');
  if(!status){ return; }
  const r = await opsPost('/api/incidents/'+id+'/label', {status:status, actor:'ui', reviewer:'operator'});
  if(!r.ok){ opsSetMsg('Status update failed for incident '+id, true); return; }
  opsSetMsg('Updated incident '+id+' to '+status);
  refreshOpsQueue();
}
async function opsClose(id){
  const r = await opsPost('/api/incidents/'+id+'/close', {actor:'ui', reviewer:'operator'});
  if(!r.ok){ opsSetMsg('Close failed for incident '+id, true); return; }
  opsSetMsg('Closed incident '+id);
  refreshOpsQueue();
}
async function opsMerge(masterId){
  const childId = Number(prompt('Child incident id to merge into '+masterId+':', ''));
  if(!Number.isFinite(childId) || childId <= 0){ return; }
  const r = await opsPost('/api/incidents/'+masterId+'/merge', {child_id:childId, actor:'ui'});
  if(!r.ok){ opsSetMsg('Merge failed: '+childId+' -> '+masterId, true); return; }
  opsSetMsg('Merged '+childId+' into '+masterId);
  refreshOpsQueue();
}
async function groupDuplicates(){
  const r = await fetch('/api/incidents/group_duplicates?window_sec=45&iou_thr=0.35', {method:'POST'});
  const d = await r.json();
  if(!r.ok){ opsSetMsg('Grouping failed: '+(d.error||'unknown'), true); return; }
  opsSetMsg('Duplicate grouping merged '+d.merged+' incident(s).');
  refreshOpsQueue();
}
async function generateReport(fmt){
  const status = document.getElementById('opsStatus').value || null;
  const overdue = document.getElementById('opsOverdue').checked;
  const r = await opsPost('/api/reports/generate', {report_type:'ops_queue', fmt:fmt, status:status, overdue:overdue, limit:500});
  const d = await r.json();
  if(!r.ok || !d.ok){ opsSetMsg('Report generation failed', true); return; }
  opsSetMsg('Report ready: '+d.report);
  setLatestReport(d.report);
  await openReportViewer(d.report);
}
async function refreshOpsQueue(){
  const status = document.getElementById('opsStatus').value || '';
  const overdue = document.getElementById('opsOverdue').checked ? 'true' : 'false';
  const q = '/api/incidents/queue?limit=50&include_children=false&overdue='+overdue + (status ? '&status='+encodeURIComponent(status) : '');
  const r = await fetch(q);
  const arr = await r.json();
  const tb = document.getElementById('opsBody');
  tb.innerHTML = '';
  for(const it of arr){
    const tr = document.createElement('tr');
    const hasImage = !!it.image;
    const imageCell = hasImage
      ? `<a class="badge" href="/incidents/${it.image}" target="_blank">${window._i18n?.view || 'view'}</a>`
      : '-';
    tr.innerHTML = `<td>${it.id}</td>
      <td>${String(it.ts||'').split('.')[0]}</td>
      <td>${it.camera_id||''}</td>
      <td><span class="pill">${it.type||''}</span> ${it.label||''}</td>
      <td>${it.level||''}</td>
      <td>${it.status||''}</td>
      <td>${it.assignee||''}</td>
      <td>${it.due_at||''}</td>
      <td>${imageCell}</td>
      <td>
        <button class="ghost" onclick="openIncidentModal(${it.id})">${window._i18n?.ops_open || 'Open'}</button>
        <button class="ghost" onclick="opsAssign(${it.id})">${window._i18n?.ops_assign || 'Assign'}</button>
        <button class="ghost" onclick="opsClose(${it.id})">${window._i18n?.ops_close || 'Close'}</button>
        <button class="ghost" onclick="opsMerge(${it.id})">${window._i18n?.ops_merge || 'Merge'}</button>
      </td>`;
    tb.appendChild(tr);
  }
  const k = await fetch('/api/kpi/ops').then(x=>x.json());
  document.getElementById('opsKpi').textContent =
    `${window._i18n?.ops_kpi_backlog || 'Backlog'}: ${k.backlog ?? 0} | ${window._i18n?.ops_kpi_mtta || 'MTTA'}: ${k.mtta_seconds ?? '-'}s | ${window._i18n?.ops_kpi_mttr || 'MTTR'}: ${k.mttr_seconds ?? '-'}s | ${window._i18n?.ops_kpi_fp || 'False+ Rate'}: ${k.false_positive_rate ?? '-'}`;
}
function loadLive(){
  const live = document.getElementById('live');
  const loader = document.getElementById('liveLoader');
  const spin = document.getElementById('liveSpin');
  live.style.display='none';
  spin.style.display='block';
  loader.style.display='grid';

  const showFrame = ()=>{
    spin.style.display='none';
    live.style.display='block';
    loader.style.display='none';
  };

  const stopSnapshotLoop = ()=>{
    if (LIVE_SNAPSHOT_TIMER){
      clearInterval(LIVE_SNAPSHOT_TIMER);
      LIVE_SNAPSHOT_TIMER = null;
    }
  };

  const startSnapshotLoop = ()=>{
    stopSnapshotLoop();
    const camSel = document.getElementById('camSelect');
    const getCamId = ()=>{
      const v = camSel && camSel.value ? Number(camSel.value) : NaN;
      if (Number.isFinite(v) && v > 0) return v;
      if (Number.isFinite(ACTIVE_CAM_ID) && ACTIVE_CAM_ID > 0) return ACTIVE_CAM_ID;
      return 1;
    };
    const tick = ()=>{
      const cid = getCamId();
      live.src = '/snapshot?camera_id=' + encodeURIComponent(cid) + '&t=' + Date.now();
    };
    tick();
    LIVE_SNAPSHOT_TIMER = setInterval(tick, 350);
  };

  // For MJPEG, some browsers do not reliably fire onload for the first chunk.
  let checks = 0;
  const poll = setInterval(()=>{
    checks += 1;
    if ((live.naturalWidth && live.naturalWidth > 0) || (live.complete && live.currentSrc)) {
      clearInterval(poll);
      showFrame();
      return;
    }
    if (checks >= 25) { // ~5s fallback to snapshot mode
      clearInterval(poll);
      startSnapshotLoop();
    }
  }, 200);

  live.onload = ()=>{ clearInterval(poll); showFrame(); };
  live.onerror = ()=>{
    clearInterval(poll);
    startSnapshotLoop();
  };
  stopSnapshotLoop();
  live.src = '/stream?t=' + Date.now();
}
function fmtTemp(v){ return (v===null || v===undefined) ? '\\u2014' : (Math.round(v)+'\\u00b0C'); }
function fmtPct(v){ return (v===null || v===undefined) ? '\\u2014' : (v+'%'); }
async function refreshStats(){
  const r = await fetch('/api/status'); const s = await r.json();
  const t = window._i18n || {};
  const vram = (s.health.vram_percent!==null && s.health.vram_percent!==undefined)
      ? `${s.health.vram_percent}% (${s.health.vram_used_mib}/${s.health.vram_total_mib} MiB)` : '\\u2014';
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
    <div class="box">${t.gpu_temp||'GPU Temp'}: ${fmtTemp(s.health.gpu_temp)}</div>`;
  document.getElementById('stats').innerHTML = kpi;
  const ac = s.active_camera; const label = (t.current_cam||'Active camera');
  ACTIVE_CAM_ID = ac && ac.id ? Number(ac.id) : ACTIVE_CAM_ID;
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
setInterval(()=>{ if(!document.getElementById('panelOps').classList.contains('hidden')) refreshOpsQueue(); }, 5000);
(async ()=>{ showTab('live'); await applyLang(); await loadRepeatDuration(); await loadCams(); loadLive(); refreshStats(); refreshRT(); refreshGrid(); })();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    if not _is_configured():
        return RedirectResponse(url="/setup", status_code=307)
    return HOME_HTML


@app.get("/setup", response_class=HTMLResponse)
def setup_page():
    return SETUP_HTML


# ---- MJPEG streaming --------------------------------------------------------
@app.get("/stream")
def mjpeg_stream():
    return StreamingResponse(_gen_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


def _gen_mjpeg():
    boundary = b"--frame\r\n"
    headers = b"Content-Type: image/jpeg\r\n\r\n"
    placeholder = _no_signal()

    def gen():
        while not S.stop_event.is_set():
            with S.annotated_jpeg_lock:
                b = S.annotated_jpeg.get("jpeg")
            if b is not None:
                yield boundary + headers + b + b"\r\n"
            elif placeholder:
                yield boundary + headers + placeholder + b"\r\n"
            time.sleep(0.04)
    return gen()


# ---- Snapshot ----------------------------------------------------------------
@app.get("/snapshot")
def api_snapshot(camera_id: int):
    # prefer annotated snap for the active cam, else raw
    with S.annotated_snaps_lock:
        b = S.annotated_snaps.get(camera_id)
    if not b:
        with S.raw_snaps_lock:
            b = S.raw_snaps.get(camera_id)
    if b:
        return Response(content=b, media_type="image/jpeg")
    ph = _no_signal()
    return Response(content=ph, media_type="image/jpeg")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• REST APIs â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

from obsero.db import (cameras_all, camera_by_id, camera_upsert, camera_set_online,
                        alert_insert, alert_query, alert_update_status, audit,
                        precision_query, db_conn, incident_assign,
                        incident_transition, incident_merge, incident_queue,
                        incident_update_fields, incident_timeline, alert_get,
                        ops_kpi)


@app.get("/api/status")
def api_status():
    cams = cameras_all()
    con = db_conn()
    total = con.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
    online = con.execute("SELECT COUNT(*) FROM cameras WHERE online=1").fetchone()[0]
    type_counts = dict(con.execute("SELECT type, COUNT(*) FROM alerts GROUP BY type").fetchall())
    level_counts = dict(con.execute("SELECT level, COUNT(*) FROM alerts GROUP BY level").fetchall())
    con.close()
    return JSONResponse({
        "sites": 1,
        "servers": 1,
        "cameras_total": total,
        "cameras_online": online,
        "health": get_health_snapshot(),
        "alert_stats": {"type": type_counts, "level": level_counts},
        "active_camera": {
            "id": S.cam_mgr.active_camera_id if S.cam_mgr else None,
            "url": next((c["url"] for c in cams if S.cam_mgr and c["id"] == S.cam_mgr.active_camera_id), None),
            "name": next((c["name"] for c in cams if S.cam_mgr and c["id"] == S.cam_mgr.active_camera_id), None),
        },
    })


@app.get("/api/cameras")
def api_cameras():
    return JSONResponse(cameras_all())


@app.get("/api/setup/status")
def api_setup_status():
  return JSONResponse({"configured": _is_configured(), "cameras": cameras_all()})


@app.get("/api/settings/repeat_duration")
def api_get_repeat_duration():
  cfg = S.cfg
  if cfg is None:
    return JSONResponse({"ok": False, "error": "Config not loaded"}, status_code=503)
  return JSONResponse({"ok": True, "seconds": float(cfg.default_cooldown_sec)})


@app.post("/api/settings/repeat_duration")
def api_set_repeat_duration(seconds: float = 8.0):
  sec = max(1.0, min(float(seconds), 3600.0))

  if S.cfg is not None:
    S.cfg.default_cooldown_sec = sec

  try:
    raw: dict = {}
    if CONFIG_PATH.exists():
      with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw["default_cooldown_sec"] = sec
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
      yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=False)
  except Exception as exc:
    return JSONResponse({"ok": False, "error": f"Failed to persist config: {exc}"}, status_code=500)

  audit("system", "settings_repeat_duration", str(sec))
  return JSONResponse({"ok": True, "seconds": sec})


@app.get("/api/setup/webcams")
def api_setup_webcams(max_index: int = 12):
  try:
    from discover_cameras import discover_webcams
    max_index = max(1, min(max_index, 32))
    cams = discover_webcams(max_index)
    return JSONResponse({"ok": True, "cameras": cams})
  except Exception as exc:
    return JSONResponse({"ok": False, "error": str(exc), "cameras": []}, status_code=500)


@app.post("/api/setup/apply")
def api_setup_apply(webcam_index: int = 0):
  try:
    from discover_cameras import build_cameras_json, save_cameras_json

    selected = {
      "name": f"Webcam-{webcam_index}",
      "url": int(webcam_index),
      "gpu_id": 0,
      "max_fps": 12,
      "target_side": 640,
      "source_type": "webcam",
      "online": True,
    }

    selected["name"] = selected.get("name") or f"Webcam-{webcam_index}"
    selected["gpu_id"] = int(selected.get("gpu_id", 0))
    selected["max_fps"] = int(selected.get("max_fps", 12))
    selected["target_side"] = int(selected.get("target_side", 640))
    selected["source_type"] = "webcam"
    selected["online"] = True

    data = build_cameras_json([selected], nvr_settings=None)
    save_cameras_json(data)

    cfg_cam = CameraCfg(
      id=1,
      name=str(selected["name"]),
      url=str(selected["url"]),
      gpu_id=selected["gpu_id"],
      max_fps=selected["max_fps"],
      target_side=selected["target_side"],
    )
    if S.cfg is not None:
      S.cfg.cameras = [cfg_cam]

    camera_upsert(cfg_cam.name, cfg_cam.url, code="CFG-1", ptz_protocol="onvif", cid=None, online=False)
    con = db_conn()
    try:
      con.execute("DELETE FROM cameras WHERE code IS NULL OR code <> 'CFG-1'")
      con.commit()
      row = con.execute("SELECT id FROM cameras WHERE code='CFG-1' LIMIT 1").fetchone()
    finally:
      con.close()

    if not row:
      return JSONResponse({"ok": False, "error": "Failed to persist configured camera"}, status_code=500)

    db_id = int(row[0])
    if S.cam_mgr is not None:
      S.cam_mgr.restart_with({db_id: (cfg_cam.url, cfg_cam.max_fps, cfg_cam.target_side)}, db_id)

    with S.raw_snaps_lock:
      S.raw_snaps.clear()
    with S.annotated_snaps_lock:
      S.annotated_snaps.clear()

    audit("system", "setup_apply_webcam", str(cfg_cam.url))
    return JSONResponse({"ok": True, "camera": {"id": db_id, "name": cfg_cam.name, "url": cfg_cam.url}})
  except Exception as exc:
    return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/api/setup/factory_reset")
def api_setup_factory_reset():
  try:
    _write_empty_camera_config()
    con = db_conn()
    try:
      con.execute("DELETE FROM cameras")
      con.execute("DELETE FROM sqlite_sequence WHERE name='cameras'")
      con.commit()
    finally:
      con.close()

    if S.cam_mgr is not None:
      S.cam_mgr.stop_all()
      S.cam_mgr.active_camera_id = None
    if S.cfg is not None:
      S.cfg.cameras = []

    with S.raw_snaps_lock:
      S.raw_snaps.clear()
    with S.annotated_snaps_lock:
      S.annotated_snaps.clear()
    with S.annotated_jpeg_lock:
      S.annotated_jpeg["jpeg"] = None

    audit("system", "factory_reset", "camera_setup")
    return JSONResponse({"ok": True})
  except Exception as exc:
    return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/api/cameras")
def api_camera_add(name: str = Form(...), url: str = Form(...),
                   code: str = Form(None), ptz_protocol: str = Form("onvif")):
    camera_upsert(name, url, code, ptz_protocol, cid=None)
    audit("system", "camera_add", name)
    return PlainTextResponse("ok")


@app.put("/api/cameras/{cid}")
def api_camera_update(cid: int, name: str = Form(...), url: str = Form(...),
                      code: str = Form(None), ptz_protocol: str = Form("onvif")):
    camera_upsert(name, url, code, ptz_protocol, cid=cid)
    audit("system", "camera_update", str(cid))
    return PlainTextResponse("ok")


@app.delete("/api/cameras/{cid}")
def api_camera_delete(cid: int):
    con = db_conn()
    con.execute("DELETE FROM cameras WHERE id=?", (cid,))
    con.commit()
    con.close()
    audit("system", "camera_delete", str(cid))
    return PlainTextResponse("ok")


@app.post("/api/select_camera")
def api_select_camera(camera_id: int):
    cam = camera_by_id(camera_id)
    if not cam:
        return JSONResponse({"error": "not found"}, status_code=404)
    if S.cam_mgr:
        S.cam_mgr.switch_active(cam["id"])
    audit("system", "select_camera", str(cam["id"]))
    return PlainTextResponse("ok")


@app.get("/api/alerts")
def api_alerts(camera_id: int | None = None, level: str | None = None,
               type: str | None = None, t_from: str | None = None,
               t_to: str | None = None, status: str | None = None,
               limit: int = 50):
    rows = alert_query(limit=limit, camera_id=camera_id, level=level,
                       type=type, t_from=t_from, t_to=t_to, status=status)
    return JSONResponse(rows)


@app.get("/api/incidents/queue")
def api_incident_queue(status: str | None = None,
                       overdue: bool = False,
                       limit: int = 100,
                       include_children: bool = False):
    rows = incident_queue(status=status, overdue=overdue, limit=limit,
                          include_children=include_children)
    return JSONResponse(rows)


@app.get("/api/incidents/{incident_id}")
def api_incident_get(incident_id: int):
    row = alert_get(incident_id)
    if not row:
        return JSONResponse({"error": "not found"}, status_code=404)
    row["timeline"] = incident_timeline(incident_id)
    return JSONResponse(row)


@app.post("/api/incidents/{incident_id}/assign")
def api_incident_assign(incident_id: int,
                        assignee: str = Form(...),
                        actor: str = Form("operator"),
                        due_at: str = Form(None)):
    try:
        incident_assign(incident_id, assignee=assignee, actor=actor, due_at=due_at)
        # Move into assigned state if currently open in early stages.
        cur = alert_get(incident_id) or {}
        if cur.get("status") in {"new", "triage", "ack"}:
            incident_transition(incident_id, "assigned", actor=actor, note="assignment")
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    audit(actor, "incident_assign", f"{incident_id}:{assignee}")
    return JSONResponse({"ok": True})


@app.post("/api/incidents/{incident_id}/label")
def api_incident_label(incident_id: int,
                       status: str = Form(...),
                       actor: str = Form("operator"),
                       reviewer: str = Form("operator"),
                       note: str = Form(""),
                       level: str = Form(None),
                       type: str = Form(None),
                       root_cause_code: str = Form(None),
                       corrective_action_code: str = Form(None),
                       policy_clause: str = Form(None),
                       risk_score: int = Form(None),
                       approver: str = Form(None)):
    try:
        incident_update_fields(
            incident_id,
            actor=actor,
            level=level,
            type=type,
            root_cause_code=root_cause_code,
            corrective_action_code=corrective_action_code,
            policy_clause=policy_clause,
            risk_score=risk_score,
            approver=approver,
        )
        incident_transition(incident_id, status, actor=actor, note=note, reviewer=reviewer)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    audit(actor, "incident_label", f"{incident_id}:{status}")
    return JSONResponse({"ok": True})


@app.post("/api/incidents/{incident_id}/close")
def api_incident_close(incident_id: int,
                       actor: str = Form("operator"),
                       reviewer: str = Form("operator"),
                       note: str = Form("manual close"),
                       corrective_action_code: str = Form(None),
                       approver: str = Form(None)):
    try:
        incident_update_fields(
            incident_id,
            actor=actor,
            corrective_action_code=corrective_action_code,
            approver=approver,
        )
        incident_transition(incident_id, "closed", actor=actor, note=note, reviewer=reviewer)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    audit(actor, "incident_close", str(incident_id))
    return JSONResponse({"ok": True})


@app.post("/api/incidents/{incident_id}/merge")
def api_incident_merge(incident_id: int,
                       child_id: int = Form(...),
                       actor: str = Form("operator")):
    try:
        incident_merge(master_id=incident_id, child_id=child_id, actor=actor)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    audit(actor, "incident_merge", f"{child_id}->{incident_id}")
    return JSONResponse({"ok": True})


@app.post("/api/incidents/group_duplicates")
def api_incident_group_duplicates(window_sec: int = 45,
                                  iou_thr: float = 0.35,
                                  open_only: bool = True):
    window_sec = max(1, min(int(window_sec), 3600))
    iou_thr = max(0.0, min(float(iou_thr), 1.0))

    sql = """
        SELECT id, ts, camera_id, type, bbox, status
          FROM alerts
         WHERE (merged_into IS NULL OR merged_into=0)
    """
    args: list = []
    if open_only:
      sql += " AND status <> 'closed'"
    sql += " ORDER BY camera_id, type, ts ASC"

    con = db_conn()
    rows = [dict(r) for r in con.execute(sql, args).fetchall()]
    con.close()

    masters: dict[tuple[int | None, str], list[dict]] = {}
    merged = 0
    for row in rows:
        cam = row.get("camera_id")
        et = str(row.get("type") or "")
        ts = _parse_db_ts(row.get("ts"))
        if ts is None:
            continue
        bbox = _bbox_from_json(row.get("bbox"))
        key = (cam, et)
        bucket = masters.setdefault(key, [])
        chosen: dict | None = None

        # Only compare against recent masters inside the same grouping key.
        keep: list[dict] = []
        for m in bucket:
            mts = m["ts"]
            if (ts - mts).total_seconds() <= window_sec:
                keep.append(m)
                if chosen is None:
                    if _bbox_iou(bbox, m["bbox"]) >= iou_thr:
                        chosen = m
        bucket[:] = keep

        if chosen is not None:
            try:
                incident_merge(master_id=int(chosen["id"]), child_id=int(row["id"]), actor="dedupe")
                merged += 1
            except Exception:
                continue
        else:
            bucket.append({"id": int(row["id"]), "ts": ts, "bbox": bbox})

    audit("dedupe", "group_duplicates", f"merged={merged};window={window_sec};iou={iou_thr}")
    return JSONResponse({"ok": True, "merged": merged, "window_sec": window_sec, "iou_thr": iou_thr})


# ---- NEW: label (confirm/false_positive/ignore/close) ----------------------
@app.post("/api/alerts/label")
def api_alert_label(alert_id: int = Form(...),
                    status: str = Form(...),
                    reviewer: str = Form("operator")):
    try:
        alert_update_status(alert_id, status, reviewer)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    audit(reviewer, "alert_label", f"{alert_id}:{status}")
    return PlainTextResponse("ok")


# ---- backward compat: /api/alerts/confirm -----------------------------------
@app.post("/api/alerts/confirm")
def api_alert_confirm(alert_id: int = Form(...),
                      status: str = Form("ack"),
                      reviewer: str = Form("operator")):
    try:
        alert_update_status(alert_id, status, reviewer)
    except ValueError:
        pass
    audit(reviewer, "alert_confirm", f"{alert_id}:{status}")
    return PlainTextResponse("ok")


# ---- NEW: precision metrics -------------------------------------------------
@app.get("/api/metrics/precision")
def api_precision(type: str | None = None, since: str | None = None,
                  min_reviewed: int = 1):
    return JSONResponse(precision_query(event_type=type, since=since,
                                        min_reviewed=min_reviewed))


@app.get("/api/kpi/ops")
def api_kpi_ops(since: str | None = None):
    return JSONResponse(ops_kpi(since=since))


@app.post("/api/reports/generate")
def api_report_generate(report_type: str = Form("shift_summary"),
                        fmt: str = Form("json"),
                        since: str | None = Form(None),
                        status: str | None = Form(None),
                        overdue: bool = Form(False),
                        limit: int = Form(500)):
    rows = incident_queue(status=status, overdue=overdue, limit=limit,
                          include_children=True)
    kpi = ops_kpi(since=since)
    payload = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "report_type": report_type,
        "filters": {"since": since, "status": status, "overdue": overdue, "limit": limit},
        "kpi": kpi,
        "incidents": rows,
    }

    safe_type = "".join(ch for ch in report_type if ch.isalnum() or ch in ("_", "-")) or "report"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt.lower() == "csv":
        out = REPORTS_DIR / f"{safe_type}_{ts}.csv"
        fieldnames = [
            "id", "ts", "camera_id", "site", "type", "label", "level", "status", "assignee",
            "approver", "risk_score", "due_at", "closed_at", "merged_into",
        ]
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})
        return JSONResponse({"ok": True, "report": out.name, "format": "csv", "path": str(out)})

    out = REPORTS_DIR / f"{safe_type}_{ts}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return JSONResponse({"ok": True, "report": out.name, "format": "json", "path": str(out)})


@app.get("/api/reports/{report_name}")
def api_report_download(report_name: str, disposition: str = "attachment"):
    safe = Path(report_name).name
    p = REPORTS_DIR / safe
    if not p.exists() or not p.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    media = "application/json" if p.suffix.lower() == ".json" else "text/csv"
    if disposition == "inline":
        return FileResponse(str(p), media_type=media)
    return FileResponse(str(p), media_type=media, filename=p.name)


@app.get("/api/reports/{report_name}/content")
def api_report_content(report_name: str):
    safe = Path(report_name).name
    p = REPORTS_DIR / safe
    if not p.exists() or not p.is_file():
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = p.read_text(encoding="latin-1", errors="replace")
    fmt = "json" if p.suffix.lower() == ".json" else "csv"
    return JSONResponse({
        "ok": True,
        "report": p.name,
        "format": fmt,
        "size_bytes": p.stat().st_size,
        "content": text,
    })


# ---- Upload / alarm levels / logs -------------------------------------------
@app.post("/api/alerts/upload")
def api_alert_upload(camera_id: int = Form(None), level: str = Form("manual"),
                     type: str = Form("MANUAL"), site: str = Form("Site-A"),
                     file: UploadFile = File(...)):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    name = f"{ts.replace(':', '-').replace(' ', '_')}_manual.jpg"
    data = file.file.read()
    (INCIDENTS_DIR / name).write_bytes(data)
    alert_insert(ts, camera_id, level, type, "manual_upload", 1.0, [0, 0, 0, 0], name, site, "new")
    audit("operator", "alert_upload", name)
    return PlainTextResponse("ok")


@app.get("/api/alarm_levels")
def api_levels_get():
    con = db_conn()
    rows = con.execute("SELECT event_type, level FROM alarm_levels").fetchall()
    con.close()
    return JSONResponse([{"event_type": r["event_type"], "level": r["level"]} for r in rows])


@app.post("/api/alarm_levels")
def api_levels_set(event_type: str = Form(...), level: str = Form(...)):
    con = db_conn()
    con.execute("INSERT INTO alarm_levels(event_type,level) VALUES(?,?) ON CONFLICT(event_type) DO UPDATE SET level=excluded.level",
                (event_type, level))
    con.commit()
    con.close()
    audit("admin", "alarm_level_set", f"{event_type}={level}")
    return PlainTextResponse("ok")


@app.get("/api/logs")
def api_logs(limit: int = 200):
    con = db_conn()
    rows = con.execute("SELECT * FROM audit_logs ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return JSONResponse([dict(r) for r in rows])


@app.get("/api/ping")
def api_ping():
    return PlainTextResponse("pong")


if __name__ == "__main__":
  raise SystemExit(
    "Do not run obsero/api.py directly. Start the app with: python run.py"
  )
