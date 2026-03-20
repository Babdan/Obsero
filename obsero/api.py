"""
obsero.api — FastAPI application, all routes, HTML UI.

Routes preserved:
  /  /stream  /snapshot  /api/cameras  /api/alerts  /api/status  /incidents/*

New routes:
  POST /api/alerts/label          — confirm / false_positive / ignore / close
  GET  /api/metrics/precision     — operational precision measurement
"""

from __future__ import annotations

import datetime, json, queue, signal, threading, time, traceback, sys, os, warnings
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil

from fastapi import FastAPI, Response, UploadFile, File, Form
from fastapi.responses import (HTMLResponse, StreamingResponse,
                                JSONResponse, PlainTextResponse, FileResponse)
from fastapi.staticfiles import StaticFiles

# ── paths ──
ROOT = Path(__file__).resolve().parent.parent
INCIDENTS_DIR = ROOT / "incidents"
STATIC_DIR = ROOT / "static"
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ── GPU monitoring ──
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

# ═══════════════════════════════════════════════════════════════════════════════
# App-level state — set by bootstrap (run.py) before uvicorn starts.
# ═══════════════════════════════════════════════════════════════════════════════
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

S = _AppState()

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI application
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="Obsero Safety Panel")
app.mount("/incidents", StaticFiles(directory=str(INCIDENTS_DIR)), name="incidents")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

FALLBACK_FAVICON_SVG = b"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
<rect width='64' height='64' fill='#0b0f14'/><path d='M36 6L14 36h14l-6 22 28-36H36z' fill='#ff7a1a'/>
</svg>"""


@app.get("./static/favicon.ico")
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
           "lang":"Language","english":"English","chinese":"中文","turkish":"Türkçe","default_cam":"Default Camera"},
    "zh": {"title":"Obsero 安全面板","tab_live":"实时画面","tab_multi":"多路画面","switch":"切换到该摄像头",
           "overall":"总体状态","sites":"场站","hosts":"分析主机","devices":"设备","online":"在线","offline":"离线",
           "cpu":"CPU","ram":"内存","gpu":"显卡","gpu_util":"显卡占用","vram":"显存","gpu_temp":"显卡温度","cpu_temp":"CPU温度",
           "realtime_alerts":"实时告警","grid_title":"分屏（快照，2秒刷新）","current_cam":"当前摄像头","view":"查看",
           "time":"时间","event":"事件","level":"级别","image":"图像","lang":"语言","english":"English","chinese":"中文","turkish":"Türkçe","default_cam":"默认相机"},
    "tr": {"title":"Obsero Güvenlik Paneli","tab_live":"Canlı Görüntü","tab_multi":"Çoklu Kamera","switch":"Kameraya Geç",
           "overall":"Genel Durum","sites":"Saha","hosts":"Analiz Sunucuları","devices":"Cihazlar","online":"çevrimiçi","offline":"çevrimdışı",
           "cpu":"CPU","ram":"RAM","gpu":"GPU","gpu_util":"GPU Kullanımı","vram":"VRAM","gpu_temp":"GPU Sıcaklığı","cpu_temp":"CPU Sıcaklığı",
           "realtime_alerts":"Anlık Alarmlar","grid_title":"Çoklu Görünüm (anlık görüntü, 2 sn)","current_cam":"Aktif kamera","view":"gör",
           "time":"Zaman","event":"Olay","level":"Seviye","image":"Görüntü","lang":"Dil","english":"English","chinese":"中文","turkish":"Türkçe","default_cam":"Varsayılan Kamera"},
}


@app.get("/api/i18n")
def api_i18n(lang: str = "en"):
    return JSONResponse(I18N.get(lang, I18N["en"]))


# ---------- Home HTML ---------------------------------------------------------
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
  const map = {'t-title':'title','t-title-2':'title','t-live':'tab_live','t-grid':'grid_title','t-lang':'lang',
    't-overall':'overall','t-rt':'realtime_alerts','t-switch':'switch',
    'th-time':'time','th-event':'event','th-level':'level','th-image':'image'};
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


# ════════════════════════════ REST APIs ══════════════════════════════════════

from obsero.db import (cameras_all, camera_by_id, camera_upsert, camera_set_online,
                        alert_insert, alert_query, alert_update_status, audit,
                        precision_query, db_conn)


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
