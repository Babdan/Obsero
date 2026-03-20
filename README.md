# Obsero — Enterprise Safety Panel (Open Source)

Real-time multi-camera AI safety monitoring for factories.  
Detects PPE violations, smoking, phone use, fire/smoke, and falls using YOLO models accelerated by TensorRT.

---

## Quick Start (Windows)

```powershell
# 1. Install dependencies
pip install -r requirements.txt
pip install pyyaml

# 2. Edit configuration
#    Adjust for your NVR / cameras / GPUs:
notepad configs\system.yaml

# 3. Run
python run.py
```

The web panel opens at **http://127.0.0.1:9009**.

> **Legacy entrypoint** `nvr_trt_multicam_pilot.py` still works but is no longer the
> recommended way to run.  Use `run.py` for all new deployments.

---

## Configuration — `configs/system.yaml`

All system behaviour is controlled by a single YAML file.

### GPUs

```yaml
gpus:
  - id: 0          # CUDA device index
    weight: 1.0    # relative capacity (informational)
  - id: 1
    weight: 0.7
```

List every GPU that should run inference.  Each camera is pinned to exactly
one GPU via `gpu_id` in the camera list.

### Cameras

```yaml
cameras:
  - id: 1
    name: "CH1"
    url: "rtsp://admin:pass@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
    gpu_id: 0        # which GPU processes this camera's frames
    max_fps: 12      # capture frame rate
    target_side: 640 # resize longest edge before sending to models
```

Up to 16 (or more) RTSP cameras supported.  Assign cameras 1–8 to GPU 0 and
9–16 to GPU 1 for balanced multi-GPU deployments.

### Models

```yaml
models:
  - key: ppe
    stem: helmet_vest_best
    conf_thr: 0.60
    keywords: ["nohat", "novest"]
    cadence_every_n: 1   # feed every Nth frame per camera
```

`cadence_every_n` is **per camera** — each camera's counter is independent.

### Temporal Rules

```yaml
rules:
  ppe:
    min_consecutive_frames: 3    # streak of positive frames required
    window_sec: 5.0              # sliding window size
    min_ratio: 0.6               # (positive / total) inside window
    cooldown_sec: 10.0           # suppress re-triggers for this long
```

An incident is only created when **both** the streak **and** the ratio gate
pass, reducing false positives to meet the 80 % precision target.

### Camera Offline Detection

```yaml
camera_offline_timeout_sec: 15.0
```

If no frame is received from a camera for this many seconds, it is marked
`online=0` in the database.

---

## Multi-GPU Mapping

The system spawns **one model worker per (model_key, gpu_id) pair** that is
actually needed by camera assignments.

| Cameras     | `gpu_id` | Workers spawned on GPU |
|-------------|----------|------------------------|
| CH1 … CH8  | 0        | ppe_gpu0, smoke_gpu0, phone_gpu0, fire_smoke_gpu0, fall_gpu0 |
| CH9 … CH16 | 1        | ppe_gpu1, smoke_gpu1, phone_gpu1, fire_smoke_gpu1, fall_gpu1 |

Frames from a camera are routed **only** to workers on the camera's assigned
GPU.  No NVLink required.

Inside each worker, `torch.cuda.set_device(gpu_id)` is called, and
`model.predict(device=gpu_id)` is used.

---

## TRT Engine Naming Convention

For heterogeneous GPUs (e.g., RTX 3090 = sm86, RTX 4090 = sm89), build
separate TRT engines:

```
models/
  helmet_vest_best_sm86.engine   ← used on GPU 0 (Ampere)
  helmet_vest_best_sm89.engine   ← used on GPU 1 (Ada Lovelace)
  helmet_vest_best.engine        ← generic fallback
  helmet_vest_best.pt            ← PyTorch fallback
```

**Search order** per worker:

1. `models/{stem}_{smXX}.engine`  — architecture-specific TRT engine
2. `models/{stem}.engine`         — generic TRT engine
3. `models/{stem}.pt`             — PyTorch weights

If a TRT engine fails to load at runtime (incompatible build), the worker
automatically falls back to `.pt` and logs a warning.

### Building TRT engines

```bash
# On a machine with GPU sm86:
python export_to_trt.py --weights models/helmet_vest_best.pt --device 0
mv models/helmet_vest_best.engine models/helmet_vest_best_sm86.engine

# On a machine with GPU sm89:
python export_to_trt.py --weights models/helmet_vest_best.pt --device 0
mv models/helmet_vest_best.engine models/helmet_vest_best_sm89.engine
```

---

## Precision Metric — Ensuring 80 % Accuracy

"80 % accuracy" is operationally enforced as **incident precision ≥ 0.80**,
measured from reviewed alerts.

### Labelling alerts

```http
POST /api/alerts/label
Content-Type: application/x-www-form-urlencoded

alert_id=42&status=confirmed&reviewer=john
```

Valid statuses: `new`, `ack`, `confirmed`, `false_positive`, `ignored`, `closed`.

### Querying precision

```http
GET /api/metrics/precision?type=PPE&since=2026-01-01&min_reviewed=10
```

Response:

```json
{
  "confirmed": 8,
  "false_positive": 2,
  "total_reviewed": 10,
  "precision": 0.8,
  "min_reviewed_met": true
}
```

Tune `conf_thr`, `min_consecutive_frames`, `min_ratio`, and `cooldown_sec`
in `system.yaml` until precision ≥ 0.80 for each event type.

---

## Evidence Storage

Each incident saves **two** images in `incidents/`:

- `*_crop.jpg` — bounding-box crop of the detected object
- `*_full.jpg` — full-frame unannotated snapshot

Both filenames are stored in the `alerts` table (`image` = crop,
`full_image` = full).

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/stream` | MJPEG live stream |
| GET | `/snapshot?camera_id=N` | Latest JPEG snapshot |
| GET | `/api/cameras` | List all cameras |
| POST | `/api/cameras` | Add camera (form: name, url, code, ptz_protocol) |
| PUT | `/api/cameras/{id}` | Update camera |
| DELETE | `/api/cameras/{id}` | Delete camera |
| POST | `/api/select_camera?camera_id=N` | Switch live view |
| GET | `/api/alerts` | Query alerts (params: camera_id, level, type, t_from, t_to, status, limit) |
| POST | `/api/alerts/label` | Label alert (form: alert_id, status, reviewer) |
| POST | `/api/alerts/confirm` | Legacy confirm (form: alert_id, status, reviewer) |
| POST | `/api/alerts/upload` | Upload manual alert image |
| GET | `/api/metrics/precision` | Precision metric (params: type, since, min_reviewed) |
| GET | `/api/status` | System health + camera status |
| GET | `/api/alarm_levels` | Get alarm level config |
| POST | `/api/alarm_levels` | Set alarm level |
| GET | `/api/logs` | Audit logs |
| GET | `/api/i18n?lang=en` | i18n strings |
| GET | `/api/ping` | Health check |

---

## Project Structure

```
run.py                    ← entrypoint (uvicorn in main thread)
configs/
  system.yaml             ← all configuration
obsero/
  __init__.py
  config.py               ← YAML loader + dataclasses
  db.py                   ← SQLite DB layer + precision query
  models.py               ← weight resolution (multi-GPU TRT)
  rules.py                ← TemporalGate + GateManager
  workers.py              ← camera procs, model workers, router, composer
  api.py                  ← FastAPI app + all routes + HTML UI
models/
  *.engine / *.pt         ← YOLO weights
incidents/                ← saved evidence images
data/
  panel.db                ← SQLite database
```

### Legacy files

| File | Purpose |
|------|---------|
| `nvr_trt_multicam_pilot.py` | Previous monolithic entrypoint (Dahua NVR scan) |
| `trt_camera_test.py` | Single-camera TRT demo |
| `export_to_trt.py` | Export `.pt` → `.engine` |
| `check_torch.py` | Verify PyTorch + CUDA |

---

## Database Schema (v2)

- **cameras**: id, name, code, url, online, ptz_protocol, created_at
- **alerts**: id, ts, camera_id, level, type, label, conf, bbox (JSON),
  image, **full_image**, **model_key**, **gpu_id**, **rule_json**, site,
  status, reviewer, reviewed_at
- **audit_logs**: id, ts, actor, action, details
- **alarm_levels**: id, event_type, level
- **users**: id, username, role
- **patrol_tasks**: id, name, camera_ids, patrol_type, frequency, …

---

## Architecture

### Process Model
- **Main Thread**: Uvicorn web server (Windows-safe)
- **Worker Thread**: Backend bootstrap (spawns everything below)
- **Camera Processes** (multiprocessing): One per RTSP camera
- **Model Workers** (multiprocessing): One per (model_key × gpu_id)
- **Router Thread**: Distributes camera frames via FanOut (per-camera cadence)
- **Collector Thread**: Temporal gating → incident saving → DB alerts
- **Composer Thread**: Annotates active camera frame → MJPEG
- **Offline Monitor Thread**: Marks cameras offline after timeout

### Data Flow
```
RTSP Camera
       ↓
Camera Process (JPEG encode, bounded queue)
       ↓
Router → FanOut (per-camera cadence, GPU-routed)
       ↓
Model Worker (TRT/PT inference on assigned GPU)
       ↓
Results Collector (temporal gate → save crop + full-frame → DB)
       ↓
Composer (annotate → MJPEG stream)
       ↓
Web Browser (dashboard + alerts + precision metrics)
```

---

## Requirements

```
torch>=2.0.0
ultralytics>=8.0.0
supervision>=0.18.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
opencv-python>=4.8.0
numpy>=1.24.0
psutil>=5.9.0
pynvml>=11.500.0
pyyaml>=6.0
```

---

## Troubleshooting

### Cameras showing "NO SIGNAL"
1. Test RTSP directly: `ffplay "rtsp://admin:pass@ip:554/…"`
2. Check NVR IP/credentials in `configs/system.yaml`
3. Ensure network connectivity and firewall rules

### Low FPS or stuttering
1. Increase model cadence (`cadence_every_n`) in system.yaml
2. Check GPU utilisation: `nvidia-smi`
3. Reduce `max_fps` or increase `target_side`

### Missing GPU acceleration
1. Run `python check_torch.py`
2. Ensure NVIDIA driver installed: `nvidia-smi`
3. Reinstall torch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

---

## License

Obsero Open Source Edition — Safety-first, production-ready monitoring.

---

**Version**: 2.0  
**Last Updated**: July 2025  
**Tested On**: Python 3.12, NVIDIA Ampere (sm86) dual-GPU, Windows 10/11
