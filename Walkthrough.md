# Safety Vision Project — Complete Analysis

## Folder Overview

The `Safety Vision Project` root contains **two sub-projects**:

| Sub-folder | Purpose |
|---|---|
| **Obsero - Open Source** | Production-grade multi-camera AI safety monitoring system (Python) |
| **Safety Vision Project - China** | Hardware specs, contracts, OS images, API docs for a China-based deployment |

---

## 1. Obsero — Open Source (The Main Codebase)

### What It Does

Real-time factory safety monitoring via RTSP cameras. Detects **5 violation types** using YOLO models accelerated by TensorRT:

| Model Key | Detects | Confidence | Cadence |
|---|---|---|---|
| `ppe` | No hardhat, no vest | 0.60 | Every frame |
| `smoke` | Cigarette/smoking | 0.60 | Every 2nd frame |
| `phone` | Phone/mobile use | 0.50 | Every 2nd frame |
| `fire_smoke` | Fire and smoke | 0.75 | Every 3rd frame |
| `fall` | Person fallen | 0.85 | Every 2nd frame |

### Architecture

```
Main Thread (Uvicorn web server)
 └─ Worker Thread (backend bootstrap)
      ├─ Camera Processes (1 per RTSP camera, multiprocessing)
      ├─ Model Workers (1 per model×GPU, multiprocessing)
      ├─ Router Thread (distributes frames via FanOut)
      ├─ Collector Thread (temporal gating → incident saving → DB)
      ├─ Composer Thread (annotates frames → MJPEG stream)
      └─ Offline Monitor Thread (marks cameras offline after timeout)
```

### Tech Stack

| Layer | Technology |
|---|---|
| ML/CV | PyTorch ≥2.0, Ultralytics YOLOv8, TensorRT, OpenCV |
| Web | FastAPI + Uvicorn (port 9009) |
| DB | SQLite (WAL mode, thread-safe) |
| Monitoring | psutil, pynvml (GPU monitoring) |
| Annotations | Supervision library |

### Source Files

| File | Lines | Purpose |
|---|---|---|
| [run.py](file:///c:/Users/PC/Desktop/Safety%20Vision%20Project/Obsero%20-%20Open%20Source/run.py) | 274 | Entrypoint — boots Uvicorn + backend |
| [obsero/config.py](file:///c:/Users/PC/Desktop/Safety%20Vision%20Project/Obsero%20-%20Open%20Source/obsero/config.py) | 238 | YAML config loader + dataclasses |
| [obsero/db.py](file:///c:/Users/PC/Desktop/Safety%20Vision%20Project/Obsero%20-%20Open%20Source/obsero/db.py) | 267 | SQLite DB layer (cameras, alerts, audit, precision) |
| [obsero/models.py](file:///c:/Users/PC/Desktop/Safety%20Vision%20Project/Obsero%20-%20Open%20Source/obsero/models.py) | 54 | Weight resolution (TRT → PT fallback, multi-GPU) |
| [obsero/rules.py](file:///c:/Users/PC/Desktop/Safety%20Vision%20Project/Obsero%20-%20Open%20Source/obsero/rules.py) | 114 | Temporal gating (streak + ratio + cooldown) |
| [obsero/workers.py](file:///c:/Users/PC/Desktop/Safety%20Vision%20Project/Obsero%20-%20Open%20Source/obsero/workers.py) | 537 | Camera procs, model workers, router, composer |
| [obsero/api.py](file:///c:/Users/PC/Desktop/Safety%20Vision%20Project/Obsero%20-%20Open%20Source/obsero/api.py) | 603 | FastAPI routes + embedded HTML/CSS/JS dashboard |
| [discover_cameras.py](file:///c:/Users/PC/Desktop/Safety%20Vision%20Project/Obsero%20-%20Open%20Source/discover_cameras.py) | 393 | NVR auto-discovery + webcam fallback → cameras.json |

### Key Features
- **Multi-GPU** — spawns workers per (model, GPU) pair; cameras pinned to specific GPUs
- **TRT engine fallback** — searches for architecture-specific [.engine](file:///c:/Users/PC/Desktop/Safety%20Vision%20Project/Obsero%20-%20Open%20Source/models/fall_best.engine) → generic [.engine](file:///c:/Users/PC/Desktop/Safety%20Vision%20Project/Obsero%20-%20Open%20Source/models/fall_best.engine) → [.pt](file:///c:/Users/PC/Desktop/Safety%20Vision%20Project/Obsero%20-%20Open%20Source/models/fall_best.pt)
- **Temporal gating** — incidents only fire when both streak AND ratio thresholds pass (reduces false positives to ≥80% precision)
- **Evidence storage** — saves both bounding-box crop + full-frame snapshot per incident
- **NVR auto-discovery** — subnet scan for Dahua NVR devices, writes [cameras.json](file:///c:/Users/PC/Desktop/Safety%20Vision%20Project/Obsero%20-%20Open%20Source/configs/cameras.json)
- **i18n** — English, Chinese (中文), Turkish (Türkçe)
- **REST API** — full CRUD for cameras, alerts, labelling, precision metrics, audit logs

### Pre-trained Model Weights (in `models/`)

6 models × 3 formats each (.pt, .onnx, .engine) = **18 files, ~2.5 GB total**

### Legacy/Archive Files

7 older scripts in `arhive/` (pilot webcam scripts, earlier panel versions, single-camera test)

---

## 2. Safety Vision Project — China (Non-Code Assets)

### Structure

| Folder | Contents |
|---|---|
| **API/** | [interface-document.pdf](file:///c:/Users/PC/Desktop/Safety%20Vision%20Project/Safety%20Vision%20Project%20-%20China/API/interface-document.pdf) — API integration spec |
| **NVR Equipment Specs/** | Dahua datasheets for NVR and IP cameras (Available + Imported devices) |
| **OS/** | ISO images (Win11 24H2/25H2, Ubuntu 20.04/22.04, Proxmox VE 9.1) + Rufus |
| **Quotes/** | Business contracts, quotation spreadsheets (UESTCO gas safety project), field test plans, benchmark stats |
| **Setup/** | DeepStream 7.1 archive, server project, setup instructions (Step1.pdf, Step2.pdf, Demo Prep Guide) |

### Key Business Documents (in Quotes/)
- Partnership agreement (Turkey + Europe territory)
- UESTCO gas safety quotations (Chinese + English versions)
- Field test plan & meeting notes (Aug 2025)
- Contract notes with review comments (territory expansion, feature additions, branding terms)

### Hardware Specs
- **NVR**: Dahua NVR4232-4KS2-L (32ch), DH-NVR4216-M (16ch)
- **Cameras**: IPC2122LB-SF40-A (bullet), DH-SD-6C3223-HNY (PTZ dome)

---

## Summary Assessment

| Aspect | Status |
|---|---|
| **Code quality** | Well-structured, modular Python with clear separation of concerns |
| **Production readiness** | Yes — multi-GPU, temporal gating, precision metrics, audit trail |
| **Documentation** | Excellent README (353 lines), inline docstrings throughout |
| **Tests** | ⚠️ No test suite found |
| **Config management** | Clean YAML-based with JSON overlay for camera discovery |
| **UI** | Functional embedded dashboard (dark theme, KPI boxes, MJPEG stream, multi-cam grid) |
| **i18n** | 3 languages (EN, ZH, TR) |
| **Deployment** | Windows-first (spawn multiprocessing), tested on Python 3.12 + NVIDIA Ampere |
