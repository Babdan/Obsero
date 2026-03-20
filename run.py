#!/usr/bin/env python3
"""
run.py — Obsero Enterprise Safety Panel entrypoint.

Windows-stable:
  • Uvicorn runs in the MAIN thread (no signal issues).
  • Backend (models / cameras / router / composer) boots in a worker thread.
  • multiprocessing start method = spawn.

Usage:
    python run.py                          # uses configs/system.yaml
    python run.py --config my_config.yaml  # custom config
    python run.py --port 9009              # override port
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
import traceback
import webbrowser
from collections import deque
from pathlib import Path

# Force RTSP-over-TCP + 5 s socket timeout for OpenCV/FFmpeg
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|stimeout;5000000")

import uvicorn

# ── Obsero modules ──
from obsero.config import load_config, SystemConfig
from obsero.db import (db_init, cameras_all, camera_upsert, camera_set_online,
                        audit, db_conn)
from obsero.rules import GateManager
from obsero.workers import (
    FanOut, MultiCamProcManager,
    mp_infer_worker, camera_router_thread, composer_thread,
    results_collector_thread, camera_offline_monitor,
)
from obsero.api import app, S   # FastAPI app + shared state


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap
# ═══════════════════════════════════════════════════════════════════════════════

def seed_cameras_from_config(cfg: SystemConfig):
    """Ensure DB camera set matches config camera set."""
    keep_codes: set[str] = set()
    for cam in cfg.cameras:
        code = f"CFG-{cam.id}"
        keep_codes.add(code)
        camera_upsert(cam.name, cam.url, code=code, ptz_protocol="onvif",
                      cid=None, online=False)

    con = db_conn()
    try:
        if keep_codes:
            marks = ",".join("?" for _ in keep_codes)
            con.execute(
                f"DELETE FROM cameras WHERE code IS NULL OR code NOT IN ({marks})",
                tuple(sorted(keep_codes)),
            )
        else:
            con.execute("DELETE FROM cameras")
        con.commit()
    finally:
        con.close()

    # update IDs so that config ids match DB ids (upsert by code)
    # re-read from DB to get authoritative IDs
    return cameras_all()


def backend_bootstrap(cfg: SystemConfig, camera_out_q: mp.Queue):
    """
    Runs in a worker thread.  Spawns:
      1. Model workers (one per (model_key, gpu_id) pair)
      2. Camera processes (one per camera)
      3. Router + collector + composer threads
      4. Offline monitor thread
    """
    try:
        print("[boot] backend bootstrap starting …", flush=True)

        # shared state
        stop = S.stop_event
        display_q: queue.Queue = queue.Queue(maxsize=1)
        results_q: queue.Queue = queue.Queue(maxsize=800)

        # ── Gate manager ──
        gate_mgr = GateManager(cfg.rules)

        # ── Model keyword lookup from config ──
        model_keywords: dict[str, list[str]] = {}
        for m in cfg.models:
            model_keywords[m.key] = m.keywords

        # ── FanOut ──
        fanout = FanOut(cfg.cadence_map())

        # camera -> gpu mapping
        cam_gpu_map = {c.id: c.gpu_id for c in cfg.cameras}
        fanout.set_camera_gpu(cam_gpu_map)

        # ── Model workers ──
        mp_output = mp.Queue(maxsize=200)
        mp_inputs: dict[str, list[tuple[mp.Queue, mp.Process]]] = {}
        needed = cfg.needed_gpu_model_pairs()
        print(f"[boot] spawning {len(needed)} model worker(s) …", flush=True)

        for gpu_id, model_key in sorted(needed):
            mcfg = cfg.model_by_key(model_key)
            if mcfg is None:
                continue
            q_in = mp.Queue(maxsize=120)
            proc_name = f"{model_key}_gpu{gpu_id}"
            p = mp.Process(
                target=mp_infer_worker,
                args=(proc_name, mcfg.stem, mcfg.conf_thr, mcfg.keywords,
                      gpu_id, q_in, mp_output),
                daemon=True,
            )
            p.start()
            mp_inputs.setdefault(model_key, []).append((q_in, p))
            fanout.register(model_key, gpu_id, q_in)

        print(f"[boot] model workers up: {sum(len(v) for v in mp_inputs.values())}",
              flush=True)

        # ── Results collector ──
        threading.Thread(
            target=results_collector_thread,
            args=(mp_output,
                  lambda: S.cam_mgr.active_camera_id if S.cam_mgr else None,
                  S.raw_snaps, S.raw_snaps_lock,
                  results_q, gate_mgr, model_keywords,
                  cfg.per_tag_cooldown, cfg.default_cooldown_sec,
                  S.incidents_ring, stop),
            daemon=True,
        ).start()

        # ── Camera processes ──
        db_cams = cameras_all()
        # build map camera_db_id -> (url, max_fps, target_side)
        # Match DB cameras to config by code (CFG-{config_id})
        cam_cfg_by_code = {f"CFG-{c.id}": c for c in cfg.cameras}
        cam_map: dict[int, tuple] = {}
        active_id = None
        for dbc in db_cams:
            cc = cam_cfg_by_code.get(dbc.get("code"))
            if cc:
                cam_map[dbc["id"]] = (cc.url, cc.max_fps, cc.target_side)
                if active_id is None:
                    active_id = dbc["id"]
            else:
                # camera in DB but not in config — use DB url with defaults
                cam_map[dbc["id"]] = (dbc["url"], 12, 640)
                if active_id is None:
                    active_id = dbc["id"]

        cam_mgr = MultiCamProcManager(camera_out_q)
        cam_mgr.start_all(cam_map, active_id)
        S.cam_mgr = cam_mgr
        S.cfg = cfg

        # ── Last-seen timestamps for offline detection ──
        last_seen: dict[int, float] = {}

        # ── Router thread ──
        threading.Thread(
            target=camera_router_thread,
            args=(camera_out_q, fanout,
                  S.raw_snaps, S.raw_snaps_lock,
                  display_q,
                  lambda: cam_mgr.active_camera_id,
                  stop,
                  lambda cid: camera_set_online(cid, True),
                  last_seen),
            daemon=True,
        ).start()

        # ── Composer thread ──
        threading.Thread(
            target=composer_thread,
            args=(stop,
                  lambda: cam_mgr.active_camera_id,
                  display_q, results_q,
                  S.annotated_jpeg, S.annotated_jpeg_lock,
                  S.annotated_snaps, S.annotated_snaps_lock),
            daemon=True,
        ).start()

        # ── Offline monitor ──
        threading.Thread(
            target=camera_offline_monitor,
            args=(last_seen, cfg.camera_offline_timeout_sec, stop,
                  lambda cid: camera_set_online(cid, False)),
            daemon=True,
        ).start()

        # store references for shutdown
        S._mp_inputs = mp_inputs

        print("[boot] backend bootstrap complete", flush=True)
    except Exception as e:
        print(f"[boot] ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)


def install_signal_handlers():
    def handle_sig(sig, _frame):
        print(f"[main] signal {sig} → shutdown", flush=True)
        S.stop_event.set()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, handle_sig)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    mp.set_start_method("spawn", force=True)
    install_signal_handlers()

    parser = argparse.ArgumentParser(description="Obsero Enterprise Safety Panel")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to system.yaml (default: configs/system.yaml)")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.host:
        cfg.server.host = args.host
    if args.port:
        cfg.server.port = args.port

    # ── DB init ──
    db_init()

    # ── Seed cameras from config ──
    seed_cameras_from_config(cfg)

    # ── IPC queue first (so children inherit handles) ──
    camera_out_q = mp.Queue(maxsize=200)

    # ── Boot backend in worker thread ──
    threading.Thread(target=backend_bootstrap, args=(cfg, camera_out_q),
                     daemon=True).start()

    # ── Uvicorn in MAIN thread (Windows-stable) ──
    host, port = cfg.server.host, cfg.server.port
    print(f"[web] starting uvicorn on http://{host}:{port}", flush=True)
    if cfg.server.open_browser:
        try:
            webbrowser.open(f"http://127.0.0.1:{port}/")
        except Exception:
            pass
    uvicorn.run(app, host=host, port=port, log_level="info")

    # ── Teardown ──
    S.stop_event.set()
    if S.cam_mgr:
        S.cam_mgr.stop_all()
    mp_inputs = getattr(S, "_mp_inputs", {})
    for key, proc_list in mp_inputs.items():
        for q_in, p in proc_list:
            try:
                q_in.put_nowait(None)
            except Exception:
                pass
            if p.is_alive():
                p.join(timeout=2.0)
    print("[main] bye", flush=True)


if __name__ == "__main__":
    main()
