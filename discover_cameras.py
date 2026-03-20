#!/usr/bin/env python3
"""
discover_cameras.py — Auto-detect NVR cameras and fall back to local webcams.

This script:
  1. Scans the local network for NVR devices (Dahua RTSP protocol).
  2. For every NVR found, probes each channel to see if it's online.
  3. If no NVR is found, falls back to enumerating local USB/built-in webcams.
  4. Writes the discovered configuration to  configs/cameras.json .

Run manually before starting Obsero, or let run.py call it on boot.

Usage:
    python discover_cameras.py                       # auto-detect everything
    python discover_cameras.py --nvr-ip 192.168.1.108 --nvr-user admin --nvr-pass "Petra123!"
    python discover_cameras.py --max-channels 16     # probe up to 16 channels
    python discover_cameras.py --force-webcam         # skip NVR, only enumerate webcams
    python discover_cameras.py --subnet 192.168.1    # scan 192.168.1.1-254 for NVRs
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000",
)
import cv2

# ───────────────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = ROOT / "configs"
CAMERAS_JSON = CONFIGS_DIR / "cameras.json"

# Default NVR credentials (overridable via CLI)
DEFAULT_NVR_USER = "admin"
DEFAULT_NVR_PASS = "admin"
DEFAULT_MAX_CHANNELS = 16
DEFAULT_RTSP_PORT = 554
DEFAULT_SUBNETS = ["192.168.1"]

# How many webcam indices to try when no NVR is found
MAX_WEBCAM_INDEX = 10

# Camera defaults written to JSON
DEFAULT_GPU_ID = 0
DEFAULT_MAX_FPS = 12
DEFAULT_TARGET_SIDE = 640


# ───────────────────────────────────────────────────────────────────────────────
# RTSP helpers
# ───────────────────────────────────────────────────────────────────────────────

def dahua_rtsp_url(ip: str, channel: int, subtype: int = 0,
                   user: str = "admin", password: str = "admin",
                   port: int = 554) -> str:
    """Build a Dahua-compatible RTSP URL."""
    return (
        f"rtsp://{user}:{password}@{ip}:{port}"
        f"/cam/realmonitor?channel={channel}&subtype={subtype}&unicast=true"
    )


def quick_probe_rtsp(url: str, timeout_sec: float = 4.0) -> bool:
    """Try to open an RTSP stream and read one frame within *timeout_sec*."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        return False
    t0 = time.time()
    ok = False
    while time.time() - t0 < timeout_sec:
        ret, frame = cap.read()
        if ret and frame is not None:
            ok = True
            break
        time.sleep(0.05)
    cap.release()
    return ok


def port_open(ip: str, port: int = 554, timeout: float = 1.0) -> bool:
    """Quick TCP connect check to see if a host has an open RTSP port."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


# ───────────────────────────────────────────────────────────────────────────────
# Discovery: NVR
# ───────────────────────────────────────────────────────────────────────────────

def scan_subnet_for_rtsp(subnet: str, port: int = 554,
                         timeout: float = 0.8) -> list[str]:
    """
    Scan *subnet*.1 – *subnet*.254  for hosts with TCP *port* open.
    Returns list of IP addresses that responded.
    """
    print(f"[discover] scanning {subnet}.1-254 for RTSP port {port} …")
    ips_found: list[str] = []

    def _check(ip: str) -> str | None:
        return ip if port_open(ip, port, timeout) else None

    with ThreadPoolExecutor(max_workers=64) as pool:
        futs = {pool.submit(_check, f"{subnet}.{i}"): i
                for i in range(1, 255)}
        for f in as_completed(futs):
            result = f.result()
            if result:
                ips_found.append(result)

    ips_found.sort(key=lambda ip: int(ip.rsplit(".", 1)[-1]))
    if ips_found:
        print(f"[discover]   found RTSP hosts: {', '.join(ips_found)}")
    else:
        print(f"[discover]   no RTSP hosts on {subnet}.*")
    return ips_found


def probe_nvr_channels(ip: str, user: str, password: str,
                       max_channels: int, port: int = 554,
                       subtypes: list[int] | None = None) -> list[dict]:
    """
    Probe *max_channels* on the NVR at *ip*.
    Returns a list of camera dicts for every channel that responds.
    """
    subtypes = subtypes or [0]
    cameras: list[dict] = []
    print(f"[discover] probing NVR {ip} channels 1..{max_channels} …")

    for ch in range(1, max_channels + 1):
        for st in subtypes:
            url = dahua_rtsp_url(ip, ch, st, user, password, port)
            name = f"CH{ch}" + (" Sub" if st == 1 else "")
            online = quick_probe_rtsp(url, timeout_sec=4.0)
            status = "online" if online else "offline"
            print(f"[discover]   {name:12s}  {status}")
            cameras.append({
                "name": name,
                "url": url,
                "online": online,
                "source_type": "nvr",
                "nvr_ip": ip,
                "channel": ch,
                "subtype": st,
                "gpu_id": DEFAULT_GPU_ID,
                "max_fps": DEFAULT_MAX_FPS,
                "target_side": DEFAULT_TARGET_SIDE,
            })
    return cameras


# ───────────────────────────────────────────────────────────────────────────────
# Discovery: Webcams (fallback / debug)
# ───────────────────────────────────────────────────────────────────────────────

def discover_webcams(max_index: int = MAX_WEBCAM_INDEX) -> list[dict]:
    """
    Enumerate local webcam indices 0..*max_index*.
    Returns a list of camera dicts for every index that opens successfully.
    """
    print(f"[discover] probing local webcams 0..{max_index - 1} …")
    cameras: list[dict] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                name = f"Webcam-{idx}"
                print(f"[discover]   {name:12s}  online")
                cameras.append({
                    "name": name,
                    "url": idx,  # integer index for OpenCV
                    "online": True,
                    "source_type": "webcam",
                    "gpu_id": DEFAULT_GPU_ID,
                    "max_fps": DEFAULT_MAX_FPS,
                    "target_side": DEFAULT_TARGET_SIDE,
                })
        else:
            cap.release()
    if not cameras:
        print("[discover]   no webcams found")
    return cameras


# ───────────────────────────────────────────────────────────────────────────────
# Build final JSON config
# ───────────────────────────────────────────────────────────────────────────────

def build_cameras_json(cameras: list[dict],
                       nvr_settings: dict | None = None) -> dict:
    """
    Assemble the JSON structure that Obsero loads at runtime.

    Schema:
      {
        "discovered_at": "2026-02-17T...",
        "nvr": { "ip": ..., "user": ..., "port": ..., "max_channels": ... } | null,
        "cameras": [
          { "id": 1, "name": "CH1", "url": "rtsp://...", "gpu_id": 0,
            "max_fps": 12, "target_side": 640, "source_type": "nvr", "online": true },
          ...
        ]
      }
    """
    cam_list = []
    for i, cam in enumerate(cameras, start=1):
        cam_list.append({
            "id": i,
            "name": cam["name"],
            "url": cam["url"],
            "gpu_id": cam.get("gpu_id", DEFAULT_GPU_ID),
            "max_fps": cam.get("max_fps", DEFAULT_MAX_FPS),
            "target_side": cam.get("target_side", DEFAULT_TARGET_SIDE),
            "source_type": cam.get("source_type", "unknown"),
            "online": cam.get("online", False),
        })

    return {
        "discovered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "nvr": nvr_settings,
        "cameras": cam_list,
    }


def save_cameras_json(data: dict, path: Path = CAMERAS_JSON) -> Path:
    """Write cameras.json (pretty-printed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[discover] saved {len(data['cameras'])} camera(s) → {path}")
    return path


def load_cameras_json(path: Path = CAMERAS_JSON) -> dict | None:
    """Load cameras.json. Returns None if missing or corrupt."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[discover] warning: failed to read {path}: {exc}")
        return None


# ───────────────────────────────────────────────────────────────────────────────
# Main discovery orchestrator
# ───────────────────────────────────────────────────────────────────────────────

def discover(
    nvr_ip: str | None = None,
    nvr_user: str = DEFAULT_NVR_USER,
    nvr_pass: str = DEFAULT_NVR_PASS,
    nvr_port: int = DEFAULT_RTSP_PORT,
    max_channels: int = DEFAULT_MAX_CHANNELS,
    subnets: list[str] | None = None,
    force_webcam: bool = False,
    include_substreams: bool = False,
    output_path: Path | None = None,
) -> dict:
    """
    Run the full discovery pipeline:
      1. If *nvr_ip* given → probe that IP directly.
      2. Else scan *subnets* for RTSP hosts, probe each as NVR.
      3. If nothing found (or *force_webcam*) → enumerate webcams.
      4. Return the cameras.json dict.
    """
    all_cameras: list[dict] = []
    nvr_settings: dict | None = None
    subtypes = [0, 1] if include_substreams else [0]

    if not force_webcam:
        nvr_ips: list[str] = []

        if nvr_ip:
            # Explicit IP supplied — use it directly
            nvr_ips = [nvr_ip]
        else:
            # Auto-detect: scan subnets for any host with RTSP port open
            for subnet in (subnets or DEFAULT_SUBNETS):
                nvr_ips.extend(scan_subnet_for_rtsp(subnet, nvr_port))

        for ip in nvr_ips:
            cams = probe_nvr_channels(ip, nvr_user, nvr_pass,
                                      max_channels, nvr_port, subtypes)
            online_cams = [c for c in cams if c["online"]]
            if online_cams:
                all_cameras.extend(online_cams)
                nvr_settings = {
                    "ip": ip,
                    "user": nvr_user,
                    "port": nvr_port,
                    "max_channels": max_channels,
                }
                print(f"[discover] NVR {ip}: {len(online_cams)} channel(s) online")
            else:
                # Include all channels even if offline so config is complete
                all_cameras.extend(cams)
                nvr_settings = {
                    "ip": ip,
                    "user": nvr_user,
                    "port": nvr_port,
                    "max_channels": max_channels,
                }
                print(f"[discover] NVR {ip}: 0 channels online (all added as offline)")

    # Fallback: local webcams
    if not all_cameras or force_webcam:
        if not force_webcam:
            print("[discover] no NVR cameras found — falling back to local webcams")
        webcams = discover_webcams()
        all_cameras.extend(webcams)

    if not all_cameras:
        print("[discover] WARNING: no cameras found at all!")

    data = build_cameras_json(all_cameras, nvr_settings)
    save_cameras_json(data, output_path or CAMERAS_JSON)
    return data


# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Discover NVR cameras or local webcams and write configs/cameras.json",
    )
    parser.add_argument("--nvr-ip", default=None,
                        help="NVR IP address (skip subnet scan)")
    parser.add_argument("--nvr-user", default=DEFAULT_NVR_USER,
                        help=f"NVR username (default: {DEFAULT_NVR_USER})")
    parser.add_argument("--nvr-pass", default=DEFAULT_NVR_PASS,
                        help=f"NVR password (default: {DEFAULT_NVR_PASS})")
    parser.add_argument("--nvr-port", type=int, default=DEFAULT_RTSP_PORT,
                        help=f"RTSP port (default: {DEFAULT_RTSP_PORT})")
    parser.add_argument("--max-channels", type=int, default=DEFAULT_MAX_CHANNELS,
                        help=f"Max NVR channels to probe (default: {DEFAULT_MAX_CHANNELS})")
    parser.add_argument("--subnet", action="append", dest="subnets",
                        help="Subnet prefix to scan (e.g. 192.168.1). Repeatable.")
    parser.add_argument("--force-webcam", action="store_true",
                        help="Skip NVR detection, only enumerate webcams")
    parser.add_argument("--include-substreams", action="store_true",
                        help="Also add subtype=1 (sub-streams) per channel")
    parser.add_argument("--output", default=str(CAMERAS_JSON),
                        help=f"Output JSON path (default: {CAMERAS_JSON})")

    args = parser.parse_args()

    output_path = Path(args.output)

    data = discover(
        nvr_ip=args.nvr_ip,
        nvr_user=args.nvr_user,
        nvr_pass=args.nvr_pass,
        nvr_port=args.nvr_port,
        max_channels=args.max_channels,
        subnets=args.subnets,
        force_webcam=args.force_webcam,
        include_substreams=args.include_substreams,
        output_path=output_path,
    )

    # Summary
    online = sum(1 for c in data["cameras"] if c.get("online"))
    total = len(data["cameras"])
    src = data["nvr"]["ip"] if data.get("nvr") else "webcam"
    print(f"\n[discover] Done — {online}/{total} camera(s) online  (source: {src})")
    print(f"[discover] Config saved to: {output_path}")


if __name__ == "__main__":
    main()
