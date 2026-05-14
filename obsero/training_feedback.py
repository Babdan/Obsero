"""
Training-feedback capture for reviewed fall-detection incidents.

False-positive fall reviews are exported as negative examples with the
incident snapshot, recent video clip, keypoint sequence, and a manifest.
"""

from __future__ import annotations

import datetime
import json
import shutil
import threading
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INCIDENTS_DIR = ROOT / "incidents"
DEFAULT_FEEDBACK_DIR = ROOT / "data" / "fall_detection" / "training_feedback"


def _resolve_repo_path(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _safe_name(value: object) -> str:
    text = str(value or "").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def _relative_incident_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(INCIDENTS_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _source_path(path_value: str | None, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _copy_asset(src: Path | None, dst_dir: Path, assets: dict[str, str], key: str) -> None:
    if src is None or not src.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    assets[key] = str(dst)


def _parse_rule_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


class FallClipBuffer:
    """Thread-safe compressed-frame buffer for short fall incident clips."""

    def __init__(self, output_dir: Path, duration_sec: float = 8.0, fps: int = 10):
        self.output_dir = output_dir
        self.duration_sec = max(1.0, float(duration_sec))
        self.fps = max(1, int(fps))
        self._buffers: dict[int, deque[tuple[int, bytes]]] = {}
        self._last_ts: dict[int, int] = {}
        self._lock = threading.Lock()

    def record_frame(self, camera_id: int, jpeg_bytes: bytes, timestamp_ms: int) -> None:
        min_interval_ms = max(1, int(1000 / self.fps))
        cutoff = int(timestamp_ms - self.duration_sec * 1000)
        with self._lock:
            last_ts = self._last_ts.get(camera_id)
            if last_ts is not None and timestamp_ms - last_ts < min_interval_ms:
                return
            self._last_ts[camera_id] = timestamp_ms
            buf = self._buffers.setdefault(camera_id, deque())
            buf.append((timestamp_ms, bytes(jpeg_bytes)))
            while buf and buf[0][0] < cutoff:
                buf.popleft()

    def save_clip(self, camera_id: int, event_id: str, timestamp_ms: int) -> dict[str, Any] | None:
        cutoff = int(timestamp_ms - self.duration_sec * 1000)
        with self._lock:
            raw_frames = [
                item for item in self._buffers.get(camera_id, deque())
                if cutoff <= item[0] <= timestamp_ms
            ]

        frames: list[np.ndarray] = []
        for _ts, jpeg in raw_frames:
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)

        if not frames:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        basename = f"fall_{camera_id}_{_safe_name(event_id or timestamp_ms)}"
        clip_path = self.output_dir / f"{basename}.avi"
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            float(self.fps),
            (width, height),
        )

        if writer.isOpened():
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
            writer.release()
            if clip_path.exists() and clip_path.stat().st_size > 0:
                return {
                    "clip_path": _relative_incident_path(clip_path),
                    "format": "mjpg_avi",
                    "frame_count": len(frames),
                    "fps": self.fps,
                    "duration_sec": round(len(frames) / self.fps, 3),
                    "pre_roll_sec": self.duration_sec,
                }
        else:
            writer.release()

        frames_dir = self.output_dir / f"{basename}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames):
            cv2.imwrite(str(frames_dir / f"frame_{idx:04d}.jpg"), frame)
        return {
            "frames_dir": _relative_incident_path(frames_dir),
            "format": "jpeg_sequence",
            "frame_count": len(frames),
            "fps": self.fps,
            "duration_sec": round(len(frames) / self.fps, 3),
            "pre_roll_sec": self.duration_sec,
        }


def export_reviewed_fall_feedback(alert_id: int,
                                  status: str,
                                  reviewer: str,
                                  note: str = "",
                                  feedback_dir: str | Path | None = None) -> dict[str, Any] | None:
    from obsero.db import alert_get

    row = alert_get(alert_id)
    if not row:
        return None
    return export_fall_feedback_row(row, status, reviewer, note, feedback_dir)


def export_fall_feedback_row(row: dict[str, Any],
                             status: str,
                             reviewer: str,
                             note: str = "",
                             feedback_dir: str | Path | None = None,
                             incidents_dir: Path = INCIDENTS_DIR) -> dict[str, Any] | None:
    if status != "false_positive":
        return None

    event_type = str(row.get("type") or "").upper()
    model_key = str(row.get("model_key") or "")
    if event_type != "FALL" and model_key != "pose_fall_detection":
        return None

    alert_id = int(row["id"])
    root = _resolve_repo_path(feedback_dir) or DEFAULT_FEEDBACK_DIR
    case_dir = root / "negative" / f"alert_{alert_id}"
    assets_dir = case_dir / "assets"
    case_dir.mkdir(parents=True, exist_ok=True)

    rule = _parse_rule_json(row.get("rule_json"))
    event = rule.get("event") if isinstance(rule.get("event"), dict) else {}
    event_meta = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    obsero_meta = rule.get("obsero") if isinstance(rule.get("obsero"), dict) else {}
    clip_info = obsero_meta.get("clip") if isinstance(obsero_meta.get("clip"), dict) else {}
    sequence_info = (
        event_meta.get("sequence")
        if isinstance(event_meta.get("sequence"), dict)
        else {}
    )

    assets: dict[str, str] = {}
    for key in ("image", "full_image"):
        _copy_asset(_source_path(row.get(key), incidents_dir), assets_dir, assets, key)

    _copy_asset(_source_path(clip_info.get("clip_path"), incidents_dir),
                assets_dir, assets, "clip")
    _copy_asset(_source_path(clip_info.get("frames_dir"), incidents_dir),
                assets_dir, assets, "clip_frames")
    _copy_asset(_source_path(sequence_info.get("sequence_path"), ROOT),
                assets_dir, assets, "keypoint_sequence")
    _copy_asset(_source_path(sequence_info.get("metadata_path"), ROOT),
                assets_dir, assets, "keypoint_metadata")

    exported_at = datetime.datetime.now().isoformat(timespec="seconds")
    manifest = {
        "schema_version": 1,
        "label": "negative",
        "reason": "false_positive",
        "exported_at": exported_at,
        "review": {
            "status": status,
            "reviewer": reviewer,
            "note": note,
        },
        "alert": {
            "id": alert_id,
            "ts": row.get("ts"),
            "camera_id": row.get("camera_id"),
            "type": row.get("type"),
            "label": row.get("label"),
            "confidence": row.get("conf"),
            "bbox": row.get("bbox"),
            "model_key": row.get("model_key"),
        },
        "assets": assets,
        "source_event": event,
        "clip": clip_info,
    }

    manifest_path = case_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)

    root.mkdir(parents=True, exist_ok=True)
    with (root / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({**manifest, "case_dir": str(case_dir)}, sort_keys=True, default=str) + "\n")

    return {
        "exported": True,
        "label": "negative",
        "case_dir": str(case_dir),
        "manifest": str(manifest_path),
        "assets": assets,
    }
