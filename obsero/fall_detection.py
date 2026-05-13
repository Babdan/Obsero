"""
Adapter for the bundled pose-based fall detector.

The detector in ``fall-detection/`` is an async MediaPipe + TCN pipeline. This
adapter starts it beside Obsero's existing camera workers and bridges its
``FallAlarmEvent`` output into Obsero alerts.
"""

from __future__ import annotations

import asyncio
import datetime
import importlib.util
import inspect
import json
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

import yaml

from obsero.config import FallDetectionCfg

ROOT = Path(__file__).resolve().parent.parent
INCIDENTS_DIR = ROOT / "incidents"

AlertWriter = Callable[..., None]
AuditWriter = Callable[[str, str, str], None]


def _resolve_repo_path(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _project_root_for_config(config_path: Path) -> Path:
    config_dir = config_path.parent
    return config_dir.parent if config_dir.name == "config" else config_dir


def _absolute_from_project(project_root: Path, path_value: str | None) -> str | None:
    if not path_value:
        return path_value
    p = Path(path_value).expanduser()
    return str(p if p.is_absolute() else (project_root / p).resolve())


def _absolutize_config_path(raw: dict[str, Any],
                            project_root: Path,
                            section: str,
                            key: str) -> None:
    value = raw.get(section, {}).get(key)
    if value:
        raw.setdefault(section, {})[key] = _absolute_from_project(project_root, str(value))


def _update_external_config(config_path: Path | None,
                            snapshot_dir: Path,
                            log_dir: Path) -> Path | None:
    """Generate an Obsero runtime config without editing the detector's file."""
    if config_path is None or not config_path.exists():
        return config_path

    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[fall-detection] could not read config {config_path}: {exc}", flush=True)
        return config_path

    if not isinstance(raw, dict):
        return config_path

    project_root = _project_root_for_config(config_path)
    _absolutize_config_path(raw, project_root, "pose", "model_path")
    _absolutize_config_path(raw, project_root, "model", "weights_path")
    _absolutize_config_path(raw, project_root, "training", "data_dir")
    _absolutize_config_path(raw, project_root, "training", "raw_video_dir")

    raw.setdefault("debug", {})["log_dir"] = str(log_dir)
    raw.setdefault("alarm", {})["snapshot_dir"] = str(snapshot_dir)

    generated = ROOT / "data" / "fall_detection" / "fall_detection_config.obsero.yaml"
    generated.parent.mkdir(parents=True, exist_ok=True)
    try:
        with generated.open("w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=False)
        print(f"[fall-detection] generated Obsero config: {generated}", flush=True)
        return generated
    except Exception as exc:
        print(f"[fall-detection] could not write generated config: {exc}", flush=True)
        return config_path


def _load_fall_detector_module(project_dir: Path) -> tuple[type | None, Any | None]:
    detector_file = project_dir / "src" / "fall_detector.py"
    if not detector_file.exists():
        print(f"[fall-detection] detector file not found: {detector_file}", flush=True)
        return None, None

    project_path = str(project_dir)
    added_path = False
    if project_path not in sys.path:
        sys.path.insert(0, project_path)
        added_path = True

    try:
        module_name = f"_obsero_fall_detector_{abs(hash(str(detector_file)))}"
        spec = importlib.util.spec_from_file_location(module_name, detector_file)
        if spec is None or spec.loader is None:
            return None, None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        cls = getattr(module, "FallDetector", None)
        if inspect.isclass(cls):
            print(f"[fall-detection] loaded FallDetector from {detector_file}", flush=True)
            return cls, module
        print(f"[fall-detection] FallDetector class not found in {detector_file}", flush=True)
    except Exception as exc:
        print(f"[fall-detection] failed to load detector: {exc}", flush=True)
        traceback.print_exc(file=sys.stdout)
    finally:
        if added_path:
            try:
                sys.path.remove(project_path)
            except ValueError:
                pass
    return None, None


def _relative_incident_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    try:
        return str(path.resolve().relative_to(INCIDENTS_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _event_bbox(event: Any) -> list[float]:
    box = getattr(event, "bounding_box", None) or {}
    if isinstance(box, dict):
        values = [box.get(k, 0.0) for k in ("x1", "y1", "x2", "y2")]
    elif isinstance(box, (list, tuple)) and len(box) >= 4:
        values = list(box[:4])
    else:
        values = [0.0, 0.0, 0.0, 0.0]
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return [0.0, 0.0, 0.0, 0.0]


def _event_camera_id(event: Any) -> int | None:
    raw = getattr(event, "camera_id", None)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _level_from_severity(severity: str | None) -> str:
    sev = (severity or "").lower()
    if sev in {"critical", "high"}:
        return "high"
    if sev in {"warning", "medium"}:
        return "medium"
    return "low"


def _event_to_rule_json(event: Any) -> str:
    data = event.to_dict() if hasattr(event, "to_dict") else {}
    return json.dumps({
        "source": "pose_fall_detection",
        "event": data,
    }, sort_keys=True, default=str)


def _install_alarm_bridge(detector_module: Any,
                          alert_writer: AlertWriter,
                          audit_writer: AuditWriter) -> None:
    original_cls = getattr(detector_module, "AlarmInterface", None)
    if original_cls is None or getattr(original_cls, "_obsero_bridge", False):
        return

    class ObseroAlarmInterface(original_cls):  # type: ignore[misc, valid-type]
        _obsero_bridge = True

        def emit(self, event: Any):
            super().emit(event)
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            camera_id = _event_camera_id(event)
            confidence = float(getattr(event, "confidence", 0.0) or 0.0)
            confirmed = bool(getattr(event, "confirmed", False))
            severity = str(getattr(event, "severity", "") or "")
            snapshot = _relative_incident_path(getattr(event, "snapshot_path", None))
            label = "FALL:confirmed" if confirmed else "FALL:suspected"

            alert_writer(
                ts,
                camera_id,
                _level_from_severity(severity),
                "FALL",
                label,
                confidence,
                _event_bbox(event),
                snapshot or "",
                site="Site-A",
                status="new",
                model_key="pose_fall_detection",
                gpu_id=None,
                rule_json=_event_to_rule_json(event),
                full_image=snapshot,
            )
            audit_writer(
                "fall-detection",
                "pose_fall_alert",
                f"{camera_id}:{confidence:.4f}:{label}",
            )

    detector_module.AlarmInterface = ObseroAlarmInterface


class _FallDetectorRuntime:
    def __init__(self, detector: Any, camera_id: int, source: object):
        self.detector = detector
        self.camera_id = camera_id
        self.source = source
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run_loop,
            name=f"fall-detector-{self.camera_id}",
            daemon=True,
        )
        self.thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        if self.loop and self.loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(self.detector.stop(), self.loop)
                fut.result(timeout=timeout)
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def push_frame(self, frame: Any, timestamp_ms: int) -> None:
        if self.detector and hasattr(self.detector, 'push_frame'):
            self.detector.push_frame(frame, timestamp_ms)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self.loop = loop
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self.detector.start(
                    camera_id=str(self.camera_id),
                    source=self.source,
                    show_preview=False,
                )
            )
        except Exception as exc:
            print(
                f"[fall-detection] runtime error for camera {self.camera_id}: {exc}",
                flush=True,
            )
            traceback.print_exc(file=sys.stdout)
        finally:
            loop.close()


class ExternalFallDetectionManager:
    def __init__(self,
                 cfg: FallDetectionCfg,
                 camera_sources: dict[int, object],
                 alert_writer: AlertWriter,
                 audit_writer: AuditWriter):
        self.cfg = cfg
        self.camera_sources = camera_sources
        self.alert_writer = alert_writer
        self.audit_writer = audit_writer
        self.instances: list[_FallDetectorRuntime] = []

    def start(self) -> None:
        if not self.cfg.enabled:
            return

        project_dir = _resolve_repo_path(self.cfg.project_dir)
        config_path = _resolve_repo_path(self.cfg.config_path)
        snapshot_dir = _resolve_repo_path(self.cfg.snapshot_dir) or (INCIDENTS_DIR / "fall_detection")
        log_dir = _resolve_repo_path(self.cfg.log_dir) or (ROOT / "data" / "fall_detection" / "logs")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        if project_dir is None or not project_dir.exists():
            print(f"[fall-detection] project directory not found: {project_dir}", flush=True)
            return

        detector_cls, detector_module = _load_fall_detector_module(project_dir)
        if detector_cls is None or detector_module is None:
            return

        _install_alarm_bridge(detector_module, self.alert_writer, self.audit_writer)
        runtime_config = _update_external_config(config_path, snapshot_dir, log_dir)
        if runtime_config is None:
            print("[fall-detection] no config path available", flush=True)
            return

        selected_sources = self._selected_sources()
        if not selected_sources:
            print("[fall-detection] no matching camera sources configured", flush=True)
            return

        for camera_id, _source in selected_sources.items():
            try:
                detector = detector_cls(str(runtime_config))
            except Exception as exc:
                print(f"[fall-detection] detector init failed for camera {camera_id}: {exc}", flush=True)
                traceback.print_exc(file=sys.stdout)
                continue

            runtime = _FallDetectorRuntime(detector, camera_id, "shared_queue")
            runtime.start()
            self.instances.append(runtime)

        print(f"[fall-detection] started {len(self.instances)} detector instance(s)", flush=True)

    def stop(self) -> None:
        for instance in self.instances:
            instance.stop()
        self.instances.clear()

    def push_frame(self, camera_id: int, frame: Any, timestamp_ms: int) -> None:
        for instance in self.instances:
            if instance.camera_id == camera_id:
                instance.push_frame(frame, timestamp_ms)

    def _selected_sources(self) -> dict[int, object]:
        if not self.cfg.camera_ids:
            return dict(self.camera_sources)
        wanted = set(self.cfg.camera_ids)
        return {cid: src for cid, src in self.camera_sources.items() if cid in wanted}
