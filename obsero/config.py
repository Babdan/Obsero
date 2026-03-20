"""
obsero.config — Load and validate configs/system.yaml + configs/cameras.json

Camera sources are loaded dynamically:
  1. If  configs/cameras.json  exists (written by discover_cameras.py),
     cameras are read from there — overriding anything in system.yaml.
  2. Otherwise cameras fall back to the [cameras] section in system.yaml.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "system.yaml"
CAMERAS_JSON_PATH = ROOT / "configs" / "cameras.json"


# ------------ Dataclasses ---------------------------------------------------

@dataclass
class GpuCfg:
    id: int
    weight: float = 1.0


@dataclass
class CameraCfg:
    id: int
    name: str
    url: str
    gpu_id: int = 0
    max_fps: int = 12
    target_side: int = 640


@dataclass
class ModelCfg:
    key: str
    stem: str
    conf_thr: float = 0.5
    keywords: list[str] = field(default_factory=list)
    cadence_every_n: int = 1


@dataclass
class RuleCfg:
    min_consecutive_frames: int = 3
    window_sec: float = 5.0
    min_ratio: float = 0.5
    cooldown_sec: float = 10.0


@dataclass
class ServerCfg:
    host: str = "0.0.0.0"
    port: int = 9009
    open_browser: bool = True


@dataclass
class SystemConfig:
    server: ServerCfg = field(default_factory=ServerCfg)
    gpus: list[GpuCfg] = field(default_factory=lambda: [GpuCfg(id=0)])
    cameras: list[CameraCfg] = field(default_factory=list)
    models: list[ModelCfg] = field(default_factory=list)
    rules: dict[str, RuleCfg] = field(default_factory=dict)
    camera_offline_timeout_sec: float = 15.0
    default_cooldown_sec: float = 5.0
    per_tag_cooldown: dict[str, float] = field(default_factory=dict)

    # ------- helpers -------
    def gpu_ids(self) -> list[int]:
        return [g.id for g in self.gpus]

    def cameras_for_gpu(self, gpu_id: int) -> list[CameraCfg]:
        return [c for c in self.cameras if c.gpu_id == gpu_id]

    def camera_by_id(self, cid: int) -> CameraCfg | None:
        return next((c for c in self.cameras if c.id == cid), None)

    def model_by_key(self, key: str) -> ModelCfg | None:
        return next((m for m in self.models if m.key == key), None)

    def rule_for(self, event_type: str) -> RuleCfg:
        return self.rules.get(event_type, RuleCfg())

    def needed_gpu_model_pairs(self) -> set[tuple[int, str]]:
        """Return set of (gpu_id, model_key) actually needed."""
        gpu_ids_with_cameras = {c.gpu_id for c in self.cameras}
        pairs = set()
        for gid in gpu_ids_with_cameras:
            for m in self.models:
                pairs.add((gid, m.key))
        return pairs

    def cadence_map(self) -> dict[str, int]:
        return {m.key: m.cadence_every_n for m in self.models}


# ------------ Parser --------------------------------------------------------

def _parse_config(raw: dict[str, Any]) -> SystemConfig:
    srv_raw = raw.get("server", {})
    server = ServerCfg(
        host=srv_raw.get("host", "0.0.0.0"),
        port=int(srv_raw.get("port", 9009)),
        open_browser=bool(srv_raw.get("open_browser", True)),
    )

    gpus = [GpuCfg(id=int(g["id"]), weight=float(g.get("weight", 1.0)))
            for g in raw.get("gpus", [{"id": 0}])]

    cameras = []
    for c in raw.get("cameras", []):
        cameras.append(CameraCfg(
            id=int(c["id"]),
            name=str(c.get("name", f"CAM-{c['id']}")),
            url=str(c["url"]),
            gpu_id=int(c.get("gpu_id", 0)),
            max_fps=int(c.get("max_fps", 12)),
            target_side=int(c.get("target_side", 640)),
        ))

    models = []
    for m in raw.get("models", []):
        models.append(ModelCfg(
            key=str(m["key"]),
            stem=str(m["stem"]),
            conf_thr=float(m.get("conf_thr", 0.5)),
            keywords=list(m.get("keywords", [])),
            cadence_every_n=int(m.get("cadence_every_n", 1)),
        ))

    rules: dict[str, RuleCfg] = {}
    for evt, rd in raw.get("rules", {}).items():
        rules[str(evt)] = RuleCfg(
            min_consecutive_frames=int(rd.get("min_consecutive_frames", 3)),
            window_sec=float(rd.get("window_sec", 5.0)),
            min_ratio=float(rd.get("min_ratio", 0.5)),
            cooldown_sec=float(rd.get("cooldown_sec", 10.0)),
        )

    per_tag = {}
    for k, v in raw.get("per_tag_cooldown", {}).items():
        per_tag[str(k)] = float(v)

    return SystemConfig(
        server=server,
        gpus=gpus,
        cameras=cameras,
        models=models,
        rules=rules,
        camera_offline_timeout_sec=float(raw.get("camera_offline_timeout_sec", 15.0)),
        default_cooldown_sec=float(raw.get("default_cooldown_sec", 5.0)),
        per_tag_cooldown=per_tag,
    )


def load_config(path: Path | str | None = None) -> SystemConfig:
    """Load system.yaml. Falls back to defaults if file missing."""
    path = Path(path) if path else CONFIG_PATH
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        print(f"[config] loaded {path}")
    else:
        print(f"[config] {path} not found – using defaults")
        raw = {}

    cfg = _parse_config(raw)

    # ── Overlay cameras from cameras.json (if present) ──
    cameras_json = _load_cameras_from_json()
    if cameras_json is not None:
        cfg.cameras = cameras_json
        print(f"[config] cameras loaded from {CAMERAS_JSON_PATH}  "
              f"({len(cameras_json)} camera(s))")
    elif not cfg.cameras:
        print("[config] WARNING: no cameras configured "
              "(run discover_cameras.py or add them to system.yaml)")

    return cfg


# ── JSON camera config loader ──────────────────────────────────────────────

def _load_cameras_from_json(path: Path | None = None) -> list[CameraCfg] | None:
    """
    Read  configs/cameras.json  (written by discover_cameras.py).
    Returns a list of CameraCfg or None if the file is absent / corrupt.
    """
    p = path or CAMERAS_JSON_PATH
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[config] warning: could not read {p}: {exc}")
        return None

    raw_cams = data.get("cameras", [])
    if not raw_cams:
        return None

    cameras: list[CameraCfg] = []
    for c in raw_cams:
        cameras.append(CameraCfg(
            id=int(c["id"]),
            name=str(c.get("name", f"CAM-{c['id']}")),
            url=str(c["url"]) if not isinstance(c["url"], int) else str(c["url"]),
            gpu_id=int(c.get("gpu_id", 0)),
            max_fps=int(c.get("max_fps", 12)),
            target_side=int(c.get("target_side", 640)),
        ))
    return cameras


def load_cameras_json_raw(path: Path | str | None = None) -> dict | None:
    """
    Public helper: load the raw cameras.json dict.
    Useful for scripts that need NVR metadata (ip, port, etc.).
    """
    p = Path(path) if path else CAMERAS_JSON_PATH
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
