"""
Alarm Interface — Structured alarm event output for fall detections.

Produces alarm events compatible with standard AI-video alarm structures:
event type, severity, bounding box, timestamp, confidence, and confirmation.
Currently outputs to JSON log files and in-memory list.
"""

import uuid
import json
import cv2
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.pose_extractor import NUM_KEYPOINTS


@dataclass
class FallAlarmEvent:
    """A structured fall detection alarm event."""

    event_id: str = ""
    event_type: str = "fall_detection"
    severity: str = "critical"           # "critical" | "warning"
    timestamp: str = ""
    timestamp_ms: int = 0
    camera_id: str = "cam_0"
    confidence: float = 0.0
    bounding_box: dict = field(default_factory=dict)
    snapshot_path: Optional[str] = None
    keypoint_summary: dict = field(default_factory=dict)
    posture_checks: dict = field(default_factory=dict)
    confirmed: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class AlarmInterface:
    """
    Manages fall alarm event creation, storage, and output.

    Args:
        config: Dictionary with alarm configuration.
    """

    def __init__(self, config: dict):
        self._event_type = config.get("event_type", "fall_detection")
        self._severity_levels = config.get("severity_levels", {
            "confirmed": "critical",
            "suspected": "warning",
        })
        self._save_snapshot = config.get("save_snapshot", True)
        self._snapshot_dir = Path(config.get("snapshot_dir", "logs/snapshots"))
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        # In-memory event store
        self._events: list[FallAlarmEvent] = []
        self._max_events = 100

    def create_event(
        self,
        camera_id: str,
        confidence: float,
        confirmed: bool,
        keypoints: Optional[np.ndarray] = None,
        posture_checks: Optional[dict] = None,
        frame: Optional[np.ndarray] = None,
        model_config: Optional[dict] = None,
    ) -> FallAlarmEvent:
        """
        Create a new fall alarm event.

        Args:
            camera_id: Camera identifier.
            confidence: Model fall probability [0, 1].
            confirmed: Whether this is a fully confirmed fall.
            keypoints: Current keypoints (17, 2) for bounding box.
            posture_checks: PostureCheckResult dict.
            frame: Current camera frame for snapshot.
            model_config: Model configuration metadata.

        Returns:
            Constructed FallAlarmEvent.
        """
        now = datetime.now()
        severity_key = "confirmed" if confirmed else "suspected"

        # Compute bounding box from keypoints
        bbox = {}
        if keypoints is not None and keypoints.shape[0] == NUM_KEYPOINTS:
            valid = keypoints[keypoints.sum(axis=1) > 0]
            if len(valid) > 0:
                bbox = {
                    "x1": float(valid[:, 0].min()),
                    "y1": float(valid[:, 1].min()),
                    "x2": float(valid[:, 0].max()),
                    "y2": float(valid[:, 1].max()),
                }

        # Save snapshot
        snapshot_path = None
        if self._save_snapshot and frame is not None:
            snap_name = f"fall_{camera_id}_{now.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            snapshot_path = str(self._snapshot_dir / snap_name)
            cv2.imwrite(snapshot_path, frame)

        event = FallAlarmEvent(
            event_id=str(uuid.uuid4()),
            event_type=self._event_type,
            severity=self._severity_levels.get(severity_key, "warning"),
            timestamp=now.isoformat(),
            timestamp_ms=int(now.timestamp() * 1000),
            camera_id=camera_id,
            confidence=round(confidence, 4),
            bounding_box=bbox,
            snapshot_path=snapshot_path,
            keypoint_summary={
                "num_keypoints": NUM_KEYPOINTS,
                "format": "coco_17",
            },
            posture_checks=posture_checks or {},
            confirmed=confirmed,
            metadata=model_config or {},
        )

        return event

    def emit(self, event: FallAlarmEvent):
        """
        Emit an alarm event (store in memory + log to console).

        Args:
            event: The fall alarm event to emit.
        """
        self._events.append(event)

        # Trim old events
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

        # Console output
        severity_icon = "🚨" if event.confirmed else "⚠️"
        print(
            f"\n{severity_icon} FALL ALARM [{event.severity.upper()}] "
            f"| Camera: {event.camera_id} "
            f"| Confidence: {event.confidence:.1%} "
            f"| Time: {event.timestamp} "
            f"| ID: {event.event_id[:8]}"
        )

    def get_recent_events(self, n: int = 10) -> list[FallAlarmEvent]:
        """Return the N most recent alarm events."""
        return self._events[-n:]

    def get_event_count(self) -> int:
        return len(self._events)
