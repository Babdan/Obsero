"""
obsero.rules — Temporal gating for enterprise compliance.

TemporalGate   – per (camera_id, event_type) gate.
GateManager    – lookup / create gates keyed by (camera_id, event_type).

A gate fires (returns True) when BOTH:
  1. ≥ min_consecutive_frames streak of positive frames in a row.
  2. Within the last window_sec, (positive / total) ≥ min_ratio.

After firing, a cooldown_sec period suppresses re-triggers.
"""

from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field

from obsero.config import RuleCfg


@dataclass
class _Sample:
    ts: float
    positive: bool


class TemporalGate:
    """Sliding-window + streak gate for one (camera, event_type) pair."""

    def __init__(self, rule: RuleCfg):
        self.min_streak: int = max(1, rule.min_consecutive_frames)
        self.window_sec: float = rule.window_sec
        self.min_ratio: float = rule.min_ratio
        self.cooldown_sec: float = rule.cooldown_sec

        self._streak: int = 0
        self._window: deque[_Sample] = deque()
        self._last_fire: float = 0.0

    # ---- public ----

    def feed(self, positive: bool, ts: float | None = None) -> bool:
        """Feed one frame result.  Returns True on a rising-edge trigger."""
        now = ts or time.time()

        # streak
        if positive:
            self._streak += 1
        else:
            self._streak = 0

        # sliding window
        self._window.append(_Sample(ts=now, positive=positive))
        cutoff = now - self.window_sec
        while self._window and self._window[0].ts < cutoff:
            self._window.popleft()

        streak_ok = self._streak >= self.min_streak
        ratio_ok = self._ratio() >= self.min_ratio
        cooldown_ok = (now - self._last_fire) >= self.cooldown_sec

        if streak_ok and ratio_ok and cooldown_ok:
            self._last_fire = now
            return True
        return False

    def snapshot(self) -> dict:
        """Serialisable stats for storage in rule_json."""
        total = len(self._window)
        pos = sum(1 for s in self._window if s.positive)
        return {
            "streak": self._streak,
            "window_total": total,
            "window_pos": pos,
            "ratio": round(pos / total, 4) if total else 0.0,
            "min_streak": self.min_streak,
            "window_sec": self.window_sec,
            "min_ratio": self.min_ratio,
            "cooldown_sec": self.cooldown_sec,
        }

    # ---- private ----

    def _ratio(self) -> float:
        total = len(self._window)
        if total == 0:
            return 0.0
        pos = sum(1 for s in self._window if s.positive)
        return pos / total


class GateManager:
    """Thread-safe manager for TemporalGate objects."""

    def __init__(self, rules: dict[str, RuleCfg]):
        self._rules = rules
        self._gates: dict[tuple[int, str], TemporalGate] = {}

    def get(self, camera_id: int, event_type: str) -> TemporalGate:
        key = (camera_id, event_type)
        if key not in self._gates:
            rule = self._rules.get(event_type, RuleCfg())
            self._gates[key] = TemporalGate(rule)
        return self._gates[key]

    def feed(self, camera_id: int, event_type: str, positive: bool,
             ts: float | None = None) -> bool:
        """Convenience: feed + return trigger bool."""
        return self.get(camera_id, event_type).feed(positive, ts)

    def snapshot(self, camera_id: int, event_type: str) -> dict:
        return self.get(camera_id, event_type).snapshot()
