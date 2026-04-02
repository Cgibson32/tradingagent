"""ICT Kill Zone timing module — ADVISORY ONLY.

Defines time-based windows where ICT setups have higher probability.
Affects confidence scoring but does NOT block entries or force exits.
Overnight holds are allowed.

All times are in Eastern Time (ET).
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# CME Globex: Sun 5:00 PM – Fri 4:00 PM ET
# Daily maintenance halt: 4:00–5:00 PM ET
DAILY_HALT_START = time(16, 0)
DAILY_HALT_END = time(17, 0)


class KillZone:
    """Represents a single kill zone time window."""

    def __init__(self, label: str, start: time, end: time, weight: float = 1.0):
        self.label = label
        self.start = start
        self.end = end
        self.weight = weight  # Confidence multiplier (1.0 = no change)

    def is_active(self, now_et: time) -> bool:
        """Check if the current ET time falls within this kill zone."""
        if self.start <= self.end:
            return self.start <= now_et <= self.end
        else:
            # Wraps past midnight (e.g., overnight 17:00-02:00)
            return now_et >= self.start or now_et <= self.end


# Define kill zones with confidence weights
KILL_ZONES = [
    KillZone("Overnight/Asia", time(17, 0), time(2, 0), weight=0.6),
    KillZone("London Open", time(2, 0), time(5, 0), weight=0.85),
    KillZone("NY Open", time(9, 30), time(11, 0), weight=1.0),  # Highest probability
    KillZone("NY Lunch", time(12, 0), time(13, 30), weight=0.7),
    KillZone("NY PM Session", time(13, 30), time(16, 0), weight=0.85),
]


class KillZoneManager:
    """Manages kill zone timing and confidence adjustments.

    ADVISORY ONLY — affects confidence scoring but does not block entries
    or force exits. Overnight holds are fully supported.
    """

    def __init__(self, penalty_outside_kz: float = 0.15):
        self._penalty = penalty_outside_kz

    def get_et_now(self) -> datetime:
        """Get current time in Eastern Time."""
        return datetime.now(ET)

    def get_active_zone(self, now: datetime | None = None) -> KillZone | None:
        """Get the currently active kill zone, or None if outside all zones."""
        if now is None:
            now = self.get_et_now()
        elif now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        elif now.tzinfo != ET:
            now = now.astimezone(ET)

        now_time = now.time()

        # During maintenance halt, no zone is active
        if DAILY_HALT_START <= now_time < DAILY_HALT_END:
            return None

        for kz in KILL_ZONES:
            if kz.is_active(now_time):
                return kz
        return None

    def get_active_zone_label(self, now: datetime | None = None) -> str | None:
        """Get the label of the currently active kill zone."""
        kz = self.get_active_zone(now)
        return kz.label if kz else None

    def is_maintenance_halt(self, now: datetime | None = None) -> bool:
        """Check if we're in the CME daily maintenance halt (4-5 PM ET)."""
        if now is None:
            now = self.get_et_now()
        elif now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        elif now.tzinfo != ET:
            now = now.astimezone(ET)

        return DAILY_HALT_START <= now.time() < DAILY_HALT_END

    def adjust_confidence(self, confidence: float, now: datetime | None = None) -> float:
        """Apply kill zone confidence adjustments (advisory only).

        In a kill zone: multiply by zone weight
        Outside all zones: subtract penalty
        During maintenance halt: return 0
        """
        if now is None:
            now = self.get_et_now()

        if self.is_maintenance_halt(now):
            return 0.0

        kz = self.get_active_zone(now)
        if kz:
            return confidence * kz.weight
        else:
            return max(confidence - self._penalty, 0.0)
