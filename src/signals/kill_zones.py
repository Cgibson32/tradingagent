"""ICT Kill Zone timing module.

Defines time-based entry windows where ICT setups have the highest
probability. Applies confidence penalties outside kill zones and
enforces the no-trade zone after 3:00 PM ET.

All times are in Eastern Time (ET).
"""

from __future__ import annotations

from datetime import datetime, time, timezone, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


class KillZone:
    """Represents a single kill zone time window."""

    def __init__(self, label: str, start: time, end: time, weight: float = 1.0):
        self.label = label
        self.start = start
        self.end = end
        self.weight = weight  # Confidence multiplier (1.0 = no change)

    def is_active(self, now_et: time) -> bool:
        """Check if the current ET time falls within this kill zone."""
        return self.start <= now_et <= self.end


# Define kill zones
KILL_ZONES = [
    KillZone("London Open", time(2, 0), time(5, 0), weight=0.85),
    KillZone("NY Open", time(9, 30), time(11, 0), weight=1.0),
    KillZone("NY Lunch", time(12, 0), time(13, 30), weight=0.7),
    KillZone("NY PM Session", time(13, 30), time(15, 0), weight=0.85),
]

# No-trade zone: only manage existing positions
NO_TRADE_START = time(15, 0)
TIGHTEN_STOPS_TIME = time(15, 30)
CLOSE_ALL_TIME = time(15, 45)
NO_NEW_ENTRIES_AFTER = time(14, 30)  # Aggregator confidence = 0 after this

# Time decay parameters
TIME_DECAY_START = time(13, 30)  # Start linear confidence decay
TIME_DECAY_END = time(14, 30)    # Confidence reaches 0


class KillZoneManager:
    """Manages kill zone timing and confidence adjustments.

    Core principle: time is a first-class input to every trading decision.
    """

    def __init__(self, penalty_outside_kz: float = 0.2):
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
        for kz in KILL_ZONES:
            if kz.is_active(now_time):
                return kz
        return None

    def get_active_zone_label(self, now: datetime | None = None) -> str | None:
        """Get the label of the currently active kill zone."""
        kz = self.get_active_zone(now)
        return kz.label if kz else None

    def is_no_trade_zone(self, now: datetime | None = None) -> bool:
        """Check if we're in the no-trade zone (after 3:00 PM ET)."""
        if now is None:
            now = self.get_et_now()
        elif now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        elif now.tzinfo != ET:
            now = now.astimezone(ET)

        return now.time() >= NO_TRADE_START

    def should_close_all(self, now: datetime | None = None) -> bool:
        """Check if it's time to close all positions (3:45 PM ET)."""
        if now is None:
            now = self.get_et_now()
        elif now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        elif now.tzinfo != ET:
            now = now.astimezone(ET)

        return now.time() >= CLOSE_ALL_TIME

    def should_tighten_stops(self, now: datetime | None = None) -> bool:
        """Check if we should tighten stops (3:30 PM ET)."""
        if now is None:
            now = self.get_et_now()
        elif now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        elif now.tzinfo != ET:
            now = now.astimezone(ET)

        return now.time() >= TIGHTEN_STOPS_TIME

    def can_enter_new_trade(self, now: datetime | None = None) -> bool:
        """Check if new trade entries are allowed at this time."""
        if now is None:
            now = self.get_et_now()
        elif now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        elif now.tzinfo != ET:
            now = now.astimezone(ET)

        now_time = now.time()

        # Hard cutoff: no entries after 2:30 PM ET
        if now_time >= NO_NEW_ENTRIES_AFTER:
            return False

        # Must be within trading session
        if now_time < time(9, 30) or now_time >= time(16, 0):
            # Allow London open entries if configured
            london_kz = KILL_ZONES[0]
            if london_kz.is_active(now_time):
                return True
            return False

        return True

    def minutes_until_close(self, now: datetime | None = None) -> float:
        """Calculate minutes until the mandatory 3:45 PM close."""
        if now is None:
            now = self.get_et_now()
        elif now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        elif now.tzinfo != ET:
            now = now.astimezone(ET)

        close_dt = now.replace(hour=15, minute=45, second=0, microsecond=0)
        if now >= close_dt:
            return 0.0

        delta = close_dt - now
        return delta.total_seconds() / 60.0

    def apply_time_decay(self, confidence: float, now: datetime | None = None) -> float:
        """Apply time-based confidence decay.

        Before 1:30 PM: no penalty (full confidence)
        1:30 PM - 2:30 PM: linear decay based on time remaining
        After 2:30 PM: confidence = 0 (no entries)

        This is applied AFTER all other signal scoring as a final gate.
        """
        if now is None:
            now = self.get_et_now()
        elif now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        elif now.tzinfo != ET:
            now = now.astimezone(ET)

        now_time = now.time()

        # Before decay start: no penalty
        if now_time < TIME_DECAY_START:
            return confidence

        # After cutoff: zero confidence
        if now_time >= TIME_DECAY_END:
            return 0.0

        # Linear decay between start and end
        start_minutes = TIME_DECAY_START.hour * 60 + TIME_DECAY_START.minute
        end_minutes = TIME_DECAY_END.hour * 60 + TIME_DECAY_END.minute
        now_minutes = now_time.hour * 60 + now_time.minute

        total_window = end_minutes - start_minutes
        elapsed = now_minutes - start_minutes
        decay_factor = 1.0 - (elapsed / total_window)

        return confidence * max(decay_factor, 0.0)

    def adjust_confidence(self, confidence: float, now: datetime | None = None) -> float:
        """Apply kill zone confidence adjustments.

        1. If in a kill zone: multiply by zone weight
        2. If outside all kill zones: subtract penalty
        3. Apply time decay as final gate
        """
        if now is None:
            now = self.get_et_now()

        # Check kill zone
        kz = self.get_active_zone(now)
        if kz:
            adjusted = confidence * kz.weight
        else:
            adjusted = max(confidence - self._penalty, 0.0)

        # Apply time decay
        adjusted = self.apply_time_decay(adjusted, now)

        return max(adjusted, 0.0)

    def get_time_adjusted_tp(
        self,
        full_tp_r: float,
        now: datetime | None = None,
        avg_trade_duration_min: float = 45.0,
    ) -> float:
        """Calculate time-adjusted take profit target.

        Before 1:30 PM: full TP (e.g., 2.5R)
        After 1:30 PM: reduce proportionally (min 1.0R)
        """
        mins_left = self.minutes_until_close(now)

        # If plenty of time, use full TP
        if mins_left > avg_trade_duration_min * 2:
            return full_tp_r

        # Scale down TP based on time remaining
        time_factor = mins_left / (avg_trade_duration_min * 2)
        adjusted = max(full_tp_r * time_factor, 1.0)

        return min(adjusted, full_tp_r)

    def has_enough_time(
        self,
        now: datetime | None = None,
        avg_trade_duration_min: float = 45.0,
    ) -> bool:
        """Check if there's enough time for a trade to reach its target.

        Rule: time_remaining must be >= avg_duration * 1.5
        """
        mins_left = self.minutes_until_close(now)
        return mins_left >= avg_trade_duration_min * 1.5
