"""Tests for kill zones and time-aware trading."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from src.signals.kill_zones import KillZoneManager, KILL_ZONES, ET


@pytest.fixture
def kz():
    return KillZoneManager(penalty_outside_kz=0.2)


def _make_et(hour, minute=0):
    """Helper to create a datetime in ET."""
    return datetime(2025, 6, 15, hour, minute, tzinfo=ET)


class TestKillZoneManager:
    def test_ny_open_active(self, kz):
        now = _make_et(10, 0)  # 10:00 AM ET
        zone = kz.get_active_zone(now)
        assert zone is not None
        assert zone.label == "NY Open"

    def test_ny_lunch_active(self, kz):
        now = _make_et(12, 30)  # 12:30 PM ET
        zone = kz.get_active_zone(now)
        assert zone is not None
        assert zone.label == "NY Lunch"

    def test_no_zone_at_midday(self, kz):
        now = _make_et(11, 30)  # 11:30 AM ET — between NY Open and Lunch
        zone = kz.get_active_zone(now)
        assert zone is None

    def test_london_open(self, kz):
        now = _make_et(3, 0)  # 3:00 AM ET
        zone = kz.get_active_zone(now)
        assert zone is not None
        assert zone.label == "London Open"

    def test_is_no_trade_zone(self, kz):
        assert kz.is_no_trade_zone(_make_et(15, 0)) is True
        assert kz.is_no_trade_zone(_make_et(15, 30)) is True
        assert kz.is_no_trade_zone(_make_et(14, 59)) is False

    def test_should_close_all(self, kz):
        assert kz.should_close_all(_make_et(15, 45)) is True
        assert kz.should_close_all(_make_et(15, 44)) is False

    def test_should_tighten_stops(self, kz):
        assert kz.should_tighten_stops(_make_et(15, 30)) is True
        assert kz.should_tighten_stops(_make_et(15, 29)) is False

    def test_can_enter_new_trade(self, kz):
        # During NY Open — yes
        assert kz.can_enter_new_trade(_make_et(10, 0)) is True
        # After 2:30 PM — no
        assert kz.can_enter_new_trade(_make_et(14, 30)) is False
        assert kz.can_enter_new_trade(_make_et(15, 0)) is False
        # London open — yes
        assert kz.can_enter_new_trade(_make_et(3, 0)) is True

    def test_minutes_until_close(self, kz):
        mins = kz.minutes_until_close(_make_et(15, 0))
        assert mins == 45.0

        mins = kz.minutes_until_close(_make_et(14, 0))
        assert mins == 105.0

        # After close time
        mins = kz.minutes_until_close(_make_et(16, 0))
        assert mins == 0.0


class TestTimeDecay:
    def test_no_decay_before_start(self, kz):
        result = kz.apply_time_decay(0.8, _make_et(10, 0))
        assert result == 0.8

    def test_zero_after_cutoff(self, kz):
        result = kz.apply_time_decay(0.8, _make_et(14, 30))
        assert result == 0.0

    def test_linear_decay_midpoint(self, kz):
        # At 2:00 PM (midpoint of 1:30-2:30 window)
        result = kz.apply_time_decay(1.0, _make_et(14, 0))
        assert 0.4 <= result <= 0.6  # Should be ~0.5

    def test_confidence_adjustment_in_kz(self, kz):
        # In NY Open (weight=1.0), before decay
        result = kz.adjust_confidence(0.8, _make_et(10, 0))
        assert result == 0.8  # No change

    def test_confidence_penalty_outside_kz(self, kz):
        # Outside any kill zone (11:30 AM), before decay
        result = kz.adjust_confidence(0.8, _make_et(11, 30))
        assert result == pytest.approx(0.6)  # 0.8 - 0.2 penalty


class TestTimeAdjustedTP:
    def test_full_tp_early(self, kz):
        tp = kz.get_time_adjusted_tp(2.5, _make_et(10, 0), avg_trade_duration_min=45)
        assert tp == 2.5

    def test_reduced_tp_late(self, kz):
        tp = kz.get_time_adjusted_tp(2.5, _make_et(14, 45), avg_trade_duration_min=45)
        assert tp < 2.5
        assert tp >= 1.0  # Minimum

    def test_has_enough_time(self, kz):
        # 2+ hours left, 45min avg → yes
        assert kz.has_enough_time(_make_et(10, 0), avg_trade_duration_min=45) is True
        # 30 min left, 45 min avg → no (need 67.5 min)
        assert kz.has_enough_time(_make_et(15, 15), avg_trade_duration_min=45) is False
