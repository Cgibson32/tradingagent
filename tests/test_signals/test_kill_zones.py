"""Tests for kill zones — advisory-only confidence weighting."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from src.signals.kill_zones import KillZoneManager, KILL_ZONES, ET


@pytest.fixture
def kz():
    return KillZoneManager(penalty_outside_kz=0.15)


def _make_et(hour, minute=0):
    """Helper to create a datetime in ET."""
    return datetime(2025, 6, 15, hour, minute, tzinfo=ET)


class TestKillZoneManager:
    def test_ny_open_active(self, kz):
        now = _make_et(10, 0)
        zone = kz.get_active_zone(now)
        assert zone is not None
        assert zone.label == "NY Open"

    def test_ny_lunch_active(self, kz):
        now = _make_et(12, 30)
        zone = kz.get_active_zone(now)
        assert zone is not None
        assert zone.label == "NY Lunch"

    def test_london_open(self, kz):
        now = _make_et(3, 0)
        zone = kz.get_active_zone(now)
        assert zone is not None
        assert zone.label == "London Open"

    def test_overnight_session(self, kz):
        now = _make_et(18, 0)  # 6 PM ET — overnight session
        zone = kz.get_active_zone(now)
        assert zone is not None
        assert zone.label == "Overnight/Asia"

    def test_overnight_wraps_midnight(self, kz):
        now = _make_et(1, 0)  # 1 AM ET — still overnight
        zone = kz.get_active_zone(now)
        assert zone is not None
        # Should be either overnight or London open
        assert zone.label in ("Overnight/Asia", "London Open")

    def test_maintenance_halt_no_zone(self, kz):
        now = _make_et(16, 30)  # During halt
        zone = kz.get_active_zone(now)
        assert zone is None

    def test_is_maintenance_halt(self, kz):
        assert kz.is_maintenance_halt(_make_et(16, 30)) is True
        assert kz.is_maintenance_halt(_make_et(10, 0)) is False
        assert kz.is_maintenance_halt(_make_et(17, 0)) is False  # Halt ends at 5 PM


class TestConfidenceAdjustment:
    def test_ny_open_full_confidence(self, kz):
        """NY Open (weight=1.0) should not reduce confidence."""
        result = kz.adjust_confidence(0.8, _make_et(10, 0))
        assert result == 0.8

    def test_ny_lunch_reduced(self, kz):
        """NY Lunch (weight=0.7) should reduce confidence."""
        result = kz.adjust_confidence(1.0, _make_et(12, 30))
        assert result == pytest.approx(0.7)

    def test_overnight_lower_confidence(self, kz):
        """Overnight (weight=0.6) gets lower confidence."""
        result = kz.adjust_confidence(1.0, _make_et(18, 0))
        assert result == pytest.approx(0.6)

    def test_maintenance_halt_zero(self, kz):
        """During maintenance halt, confidence is 0."""
        result = kz.adjust_confidence(0.9, _make_et(16, 30))
        assert result == 0.0

    def test_penalty_outside_all_zones(self, kz):
        """Between zones, penalty is applied."""
        # 5:30 AM ET — between London and NY
        now = _make_et(5, 30)
        zone = kz.get_active_zone(now)
        if zone is None:
            result = kz.adjust_confidence(0.8, now)
            assert result == pytest.approx(0.65)  # 0.8 - 0.15

    def test_no_hard_entry_blocks(self, kz):
        """Advisory only — no time should produce exactly 0 confidence
        except during maintenance halt."""
        for hour in range(24):
            if 16 <= hour < 17:  # Skip maintenance halt
                continue
            result = kz.adjust_confidence(0.8, _make_et(hour, 0))
            assert result > 0, f"Confidence should be > 0 at {hour}:00 ET"
