"""Tests for circuit breaker."""

from __future__ import annotations

import time

import pytest

from src.execution.circuit_breaker import CircuitBreaker


@pytest.fixture
def cb():
    return CircuitBreaker(
        flash_crash_atr_multiple=5.0,
        spread_blowout_multiple=4.0,
        stale_quote_seconds=30.0,
        max_consecutive_losses=3,
    )


class TestCircuitBreaker:
    def test_initial_state(self, cb):
        assert cb.is_tripped is False
        safe, reason = cb.check()
        assert safe is True

    def test_consecutive_losses_trip(self, cb):
        cb.record_loss()
        cb.record_loss()
        assert cb.is_tripped is False
        cb.record_loss()  # 3rd loss
        assert cb.is_tripped is True
        assert "consecutive_losses" in cb.trip_reason

    def test_win_resets_losses(self, cb):
        cb.record_loss()
        cb.record_loss()
        cb.record_win()
        cb.record_loss()  # Only 1 loss now, not 3
        assert cb.is_tripped is False

    def test_flash_crash_detection(self, cb):
        cb.set_baseline(normal_atr=10.0, normal_spread=0.25)
        now = time.time()

        # Simulate normal prices
        for i in range(50):
            cb.update_quote(20100.0 + i * 0.5, 0.25)

        # Simulate flash crash: 60pt move (6x ATR)
        for i in range(10):
            cb.update_quote(20100.0 + 60.0, 0.25)

        safe, reason = cb.check()
        assert safe is False
        assert "flash_crash" in reason

    def test_spread_blowout(self, cb):
        cb.set_baseline(normal_atr=10.0, normal_spread=0.25)

        # Build normal spread history
        for _ in range(30):
            cb.update_quote(20100.0, 0.25)

        # Blow out the spread
        cb.update_quote(20100.0, 2.0)  # 8x normal

        safe, reason = cb.check()
        assert safe is False
        assert "spread_blowout" in reason

    def test_stale_quotes(self, cb):
        cb._last_quote_time = time.time() - 60  # 60 seconds ago
        safe, reason = cb.check()
        assert safe is False
        assert "stale_quotes" in reason

    def test_max_contracts_override_normal(self, cb):
        cb.set_baseline(normal_atr=10.0, normal_spread=0.25)
        cb.update_atr(10.0)
        assert cb.get_max_contracts_override() is None

    def test_max_contracts_override_high_vol(self, cb):
        cb.set_baseline(normal_atr=10.0, normal_spread=0.25)
        cb.update_atr(25.0)  # 2.5x normal
        assert cb.get_max_contracts_override() == 2

    def test_manual_reset(self, cb):
        cb.record_loss()
        cb.record_loss()
        cb.record_loss()
        assert cb.is_tripped is True
        cb.reset()
        assert cb.is_tripped is False
