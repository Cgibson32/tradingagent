"""Tests for the immutable risk guard.

These tests verify that the safety layer CANNOT be bypassed.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from src.execution.risk_guard import RiskGuard, _MAX_CONTRACTS

ET = ZoneInfo("America/New_York")


def _make_et(hour, minute=0):
    return datetime(2025, 6, 15, hour, minute, tzinfo=ET)


@pytest.fixture
def guard():
    g = RiskGuard()
    g.update_equity(150000.0)
    return g


class TestRiskGuardTrading:
    def test_can_trade_during_session(self, guard):
        allowed, violations = guard.can_trade(_make_et(10, 0))
        assert allowed is True
        assert len(violations) == 0

    def test_blocked_outside_hours(self, guard):
        allowed, violations = guard.can_trade(_make_et(8, 0))
        assert allowed is False
        assert any(v.rule == "trading_hours" for v in violations)

    def test_blocked_after_3pm(self, guard):
        allowed, violations = guard.can_trade(_make_et(15, 0))
        assert allowed is False
        assert any(v.rule == "no_new_entries" for v in violations)

    def test_blocked_daily_loss(self, guard):
        guard.update_pnl(-1500.0)  # Hit effective limit
        allowed, violations = guard.can_trade(_make_et(10, 0))
        assert allowed is False
        assert any(v.rule == "daily_loss" for v in violations)

    def test_blocked_max_trades(self, guard):
        for _ in range(6):
            guard.record_trade(is_loss=False)
        allowed, violations = guard.can_trade(_make_et(10, 0))
        assert allowed is False
        assert any(v.rule == "max_trades" for v in violations)

    def test_blocked_max_positions(self, guard):
        guard.set_open_positions(2)
        allowed, violations = guard.can_trade(_make_et(10, 0))
        assert allowed is False
        assert any(v.rule == "max_positions" for v in violations)

    def test_cooldown_after_loss(self, guard):
        guard.record_trade(is_loss=True)
        # Immediately after loss — should be in cooldown
        allowed, violations = guard.can_trade(_make_et(10, 0))
        assert allowed is False
        assert any(v.rule == "cooldown" for v in violations)

    def test_consecutive_losses(self, guard):
        for _ in range(3):
            guard.record_trade(is_loss=True)
        # Wait past cooldown by checking well after
        guard._last_loss_time = _make_et(9, 0)
        allowed, violations = guard.can_trade(_make_et(10, 0))
        assert allowed is False
        assert any(v.rule == "consecutive_losses" for v in violations)


class TestRiskGuardOrderValidation:
    def test_valid_order(self, guard):
        valid, violations = guard.validate_order(
            qty=2, entry_price=20100.0, stop_loss=20080.0, account_balance=150000.0
        )
        assert valid is True

    def test_reject_over_max_contracts(self, guard):
        valid, violations = guard.validate_order(
            qty=5, entry_price=20100.0, stop_loss=20080.0, account_balance=150000.0
        )
        assert valid is False
        assert any(v.rule == "max_contracts" for v in violations)

    def test_reject_no_stop_loss(self, guard):
        valid, violations = guard.validate_order(
            qty=1, entry_price=20100.0, stop_loss=0, account_balance=150000.0
        )
        assert valid is False
        assert any(v.rule == "no_stop_loss" for v in violations)

    def test_reject_excessive_risk(self, guard):
        """Wide stop that risks >1.5% should be rejected."""
        valid, violations = guard.validate_order(
            qty=4, entry_price=20100.0, stop_loss=19900.0,  # 200pt stop
            account_balance=150000.0,
        )
        assert valid is False
        assert any(v.rule == "max_risk" for v in violations)

    def test_reject_would_breach_daily(self, guard):
        guard.update_pnl(-1400.0)  # Only $100 left before limit
        valid, violations = guard.validate_order(
            qty=1, entry_price=20100.0, stop_loss=20080.0,
            account_balance=150000.0,
        )
        assert valid is False
        assert any(v.rule == "daily_loss_projected" for v in violations)


class TestRiskGuardImmutability:
    def test_max_contracts_is_4(self):
        """Verify the hardcoded max contracts cannot be changed."""
        assert _MAX_CONTRACTS == 4

    def test_effective_limits(self, guard):
        assert guard.effective_daily_limit == 1500.0
        assert guard.effective_drawdown_limit == 4000.0


class TestRiskGuardEmergency:
    def test_emergency_exit_near_floor(self, guard):
        guard._drawdown_floor = 146000.0
        guard.update_equity(146050.0)  # Only $50 above floor
        assert guard.check_emergency_exit() is True

    def test_no_emergency_when_safe(self, guard):
        guard._drawdown_floor = 145600.0
        guard.update_equity(150000.0)
        assert guard.check_emergency_exit() is False

    def test_must_close_at_345(self, guard):
        assert guard.must_close_all(_make_et(15, 45)) is True
        assert guard.must_close_all(_make_et(15, 44)) is False

    def test_daily_reset(self, guard):
        guard.update_pnl(-500.0)
        guard.record_trade(is_loss=True)
        guard.record_trade(is_loss=True)
        guard.reset_daily()
        assert guard._daily_pnl == 0.0
        assert guard._trades_today == 0
        assert guard._consecutive_losses == 0
