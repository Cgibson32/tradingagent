"""Tests for risk guard — personal account rules."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from src.execution.risk_guard import RiskGuard, RiskViolation

ET = ZoneInfo("America/New_York")


def _make_et(hour, minute=0):
    return datetime(2025, 6, 15, hour, minute, tzinfo=ET)


@pytest.fixture
def guard():
    g = RiskGuard(
        max_risk_per_trade_pct=0.02,
        max_daily_loss_pct=0.03,
        max_weekly_loss_pct=0.05,
        max_contracts=4,
        max_concurrent_positions=2,
        max_trades_per_day=6,
        cooldown_after_loss_minutes=15,
        news_blackout_minutes=15,
        max_consecutive_losses=3,
        account_size=10000.0,
    )
    return g


class TestRiskGuard:
    def test_can_trade_normal(self, guard):
        allowed, violations = guard.can_trade(_make_et(10, 0))
        assert allowed is True
        assert violations == []

    def test_blocks_during_maintenance_halt(self, guard):
        allowed, violations = guard.can_trade(_make_et(16, 30))
        assert allowed is False
        assert any(v.rule == "daily_halt" for v in violations)

    def test_allows_after_halt(self, guard):
        allowed, _ = guard.can_trade(_make_et(17, 1))
        assert allowed is True

    def test_daily_loss_limit_percentage(self, guard):
        # $10K * 3% = $300 daily limit
        assert guard.daily_loss_limit == 300.0
        guard.update_pnl(daily_pnl=-310.0)
        allowed, violations = guard.can_trade(_make_et(10, 0))
        assert allowed is False
        assert any(v.rule == "daily_loss" for v in violations)

    def test_weekly_loss_limit(self, guard):
        # $10K * 5% = $500 weekly limit
        assert guard.weekly_loss_limit == 500.0
        guard.update_pnl(daily_pnl=-100, weekly_pnl=-510.0)
        allowed, violations = guard.can_trade(_make_et(10, 0))
        assert allowed is False
        assert any(v.rule == "weekly_loss" for v in violations)

    def test_max_trades_per_day(self, guard):
        for _ in range(6):
            guard.record_trade(is_loss=False)
        allowed, violations = guard.can_trade(_make_et(10, 0))
        assert allowed is False
        assert any(v.rule == "max_trades" for v in violations)

    def test_max_concurrent_positions(self, guard):
        guard.set_open_positions(2)
        allowed, violations = guard.can_trade(_make_et(10, 0))
        assert allowed is False
        assert any(v.rule == "max_positions" for v in violations)

    def test_cooldown_after_loss(self, guard):
        guard.record_trade(is_loss=True)
        # Immediately after loss — should be blocked
        allowed, violations = guard.can_trade(_make_et(10, 0))
        assert allowed is False
        assert any(v.rule == "cooldown" for v in violations)

    def test_consecutive_losses(self, guard):
        for _ in range(3):
            guard.record_trade(is_loss=True)
        # Reset cooldown time to avoid that violation
        guard._last_loss_time = _make_et(8, 0)
        allowed, violations = guard.can_trade(_make_et(10, 0))
        assert allowed is False
        assert any(v.rule == "consecutive_losses" for v in violations)

    def test_news_blackout(self, guard):
        news_time = _make_et(10, 5)
        guard.set_news_times([news_time])
        allowed, violations = guard.can_trade(_make_et(10, 0))
        assert allowed is False
        assert any(v.rule == "news_blackout" for v in violations)

    def test_reset_daily(self, guard):
        guard.update_pnl(daily_pnl=-200)
        for _ in range(5):
            guard.record_trade(is_loss=True)
        guard.reset_daily()
        assert guard._daily_pnl == 0.0
        assert guard._trades_today == 0
        assert guard._consecutive_losses == 0

    def test_allows_overnight_trading(self, guard):
        """Personal account: overnight trading is allowed."""
        allowed, _ = guard.can_trade(_make_et(20, 0))
        assert allowed is True

        allowed, _ = guard.can_trade(_make_et(2, 0))
        assert allowed is True


class TestValidateOrder:
    def test_valid_order(self, guard):
        valid, violations = guard.validate_order(
            qty=1, entry_price=20100.0, stop_loss=20090.0,
            point_value=2.0,
        )
        assert valid is True

    def test_exceeds_max_contracts(self, guard):
        valid, violations = guard.validate_order(
            qty=5, entry_price=20100.0, stop_loss=20090.0, point_value=2.0,
        )
        assert valid is False
        assert any(v.rule == "max_contracts" for v in violations)

    def test_no_stop_loss(self, guard):
        valid, violations = guard.validate_order(
            qty=1, entry_price=20100.0, stop_loss=0, point_value=2.0,
        )
        assert valid is False
        assert any(v.rule == "no_stop_loss" for v in violations)

    def test_risk_too_high(self, guard):
        valid, violations = guard.validate_order(
            qty=1, entry_price=20100.0, stop_loss=19900.0, point_value=2.0,
        )
        assert valid is False
        assert any(v.rule == "max_risk" for v in violations)

    def test_margin_check(self, guard):
        valid, violations = guard.validate_order(
            qty=1, entry_price=20100.0, stop_loss=20090.0,
            point_value=2.0, margin_required=9000.0,
        )
        assert valid is False
        assert any(v.rule == "insufficient_margin" for v in violations)


class TestEmergencyExit:
    def test_emergency_near_daily_limit(self, guard):
        guard.update_pnl(daily_pnl=-275.0)
        assert guard.check_emergency_exit() is True

    def test_no_emergency_normal(self, guard):
        guard.update_pnl(daily_pnl=-50.0)
        assert guard.check_emergency_exit() is False

    def test_limits_scale_with_equity(self, guard):
        guard.update_equity(20000.0)
        assert guard.daily_loss_limit == 600.0
        assert guard.weekly_loss_limit == 1000.0
