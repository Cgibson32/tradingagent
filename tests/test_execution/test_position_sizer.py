"""Tests for position sizing."""

from __future__ import annotations

import pytest

from src.execution.position_sizer import PositionSizer


@pytest.fixture
def sizer():
    return PositionSizer(
        point_value=20.0,
        tick_size=0.25,
        tick_value=5.0,
        commission_per_contract=0.82,
        slippage_ticks=2,
        max_contracts=4,
        risk_per_trade_pct=0.01,
        max_risk_per_trade_pct=0.015,
    )


class TestPositionSizer:
    def test_basic_sizing(self, sizer):
        """$150K account, 20pt stop, 1% risk = $1,500 risk / ~$411 per contract = 3."""
        size = sizer.calculate(
            account_balance=150000,
            entry_price=20100.0,
            stop_loss_price=20080.0,  # 20 point stop
        )
        # Risk per contract: 20 * $20 + 2 * $5 + $0.82 = $400 + $10 + $0.82 = $410.82
        # Target risk: $150K * 1% = $1,500
        # Size: floor(1500 / 410.82) = 3
        assert size == 3

    def test_max_contracts_cap(self, sizer):
        """Even with huge account, cap at 4 contracts."""
        size = sizer.calculate(
            account_balance=1000000,  # $1M
            entry_price=20100.0,
            stop_loss_price=20095.0,  # Tight 5pt stop
        )
        assert size <= 4

    def test_zero_stop_distance(self, sizer):
        size = sizer.calculate(
            account_balance=150000,
            entry_price=20100.0,
            stop_loss_price=20100.0,  # Zero distance
        )
        assert size == 0

    def test_constrained_by_daily_loss(self, sizer):
        """If only $300 of daily loss remaining, should size accordingly."""
        size = sizer.calculate(
            account_balance=150000,
            entry_price=20100.0,
            stop_loss_price=20080.0,  # 20pt stop, ~$411/contract
            daily_loss_remaining=300.0,  # Can't even afford 1 contract
        )
        assert size == 0

    def test_constrained_by_drawdown(self, sizer):
        size = sizer.calculate(
            account_balance=150000,
            entry_price=20100.0,
            stop_loss_price=20080.0,
            drawdown_remaining=500.0,  # Only room for 1 contract
        )
        assert size == 1

    def test_max_risk_dollars(self, sizer):
        risk = sizer.max_risk_dollars(
            contracts=2,
            entry_price=20100.0,
            stop_loss=20080.0,
        )
        # 2 * (20 * 20 + 2 * 5 + 0.82) = 2 * 410.82 = 821.64
        assert abs(risk - 821.64) < 1.0

    def test_small_account(self, sizer):
        """Very small account should still get at least 0 or 1."""
        size = sizer.calculate(
            account_balance=5000,
            entry_price=20100.0,
            stop_loss_price=20080.0,
        )
        # $5K * 1% = $50 risk, $411/contract → 0 contracts
        assert size == 0

    def test_wide_stop(self, sizer):
        """Wide stop should reduce size."""
        narrow = sizer.calculate(150000, 20100.0, 20090.0)  # 10pt stop
        wide = sizer.calculate(150000, 20100.0, 20060.0)    # 40pt stop
        assert narrow >= wide
