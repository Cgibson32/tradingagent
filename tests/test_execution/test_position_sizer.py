"""Tests for position sizer with MNQ/NQ auto-scaling."""

from __future__ import annotations

import pytest

from src.execution.position_sizer import PositionSizer


@pytest.fixture
def sizer():
    return PositionSizer()


class TestInstrumentSelection:
    def test_mnq_below_25k(self, sizer):
        assert sizer.get_instrument(10000) == "mnq"
        assert sizer.get_instrument(24999) == "mnq"

    def test_nq_at_50k(self, sizer):
        assert sizer.get_instrument(50000) == "nq"
        assert sizer.get_instrument(100000) == "nq"

    def test_mnq_between_25k_50k(self, sizer):
        assert sizer.get_instrument(30000) == "mnq"


class TestMNQSizing:
    def test_basic_mnq_sizing(self, sizer):
        """$10K account, 20pt stop on MNQ, 1% risk = $100.
        Risk per contract = 20 * $2 + 2*$0.50 + $0.62 = $41.62
        $100 / $41.62 = 2.4 → 2 contracts."""
        size, instrument = sizer.calculate(
            account_equity=10000,
            entry_price=20100.0,
            stop_loss_price=20080.0,
        )
        assert instrument == "mnq"
        assert size == 2

    def test_max_mnq_contracts(self, sizer):
        size, instrument = sizer.calculate(
            account_equity=20000,
            entry_price=20100.0,
            stop_loss_price=20095.0,
        )
        assert instrument == "mnq"
        assert size <= 4

    def test_zero_size_on_tiny_account(self, sizer):
        size, instrument = sizer.calculate(
            account_equity=500,
            entry_price=20100.0,
            stop_loss_price=19900.0,
        )
        assert size == 0

    def test_zero_stop_distance(self, sizer):
        size, _ = sizer.calculate(
            account_equity=10000,
            entry_price=20100.0,
            stop_loss_price=20100.0,
        )
        assert size == 0


class TestNQSizing:
    def test_nq_at_50k(self, sizer):
        size, instrument = sizer.calculate(
            account_equity=50000,
            entry_price=20100.0,
            stop_loss_price=20080.0,
        )
        assert instrument == "nq"
        assert size == 1

    def test_max_nq_contracts(self, sizer):
        size, instrument = sizer.calculate(
            account_equity=200000,
            entry_price=20100.0,
            stop_loss_price=20095.0,
        )
        assert instrument == "nq"
        assert size <= 2


class TestMarginChecking:
    def test_margin_limits_size(self, sizer):
        size, instrument = sizer.calculate(
            account_equity=10000,
            entry_price=20100.0,
            stop_loss_price=20095.0,
            check_overnight_margin=True,
        )
        assert instrument == "mnq"
        assert size <= 3

    def test_day_margin_allows_more(self, sizer):
        size_overnight, _ = sizer.calculate(
            account_equity=10000,
            entry_price=20100.0,
            stop_loss_price=20095.0,
            check_overnight_margin=True,
        )
        size_day, _ = sizer.calculate(
            account_equity=10000,
            entry_price=20100.0,
            stop_loss_price=20095.0,
            check_overnight_margin=False,
        )
        assert size_day >= size_overnight


class TestRiskCalculation:
    def test_max_risk_dollars_mnq(self, sizer):
        risk = sizer.max_risk_dollars(
            contracts=2, entry_price=20100.0, stop_loss=20080.0,
            instrument="mnq",
        )
        assert 80 < risk < 90

    def test_margin_required(self, sizer):
        margin = sizer.margin_required(2, "mnq", overnight=True)
        assert margin == 4200.0

    def test_force_instrument(self, sizer):
        size, instrument = sizer.calculate(
            account_equity=10000,
            entry_price=20100.0,
            stop_loss_price=20080.0,
            instrument="nq",
        )
        assert instrument == "nq"
        assert size >= 0
