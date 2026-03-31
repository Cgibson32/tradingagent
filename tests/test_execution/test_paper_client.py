"""Tests for paper trading client."""

from __future__ import annotations

import pytest

from src.data.models import OrderAction, OrderType
from src.execution.paper_client import PaperTradingClient


@pytest.fixture
def paper():
    client = PaperTradingClient(initial_balance=150000.0)
    client.update_price("NQU5", 20100.0)
    return client


class TestPaperTradingClient:
    def test_initial_state(self, paper):
        assert paper.balance == 150000.0
        assert paper.daily_pnl == 0.0

    @pytest.mark.asyncio
    async def test_market_buy_fill(self, paper):
        result = await paper.place_order(
            account_id=1, account_spec="test",
            symbol="NQU5", action=OrderAction.BUY, qty=1,
            order_type=OrderType.MARKET,
        )
        assert result.status == "filled"
        # Should have a position now
        positions = await paper.list_positions()
        nq = [p for p in positions if p.symbol == "NQU5" and not p.is_flat]
        assert len(nq) == 1
        assert nq[0].net_pos == 1

    @pytest.mark.asyncio
    async def test_round_trip_pnl(self, paper):
        # Buy at 20100
        paper.update_price("NQU5", 20100.0)
        await paper.place_order(
            account_id=1, account_spec="test",
            symbol="NQU5", action=OrderAction.BUY, qty=1,
            order_type=OrderType.MARKET,
        )

        # Sell at 20120 (20 points profit)
        paper.update_price("NQU5", 20120.0)
        await paper.place_order(
            account_id=1, account_spec="test",
            symbol="NQU5", action=OrderAction.SELL, qty=1,
            order_type=OrderType.MARKET,
        )

        # P&L should be roughly 20 * $20 - commissions - slippage
        # 20 * 20 = $400, minus ~$1.64 commission, minus some slippage
        assert paper.daily_pnl > 350  # Generous range for slippage
        assert paper.daily_pnl < 400

    @pytest.mark.asyncio
    async def test_stop_order_trigger(self, paper):
        # Buy first
        await paper.place_order(
            account_id=1, account_spec="test",
            symbol="NQU5", action=OrderAction.BUY, qty=1,
            order_type=OrderType.MARKET,
        )

        # Place a sell stop at 20080
        await paper.place_order(
            account_id=1, account_spec="test",
            symbol="NQU5", action=OrderAction.SELL, qty=1,
            order_type=OrderType.STOP, stop_price=20080.0,
        )

        # Price drops to trigger stop
        paper.update_price("NQU5", 20079.0)

        # Position should be closed
        positions = await paper.list_positions()
        nq = [p for p in positions if p.symbol == "NQU5"]
        assert all(p.is_flat for p in nq)

    @pytest.mark.asyncio
    async def test_cancel_order(self, paper):
        result = await paper.place_order(
            account_id=1, account_spec="test",
            symbol="NQU5", action=OrderAction.BUY, qty=1,
            order_type=OrderType.LIMIT, price=20050.0,
        )
        # Limit not filled yet
        orders = await paper.list_orders()
        assert len(orders) == 1

        await paper.cancel_order(result.order_id)
        orders = await paper.list_orders()
        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_cash_balance(self, paper):
        balance = await paper.get_cash_balance(1)
        assert balance.total_equity == 150000.0
        assert balance.cash_balance == 150000.0

    @pytest.mark.asyncio
    async def test_find_contract(self, paper):
        contract = await paper.find_contract("NQU5")
        assert contract["name"] == "NQU5"

    def test_daily_reset(self, paper):
        paper._daily_pnl = 500.0
        paper.reset_daily()
        assert paper.daily_pnl == 0.0
