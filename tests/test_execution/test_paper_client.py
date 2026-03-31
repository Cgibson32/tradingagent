"""Tests for paper trading client."""

from __future__ import annotations

import pytest

from src.data.models import OrderAction, OrderType
from src.execution.paper_client import PaperTradingClient


@pytest.fixture
def client():
    return PaperTradingClient(initial_balance=10000.0)


class TestPaperTradingClient:
    def test_initial_balance(self, client):
        assert client.balance == 10000.0

    @pytest.mark.asyncio
    async def test_market_buy(self, client):
        client.update_price("MNQU5", 20100.0)
        result = await client.place_order(
            account_id=1, account_spec="demo", symbol="MNQU5",
            action=OrderAction.BUY, qty=1, order_type=OrderType.MARKET,
        )
        assert result.status == "filled"
        positions = await client.list_positions()
        assert any(not p.is_flat for p in positions)

    @pytest.mark.asyncio
    async def test_round_trip_pnl(self, client):
        client.update_price("MNQU5", 20100.0)
        await client.place_order(
            account_id=1, account_spec="demo", symbol="MNQU5",
            action=OrderAction.BUY, qty=1, order_type=OrderType.MARKET,
        )
        client.update_price("MNQU5", 20110.0)
        await client.place_order(
            account_id=1, account_spec="demo", symbol="MNQU5",
            action=OrderAction.SELL, qty=1, order_type=OrderType.MARKET,
        )
        assert client.daily_pnl > 0
        assert client.balance > 10000.0

    @pytest.mark.asyncio
    async def test_stop_order_triggers(self, client):
        client.update_price("MNQU5", 20100.0)
        await client.place_order(
            account_id=1, account_spec="demo", symbol="MNQU5",
            action=OrderAction.BUY, qty=1, order_type=OrderType.MARKET,
        )
        await client.place_order(
            account_id=1, account_spec="demo", symbol="MNQU5",
            action=OrderAction.SELL, qty=1, order_type=OrderType.STOP,
            stop_price=20080.0,
        )
        client.update_price("MNQU5", 20080.0)
        positions = await client.list_positions()
        flat = all(p.is_flat for p in positions)
        assert flat is True

    @pytest.mark.asyncio
    async def test_cash_balance(self, client):
        balance = await client.get_cash_balance(1)
        assert balance.cash_balance == 10000.0

    @pytest.mark.asyncio
    async def test_find_contract(self, client):
        contract = await client.find_contract("MNQU5")
        assert contract["name"] == "MNQU5"

    def test_daily_reset(self, client):
        client._daily_pnl = 500.0
        client.reset_daily()
        assert client.daily_pnl == 0.0
