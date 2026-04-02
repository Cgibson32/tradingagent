"""Tests for trade logger."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from src.feedback.trade_logger import TradeLogger


@pytest.fixture
async def tl(tmp_path):
    db_path = str(tmp_path / "test_trading.db")
    logger = TradeLogger(db_path)
    await logger._ensure_db()
    return logger


class TestTradeLogger:
    @pytest.mark.asyncio
    async def test_log_trade_open(self, tl):
        trade_id = await tl.log_trade_open(
            symbol="MNQU5", direction="long",
            entry_price=20100.0, stop_loss=20080.0, take_profit=20150.0,
            position_size=2, strategy_name="ict_aggregated",
            signal_confidence=0.75, regime="trending_bullish",
        )
        assert trade_id is not None
        assert len(trade_id) > 0

    @pytest.mark.asyncio
    async def test_log_and_retrieve(self, tl):
        trade_id = await tl.log_trade_open(
            symbol="MNQU5", direction="long",
            entry_price=20100.0, stop_loss=20080.0, take_profit=20150.0,
            position_size=1,
        )
        trades = await tl.get_recent_trades(5)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "MNQU5"
        assert trades[0]["status"] == "open"

    @pytest.mark.asyncio
    async def test_log_trade_close(self, tl):
        trade_id = await tl.log_trade_open(
            symbol="MNQU5", direction="long",
            entry_price=20100.0, stop_loss=20080.0, take_profit=20150.0,
            position_size=1,
        )
        await tl.log_trade_close(
            trade_id, exit_price=20140.0,
            pnl_dollars=80.0, pnl_r=2.0, fees=1.24,
        )
        trades = await tl.get_recent_trades()
        assert trades[0]["status"] == "win"
        assert trades[0]["pnl_r"] == 2.0

    @pytest.mark.asyncio
    async def test_log_losing_trade(self, tl):
        trade_id = await tl.log_trade_open(
            symbol="MNQU5", direction="short",
            entry_price=20100.0, stop_loss=20120.0, take_profit=20050.0,
            position_size=1,
        )
        await tl.log_trade_close(
            trade_id, exit_price=20120.0,
            pnl_dollars=-40.0, pnl_r=-1.0,
        )
        trades = await tl.get_recent_trades()
        assert trades[0]["status"] == "loss"

    @pytest.mark.asyncio
    async def test_log_ai_decision(self, tl):
        await tl.log_ai_decision(
            decision_type="signal_eval",
            input_context={"signal": "long"},
            output_decision={"decision": "approve"},
            model_used="claude-sonnet-4-20250514",
            tokens_used=500, cost_usd=0.01, latency_ms=1500,
        )
        # Should not raise

    @pytest.mark.asyncio
    async def test_log_daily_stats(self, tl):
        await tl.log_daily_stats(
            date="2025-06-15", starting_equity=10000.0,
            ending_equity=10200.0, total_pnl=200.0,
            trades_taken=3, wins=2, losses=1,
        )
        stats = await tl.get_daily_stats(5)
        assert len(stats) == 1
        assert stats[0]["total_pnl"] == 200.0

    @pytest.mark.asyncio
    async def test_multiple_trades(self, tl):
        for i in range(5):
            tid = await tl.log_trade_open(
                symbol="MNQU5", direction="long",
                entry_price=20100.0 + i, stop_loss=20080.0, take_profit=20150.0,
                position_size=1,
            )
            await tl.log_trade_close(tid, exit_price=20110.0, pnl_dollars=20.0, pnl_r=1.0)
        trades = await tl.get_recent_trades(10)
        assert len(trades) == 5
