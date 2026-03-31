"""Tests for trade logger."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from src.feedback.trade_logger import TradeLogger


@pytest.fixture
async def logger_instance(tmp_path):
    db_path = str(tmp_path / "test_trading.db")
    tl = TradeLogger(db_path)
    await tl.initialize()
    return tl


class TestTradeLogger:
    @pytest.mark.asyncio
    async def test_initialize(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        tl = TradeLogger(db_path)
        await tl.initialize()
        assert os.path.exists(db_path)

    @pytest.mark.asyncio
    async def test_log_and_retrieve_trade(self, logger_instance):
        tl = logger_instance
        trade_id = await tl.log_trade(
            symbol="NQU5",
            direction="long",
            entry_time=datetime.now(timezone.utc),
            exit_time=None,
            entry_price=20100.0,
            exit_price=None,
            position_size=2,
            stop_loss=20080.0,
            take_profit=20150.0,
            status="open",
            regime="trending_bullish",
            strategy_name="ict_aggregated",
        )
        assert trade_id is not None

        trades = await tl.get_recent_trades(n=5)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "NQU5"

    @pytest.mark.asyncio
    async def test_update_trade(self, logger_instance):
        tl = logger_instance
        trade_id = await tl.log_trade(
            symbol="NQU5", direction="long",
            entry_time=datetime.now(timezone.utc), exit_time=None,
            entry_price=20100.0, exit_price=None,
            position_size=1, stop_loss=20080.0, take_profit=20150.0,
        )

        await tl.update_trade(
            trade_id, exit_price=20140.0,
            exit_time=datetime.now(timezone.utc),
            pnl_dollars=800.0, pnl_r=2.0, status="win",
        )

        trades = await tl.get_recent_trades()
        assert trades[0]["status"] == "win"
        assert trades[0]["pnl_r"] == 2.0

    @pytest.mark.asyncio
    async def test_log_ai_decision(self, logger_instance):
        tl = logger_instance
        await tl.log_ai_decision(
            decision_type="signal_eval",
            input_context='{"signal": "long"}',
            output_decision='{"decision": "approve"}',
            model="claude-sonnet-4-20250514",
            tokens=500,
            cost_usd=0.01,
            latency_ms=1500,
        )
        # Should not raise
