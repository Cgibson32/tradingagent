"""Tests for signal evaluator."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.brain.claude_client import ClaudeClient
from src.brain.evaluator import SignalEvaluator, EvaluationResult
from src.data.models import (
    Direction,
    KeyLevels,
    MarketRegime,
    MarketState,
    OrderFlowState,
    TradeSignal,
)


@pytest.fixture
def mock_claude():
    client = ClaudeClient(api_key="test")
    client.ask = AsyncMock(return_value={
        "decision": "approve",
        "adjusted_size": 2,
        "adjusted_tp_r": 2.0,
        "time_feasibility": "feasible",
        "confidence": 0.8,
        "reasoning": "Good setup with confluence",
        "risk_notes": "None",
        "cautions": [],
    })
    return client


@pytest.fixture
def evaluator(mock_claude):
    return SignalEvaluator(mock_claude)


@pytest.fixture
def sample_signal():
    return TradeSignal(
        direction=Direction.LONG,
        entry_price=20100.0,
        stop_loss=20080.0,
        take_profit=20150.0,
        time_adjusted_tp=20140.0,
        confidence=0.75,
        minutes_until_close=180.0,
        estimated_duration_minutes=45.0,
        strategy_name="ict_aggregated",
    )


@pytest.fixture
def sample_state():
    return MarketState(
        symbol="MNQU5",
        current_price=20100.0,
        regime=MarketRegime.TRENDING_BULLISH,
        regime_confidence=0.8,
        equity=10000.0,
        daily_pnl=50.0,
        trades_today=1,
        consecutive_losses=0,
    )


class TestSignalEvaluator:
    @pytest.mark.asyncio
    async def test_approve_signal(self, evaluator, sample_signal, sample_state):
        result = await evaluator.evaluate(sample_signal, sample_state)
        assert result.decision == "approve"
        assert result.adjusted_size == 2
        assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_fallback_on_claude_failure(self, evaluator, sample_signal, sample_state):
        evaluator._claude.ask = AsyncMock(return_value={"error": "Timeout", "fallback": True})
        result = await evaluator.evaluate(sample_signal, sample_state)
        assert result.used_fallback is True
        # High confidence, good R:R, enough time → should approve in fallback
        assert result.decision == "approve"

    @pytest.mark.asyncio
    async def test_fallback_rejects_low_confidence(self, evaluator, sample_state):
        low_conf_signal = TradeSignal(
            direction=Direction.LONG,
            entry_price=20100.0,
            stop_loss=20080.0,
            take_profit=20150.0,
            confidence=0.60,  # Below fallback threshold of 0.70
            minutes_until_close=180.0,
            estimated_duration_minutes=45.0,
        )
        evaluator._claude.ask = AsyncMock(return_value={"error": "Timeout", "fallback": True})
        result = await evaluator.evaluate(low_conf_signal, sample_state)
        assert result.decision == "reject"

    @pytest.mark.asyncio
    async def test_fallback_rejects_insufficient_time(self, evaluator, sample_state):
        late_signal = TradeSignal(
            direction=Direction.LONG,
            entry_price=20100.0,
            stop_loss=20080.0,
            take_profit=20150.0,
            confidence=0.80,
            minutes_until_close=30.0,  # Not enough for 45min avg trade
            estimated_duration_minutes=45.0,
        )
        evaluator._claude.ask = AsyncMock(return_value={"error": "Timeout", "fallback": True})
        result = await evaluator.evaluate(late_signal, sample_state)
        assert result.decision == "reject"
        assert "time" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_fallback_rejects_consecutive_losses(self, evaluator, sample_signal):
        state = MarketState(
            symbol="MNQU5", current_price=20100.0,
            regime=MarketRegime.TRENDING_BULLISH,
            equity=10000.0, consecutive_losses=2,
        )
        evaluator._claude.ask = AsyncMock(return_value={"error": "Timeout", "fallback": True})
        result = await evaluator.evaluate(sample_signal, state)
        assert result.decision == "reject"
