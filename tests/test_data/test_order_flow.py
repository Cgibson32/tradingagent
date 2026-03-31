"""Tests for order flow analysis."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.data.models import DOMLevel, DOMSnapshot, Direction, Quote
from src.data.order_flow import OrderFlowAnalyzer


@pytest.fixture
def analyzer():
    return OrderFlowAnalyzer()


@pytest.fixture
def bullish_dom():
    """DOM with stacked bid imbalances (bullish)."""
    return DOMSnapshot(
        timestamp=datetime.now(timezone.utc),
        symbol="NQU5",
        levels=[
            DOMLevel(price=20103.0, bid_size=10, ask_size=50),
            DOMLevel(price=20102.75, bid_size=10, ask_size=40),
            DOMLevel(price=20102.50, bid_size=150, ask_size=30),  # bid imbalance
            DOMLevel(price=20102.25, bid_size=120, ask_size=25),  # bid imbalance
            DOMLevel(price=20102.00, bid_size=200, ask_size=40),  # bid imbalance
            DOMLevel(price=20101.75, bid_size=180, ask_size=35),  # bid imbalance
            DOMLevel(price=20101.50, bid_size=50, ask_size=60),
        ],
    )


@pytest.fixture
def neutral_dom():
    """DOM with balanced bid/ask."""
    return DOMSnapshot(
        timestamp=datetime.now(timezone.utc),
        symbol="NQU5",
        levels=[
            DOMLevel(price=20102.0, bid_size=50, ask_size=45),
            DOMLevel(price=20101.75, bid_size=40, ask_size=42),
            DOMLevel(price=20101.50, bid_size=55, ask_size=50),
        ],
    )


class TestOrderFlowAnalyzer:
    def test_initial_state(self, analyzer):
        state = analyzer.state
        assert state.cumulative_delta == 0.0
        assert state.delta_trend == "neutral"
        assert state.absorption_detected is False
        assert state.delta_divergence is False

    def test_process_buy_tick(self, analyzer):
        """Trade at ask price = aggressive buy → positive delta."""
        quote = Quote(
            timestamp=datetime.now(timezone.utc),
            symbol="NQU5",
            bid_price=20100.0,
            ask_price=20100.25,
            last_price=20100.25,  # At ask = buy
            last_size=10,
        )
        analyzer.process_quote(quote)
        assert analyzer._cumulative_delta == 10

    def test_process_sell_tick(self, analyzer):
        """Trade at bid price = aggressive sell → negative delta."""
        quote = Quote(
            timestamp=datetime.now(timezone.utc),
            symbol="NQU5",
            bid_price=20100.0,
            ask_price=20100.25,
            last_price=20100.0,  # At bid = sell
            last_size=15,
        )
        analyzer.process_quote(quote)
        assert analyzer._cumulative_delta == -15

    def test_cumulative_delta(self, analyzer):
        """Delta should accumulate across ticks."""
        now = datetime.now(timezone.utc)

        # Buy 10
        analyzer.process_quote(Quote(
            timestamp=now, symbol="NQU5",
            bid_price=20100.0, ask_price=20100.25,
            last_price=20100.25, last_size=10,
        ))
        # Sell 5
        analyzer.process_quote(Quote(
            timestamp=now, symbol="NQU5",
            bid_price=20100.0, ask_price=20100.25,
            last_price=20100.0, last_size=5,
        ))
        assert analyzer._cumulative_delta == 5  # 10 - 5

    def test_session_reset(self, analyzer):
        now = datetime.now(timezone.utc)
        analyzer.process_quote(Quote(
            timestamp=now, symbol="NQU5",
            bid_price=20100.0, ask_price=20100.25,
            last_price=20100.25, last_size=100,
        ))
        assert analyzer._cumulative_delta == 100

        analyzer.reset_session()
        assert analyzer._cumulative_delta == 0.0

    def test_stacked_bid_imbalances(self, analyzer, bullish_dom):
        """Should detect stacked bid imbalances as bullish."""
        state = analyzer.process_dom(bullish_dom)
        assert len(state.stacked_imbalances) >= 3
        assert state.imbalance_direction == Direction.LONG

    def test_no_imbalances_on_balanced_dom(self, analyzer, neutral_dom):
        """Balanced DOM should not show imbalances."""
        state = analyzer.process_dom(neutral_dom)
        assert len(state.stacked_imbalances) == 0
        assert state.imbalance_direction is None

    def test_zero_size_tick_ignored(self, analyzer):
        """Ticks with zero size should not affect delta."""
        quote = Quote(
            timestamp=datetime.now(timezone.utc),
            symbol="NQU5",
            bid_price=20100.0,
            ask_price=20100.25,
            last_price=20100.25,
            last_size=0,
        )
        analyzer.process_quote(quote)
        assert analyzer._cumulative_delta == 0.0
