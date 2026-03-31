"""Tests for data models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.data.models import (
    Candle,
    DOMLevel,
    DOMSnapshot,
    Direction,
    ICTSignal,
    KeyLevels,
    MarketState,
    OrderAction,
    OrderFlowState,
    OrderType,
    PlaceOrderRequest,
    Position,
    Quote,
    TradeSignal,
)


class TestCandle:
    def test_bullish_candle(self):
        c = Candle(
            timestamp=datetime.now(timezone.utc),
            open=100.0, high=110.0, low=95.0, close=108.0,
        )
        assert c.is_bullish is True
        assert c.is_bearish is False
        assert c.body_size == 8.0
        assert c.upper_wick == 2.0
        assert c.lower_wick == 5.0
        assert c.midpoint == 102.5

    def test_bearish_candle(self):
        c = Candle(
            timestamp=datetime.now(timezone.utc),
            open=108.0, high=110.0, low=95.0, close=100.0,
        )
        assert c.is_bearish is True
        assert c.is_bullish is False
        assert c.body_size == 8.0
        assert c.upper_wick == 2.0
        assert c.lower_wick == 5.0


class TestQuote:
    def test_spread(self):
        q = Quote(
            timestamp=datetime.now(timezone.utc),
            symbol="NQU5",
            bid_price=20100.00,
            ask_price=20100.25,
        )
        assert q.spread == 0.25
        assert q.mid_price == 20100.125


class TestDOMLevel:
    def test_imbalance_ratio(self):
        level = DOMLevel(price=20100.0, bid_size=100, ask_size=25)
        ratio = level.imbalance_ratio
        assert ratio == 0.6  # (100 - 25) / 125

    def test_zero_volume(self):
        level = DOMLevel(price=20100.0, bid_size=0, ask_size=0)
        assert level.imbalance_ratio == 0.0


class TestDOMSnapshot:
    def test_total_volumes(self):
        dom = DOMSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="NQU5",
            levels=[
                DOMLevel(price=20100.0, bid_size=50, ask_size=30),
                DOMLevel(price=20100.25, bid_size=40, ask_size=60),
            ],
        )
        assert dom.total_bid_volume == 90
        assert dom.total_ask_volume == 90


class TestPosition:
    def test_long_position(self):
        p = Position(netPos=2, netPrice=20100.0)
        assert p.direction == Direction.LONG
        assert p.size == 2
        assert p.is_flat is False

    def test_short_position(self):
        p = Position(netPos=-3, netPrice=20100.0)
        assert p.direction == Direction.SHORT
        assert p.size == 3

    def test_flat_position(self):
        p = Position(netPos=0, netPrice=0.0)
        assert p.direction is None
        assert p.is_flat is True


class TestPlaceOrderRequest:
    def test_market_order(self):
        order = PlaceOrderRequest(
            action=OrderAction.BUY,
            symbol="NQU5",
            orderQty=2,
            orderType=OrderType.MARKET,
        )
        assert order.action == OrderAction.BUY
        assert order.order_qty == 2
        assert order.is_automated is True


class TestTradeSignal:
    def test_risk_reward(self):
        signal = TradeSignal(
            direction=Direction.LONG,
            entry_price=20100.0,
            stop_loss=20080.0,
            take_profit=20150.0,
            confidence=0.75,
        )
        assert signal.risk_points == 20.0
        assert signal.reward_points == 50.0
        assert signal.risk_reward_ratio == 2.5

    def test_time_adjusted_tp(self):
        signal = TradeSignal(
            direction=Direction.LONG,
            entry_price=20100.0,
            stop_loss=20080.0,
            take_profit=20150.0,
            time_adjusted_tp=20130.0,
            confidence=0.70,
        )
        # With time-adjusted TP, reward should use adjusted value
        assert signal.reward_points == 30.0
        assert signal.risk_reward_ratio == 1.5

    def test_zero_risk(self):
        signal = TradeSignal(
            direction=Direction.LONG,
            entry_price=20100.0,
            stop_loss=20100.0,  # SL at entry (shouldn't happen but handle gracefully)
            take_profit=20150.0,
            confidence=0.5,
        )
        assert signal.risk_reward_ratio == 0.0


class TestICTSignal:
    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            ICTSignal(
                pattern="fvg",
                direction=Direction.LONG,
                price_level=20100.0,
                confidence=1.5,  # Out of bounds
                timeframe="5m",
                timestamp=datetime.now(timezone.utc),
            )
