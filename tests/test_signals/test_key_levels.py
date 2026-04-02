"""Tests for key levels engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.signals.key_levels import KeyLevelsEngine


@pytest.fixture
def daily_df():
    """30 days of daily OHLCV data."""
    dates = [datetime(2025, 5, 15, tzinfo=timezone.utc) + timedelta(days=i) for i in range(30)]
    base = 20000
    return pd.DataFrame({
        "timestamp": dates,
        "open": [base + i * 10 for i in range(30)],
        "high": [base + i * 10 + 50 for i in range(30)],
        "low": [base + i * 10 - 30 for i in range(30)],
        "close": [base + i * 10 + 20 for i in range(30)],
        "volume": [50000] * 30,
    })


class TestKeyLevelsEngine:
    def test_prev_day_levels(self, daily_df):
        engine = KeyLevelsEngine()
        levels = engine.update_from_daily(daily_df)

        # Previous day = second to last bar
        prev = daily_df.iloc[-2]
        assert levels.prev_day_high == prev["high"]
        assert levels.prev_day_low == prev["low"]
        assert levels.prev_day_close == prev["close"]
        assert levels.prev_day_midpoint == (prev["high"] + prev["low"]) / 2

    def test_prev_week_levels(self, daily_df):
        engine = KeyLevelsEngine()
        levels = engine.update_from_daily(daily_df)

        # Should have some weekly data
        assert levels.prev_week_high > 0
        assert levels.prev_week_low > 0
        assert levels.prev_week_high > levels.prev_week_low

    def test_quarterly_open(self, daily_df):
        engine = KeyLevelsEngine()
        levels = engine.update_from_daily(daily_df)
        # Should have a quarterly open
        assert levels.quarterly_open > 0

    def test_nearest_levels(self, daily_df):
        engine = KeyLevelsEngine()
        engine.update_from_daily(daily_df)

        nearest = engine.get_nearest_levels(20200.0, n=3)
        assert len(nearest) <= 3
        # Should be sorted by distance
        for i in range(1, len(nearest)):
            assert nearest[i][2] >= nearest[i - 1][2]

    def test_price_near_level(self, daily_df):
        engine = KeyLevelsEngine()
        engine.update_from_daily(daily_df)

        # Test with a price very close to prev day high
        near_high = daily_df.iloc[-2]["high"]
        result = engine.price_near_level(near_high + 1, tolerance_points=5.0)
        assert result is not None

    def test_too_few_bars(self):
        engine = KeyLevelsEngine()
        df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "open": [100], "high": [101], "low": [99], "close": [100],
        })
        levels = engine.update_from_daily(df)
        assert levels.prev_day_high == 0.0  # Not enough data

    def test_last_updated(self, daily_df):
        engine = KeyLevelsEngine()
        engine.update_from_daily(daily_df)
        assert engine.levels.last_updated is not None
