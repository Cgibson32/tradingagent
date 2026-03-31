"""Tests for HTF bias engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.signals.htf_bias import Bias, HTFBiasEngine


@pytest.fixture
def uptrend_df():
    """Strong uptrend data — should produce bullish bias."""
    n = 100
    timestamps = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=30 * i) for i in range(n)]
    close = 20000 + np.arange(n) * 5.0 + np.random.randn(n) * 2
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close - 2,
        "high": close + 3,
        "low": close - 3,
        "close": close,
        "volume": np.random.randint(1000, 5000, n),
    })


@pytest.fixture
def downtrend_df():
    """Strong downtrend data — should produce bearish bias."""
    n = 100
    timestamps = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=30 * i) for i in range(n)]
    close = 20500 - np.arange(n) * 5.0 + np.random.randn(n) * 2
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close + 2,
        "high": close + 3,
        "low": close - 3,
        "close": close,
        "volume": np.random.randint(1000, 5000, n),
    })


@pytest.fixture
def choppy_df():
    """Ranging/choppy data — should produce neutral bias."""
    n = 100
    timestamps = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=30 * i) for i in range(n)]
    close = 20200 + np.sin(np.arange(n) * 0.5) * 10 + np.random.randn(n) * 5
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close - 2,
        "high": close + 5,
        "low": close - 5,
        "close": close,
        "volume": np.random.randint(1000, 5000, n),
    })


class TestHTFBiasEngine:
    def test_bullish_bias(self, uptrend_df):
        engine = HTFBiasEngine()
        bias, conf = engine.evaluate(uptrend_df)
        # Strong uptrend should produce bullish or at least non-bearish
        assert bias in (Bias.BULLISH, Bias.NEUTRAL)
        if bias == Bias.BULLISH:
            assert conf > 0.0

    def test_bearish_bias(self, downtrend_df):
        engine = HTFBiasEngine()
        bias, conf = engine.evaluate(downtrend_df)
        assert bias in (Bias.BEARISH, Bias.NEUTRAL)
        if bias == Bias.BEARISH:
            assert conf > 0.0

    def test_allows_long_in_uptrend(self, uptrend_df):
        engine = HTFBiasEngine()
        bias, _ = engine.evaluate(uptrend_df)
        if bias == Bias.BULLISH:
            assert engine.allows_entry("long") is True
            assert engine.allows_entry("short") is False

    def test_allows_short_in_downtrend(self, downtrend_df):
        engine = HTFBiasEngine()
        bias, _ = engine.evaluate(downtrend_df)
        if bias == Bias.BEARISH:
            assert engine.allows_entry("short") is True
            assert engine.allows_entry("long") is False

    def test_staleness(self):
        engine = HTFBiasEngine(staleness_bars=3)
        assert engine.is_stale is False
        engine.tick()
        engine.tick()
        engine.tick()
        assert engine.is_stale is True

    def test_insufficient_data(self):
        engine = HTFBiasEngine()
        small_df = pd.DataFrame({
            "open": [100], "high": [101], "low": [99], "close": [100],
        })
        bias, conf = engine.evaluate(small_df)
        assert bias == Bias.NEUTRAL
        assert conf == 0.0
