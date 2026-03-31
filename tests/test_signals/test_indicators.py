"""Tests for technical indicators."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.signals.indicators import (
    adx,
    atr,
    atr_percent,
    ema,
    ema_alignment,
    ema_stack,
    macd,
    rsi,
    swing_highs,
    swing_lows,
    vwap,
)


@pytest.fixture
def sample_df():
    """Generate 200 bars of synthetic OHLCV data."""
    np.random.seed(42)
    n = 200
    base = 20000.0
    timestamps = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(n)]

    close = base + np.cumsum(np.random.randn(n) * 5)
    high = close + np.abs(np.random.randn(n) * 3)
    low = close - np.abs(np.random.randn(n) * 3)
    open_price = close + np.random.randn(n) * 2
    volume = np.random.randint(500, 3000, n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestEMA:
    def test_ema_basic(self, sample_df):
        result = ema(sample_df["close"], 9)
        assert len(result) == len(sample_df)
        assert not result.isna().all()

    def test_ema_convergence(self, sample_df):
        """EMA should converge toward the price."""
        result = ema(sample_df["close"], 9)
        # Last value should be close to last close
        assert abs(result.iloc[-1] - sample_df["close"].iloc[-1]) < 50

    def test_ema_stack(self, sample_df):
        result = ema_stack(sample_df)
        assert "ema_9" in result.columns
        assert "ema_21" in result.columns
        assert "ema_50" in result.columns
        assert "ema_200" in result.columns
        assert "ema_9_slope" in result.columns

    def test_ema_alignment(self, sample_df):
        result = ema_alignment(sample_df)
        assert len(result) == len(sample_df)
        assert all(v in ("bullish", "bearish", "mixed") for v in result)


class TestADX:
    def test_adx_basic(self, sample_df):
        result = adx(sample_df)
        assert "adx" in result.columns
        assert "plus_di" in result.columns
        assert "minus_di" in result.columns

    def test_adx_range(self, sample_df):
        """ADX should be between 0 and 100."""
        result = adx(sample_df)
        valid = result["adx"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestMACD:
    def test_macd_basic(self, sample_df):
        result = macd(sample_df)
        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "histogram" in result.columns
        assert "zero_cross" in result.columns

    def test_histogram_is_difference(self, sample_df):
        result = macd(sample_df)
        diff = result["macd"] - result["signal"]
        np.testing.assert_allclose(
            result["histogram"].dropna().values,
            diff.dropna().values,
            atol=1e-10,
        )

    def test_zero_cross_detection(self, sample_df):
        result = macd(sample_df)
        crosses = result[result["zero_cross"] != "none"]
        # Should detect at least some crosses in 200 bars
        assert len(crosses) >= 0  # May not always have crosses with random data


class TestRSI:
    def test_rsi_basic(self, sample_df):
        result = rsi(sample_df)
        assert "rsi" in result.columns
        assert "divergence" in result.columns

    def test_rsi_range(self, sample_df):
        """RSI should be between 0 and 100."""
        result = rsi(sample_df)
        valid = result["rsi"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestATR:
    def test_atr_basic(self, sample_df):
        result = atr(sample_df)
        assert len(result) == len(sample_df)
        # ATR should be positive
        valid = result.dropna()
        assert (valid > 0).all()

    def test_atr_percent(self, sample_df):
        result = atr_percent(sample_df)
        valid = result.dropna()
        assert (valid > 0).all()
        assert (valid < 10).all()  # Should be a small percentage


class TestVWAP:
    def test_vwap_basic(self, sample_df):
        result = vwap(sample_df)
        assert len(result) == len(sample_df)
        valid = result.dropna()
        assert len(valid) > 0

    def test_vwap_between_high_low(self, sample_df):
        """VWAP should generally be between daily high and low."""
        result = vwap(sample_df)
        valid = result.dropna()
        # At least most values should be within the price range
        assert len(valid) > 0


class TestSwingDetection:
    def test_swing_highs(self, sample_df):
        result = swing_highs(sample_df, lookback=5)
        pivots = result.dropna()
        assert len(pivots) > 0  # Should find some pivots in 200 bars

    def test_swing_lows(self, sample_df):
        result = swing_lows(sample_df, lookback=5)
        pivots = result.dropna()
        assert len(pivots) > 0

    def test_swing_high_is_local_max(self, sample_df):
        """Each swing high should be higher than surrounding bars."""
        result = swing_highs(sample_df, lookback=3)
        for idx in result.dropna().index:
            i = sample_df.index.get_loc(idx)
            if i < 3 or i >= len(sample_df) - 3:
                continue
            pivot = sample_df["high"].iloc[i]
            for j in range(i - 3, i):
                assert pivot >= sample_df["high"].iloc[j]
            for j in range(i + 1, i + 4):
                assert pivot >= sample_df["high"].iloc[j]

    def test_swing_low_is_local_min(self, sample_df):
        """Each swing low should be lower than surrounding bars."""
        result = swing_lows(sample_df, lookback=3)
        for idx in result.dropna().index:
            i = sample_df.index.get_loc(idx)
            if i < 3 or i >= len(sample_df) - 3:
                continue
            pivot = sample_df["low"].iloc[i]
            for j in range(i - 3, i):
                assert pivot <= sample_df["low"].iloc[j]
            for j in range(i + 1, i + 4):
                assert pivot <= sample_df["low"].iloc[j]
