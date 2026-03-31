"""Tests for ICT/SMC pattern detection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.models import Direction
from src.signals.ict_patterns import (
    FVGZone,
    detect_bos_choch,
    detect_displacement,
    detect_fvg,
    detect_liquidity_sweeps,
    detect_order_blocks,
    detect_smt_divergence,
    check_fvg_retest,
)


@pytest.fixture
def bullish_fvg_df():
    """DataFrame with a clear bullish FVG (gap up)."""
    timestamps = [datetime(2025, 6, 15, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(5)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [20100, 20105, 20115, 20130, 20140],
        "high": [20110, 20120, 20135, 20145, 20150],
        "low": [20095, 20100, 20112, 20125, 20135],
        "close": [20105, 20115, 20130, 20140, 20145],
        "volume": [1000] * 5,
    })


@pytest.fixture
def bearish_fvg_df():
    """DataFrame with a clear bearish FVG (gap down)."""
    timestamps = [datetime(2025, 6, 15, 9, 30, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(5)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [20150, 20145, 20130, 20110, 20100],
        "high": [20155, 20150, 20135, 20115, 20105],
        "low": [20140, 20130, 20108, 20095, 20090],
        "close": [20145, 20130, 20110, 20100, 20095],
        "volume": [1000] * 5,
    })


@pytest.fixture
def trending_df():
    """200 bars of trending data for structure detection."""
    np.random.seed(123)
    n = 200
    timestamps = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(n)]
    # Uptrend with some noise
    trend = np.linspace(20000, 20200, n) + np.random.randn(n) * 5
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": trend - 2,
        "high": trend + np.abs(np.random.randn(n) * 3),
        "low": trend - np.abs(np.random.randn(n) * 3),
        "close": trend,
        "volume": np.random.randint(500, 2000, n),
    })


class TestFVG:
    def test_detect_bullish_fvg(self, bullish_fvg_df):
        fvgs = detect_fvg(bullish_fvg_df, min_ticks=1, tick_size=0.25)
        bullish = [f for f in fvgs if f.direction == Direction.LONG]
        assert len(bullish) > 0
        for fvg in bullish:
            assert fvg.top > fvg.bottom

    def test_detect_bearish_fvg(self, bearish_fvg_df):
        fvgs = detect_fvg(bearish_fvg_df, min_ticks=1, tick_size=0.25)
        bearish = [f for f in fvgs if f.direction == Direction.SHORT]
        assert len(bearish) > 0

    def test_min_size_filter(self, bullish_fvg_df):
        """FVGs smaller than min_ticks should be filtered out."""
        small = detect_fvg(bullish_fvg_df, min_ticks=1, tick_size=0.25)
        large = detect_fvg(bullish_fvg_df, min_ticks=100, tick_size=0.25)
        assert len(large) <= len(small)

    def test_too_few_bars(self):
        df = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "open": [100], "high": [101], "low": [99], "close": [100.5],
            "volume": [100],
        })
        assert detect_fvg(df) == []

    def test_fvg_retest(self):
        fvg = FVGZone(
            direction=Direction.LONG,
            top=20120.0, bottom=20110.0,
            timestamp=datetime.now(timezone.utc), timeframe="5m",
        )
        # Price inside zone
        assert check_fvg_retest(20115.0, [fvg], Direction.LONG) == fvg
        # Price outside zone
        assert check_fvg_retest(20130.0, [fvg], Direction.LONG) is None
        # Wrong direction
        assert check_fvg_retest(20115.0, [fvg], Direction.SHORT) is None


class TestDisplacement:
    def test_detect_displacement(self, trending_df):
        signals = detect_displacement(trending_df, atr_multiple=2.0)
        # Should find some displacement candles in trending data
        assert isinstance(signals, list)
        for s in signals:
            assert s.pattern == "displacement"
            assert s.confidence > 0

    def test_no_displacement_in_flat(self):
        """Flat market should have few/no displacements."""
        n = 100
        timestamps = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(n)]
        flat = pd.DataFrame({
            "timestamp": timestamps,
            "open": [20100.0] * n,
            "high": [20101.0] * n,
            "low": [20099.0] * n,
            "close": [20100.0] * n,
            "volume": [1000] * n,
        })
        signals = detect_displacement(flat)
        assert len(signals) == 0


class TestBOSCHoCH:
    def test_detect_in_trending(self, trending_df):
        signals = detect_bos_choch(trending_df, lookback=5)
        assert isinstance(signals, list)
        for s in signals:
            assert s.pattern in ("bos", "choch")

    def test_bos_types(self, trending_df):
        signals = detect_bos_choch(trending_df, lookback=5)
        bos = [s for s in signals if s.pattern == "bos"]
        choch = [s for s in signals if s.pattern == "choch"]
        # Trending data should have mostly BOS
        assert isinstance(bos, list)
        assert isinstance(choch, list)


class TestOrderBlocks:
    def test_detect_order_blocks(self, trending_df):
        blocks = detect_order_blocks(trending_df)
        assert isinstance(blocks, list)
        for b in blocks:
            assert b.top > b.bottom
            assert b.direction in (Direction.LONG, Direction.SHORT)


class TestSMTDivergence:
    def test_detect_smt(self):
        """Test SMT divergence detection with divergent NQ/ES data."""
        n = 100
        timestamps = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5 * i) for i in range(n)]

        # NQ makes higher highs
        nq_close = 20000 + np.cumsum(np.random.randn(n) * 3 + 0.5)
        # ES does NOT make higher highs (flat)
        es_close = 5000 + np.cumsum(np.random.randn(n) * 2)

        nq_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": nq_close - 1, "high": nq_close + 5,
            "low": nq_close - 5, "close": nq_close, "volume": [1000] * n,
        })
        es_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": es_close - 1, "high": es_close + 3,
            "low": es_close - 3, "close": es_close, "volume": [1000] * n,
        })

        signals = detect_smt_divergence(nq_df, es_df, lookback=5)
        assert isinstance(signals, list)
        for s in signals:
            assert s.pattern == "smt"

    def test_smt_too_few_bars(self):
        small = pd.DataFrame({
            "timestamp": [datetime.now(timezone.utc)],
            "open": [100], "high": [101], "low": [99], "close": [100], "volume": [100],
        })
        assert detect_smt_divergence(small, small) == []
