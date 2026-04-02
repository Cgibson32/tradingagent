"""Tests for historical data management."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data.historical import HistoricalDataManager, TIMEFRAME_TO_CHART_DESC
from src.data.models import Candle


class TestHistoricalDataManager:
    def test_timeframe_mappings(self):
        assert "5m" in TIMEFRAME_TO_CHART_DESC
        assert "30m" in TIMEFRAME_TO_CHART_DESC
        assert "1h" in TIMEFRAME_TO_CHART_DESC
        assert "1d" in TIMEFRAME_TO_CHART_DESC

        desc_5m = TIMEFRAME_TO_CHART_DESC["5m"]
        assert desc_5m["underlyingType"] == "MinuteBar"
        assert desc_5m["elementSize"] == 5

    def test_save_and_load(self, tmp_parquet_dir, sample_candle_data):
        mgr = HistoricalDataManager(tmp_parquet_dir)

        # Save
        path = mgr.save_candles(sample_candle_data, "NQU5", "5m")
        assert path.exists()

        # Load
        df = mgr.load_candles("NQU5", "5m")
        assert len(df) == 3
        assert "open" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_deduplication_on_save(self, tmp_parquet_dir, sample_candle_data):
        mgr = HistoricalDataManager(tmp_parquet_dir)

        # Save twice — should deduplicate
        mgr.save_candles(sample_candle_data, "NQU5", "5m")
        mgr.save_candles(sample_candle_data, "NQU5", "5m")

        df = mgr.load_candles("NQU5", "5m")
        assert len(df) == 3  # No duplicates

    def test_has_sufficient_data(self, tmp_parquet_dir, sample_candle_data):
        mgr = HistoricalDataManager(tmp_parquet_dir)

        assert mgr.has_sufficient_data("NQU5", "5m", min_bars=3) is False

        mgr.save_candles(sample_candle_data, "NQU5", "5m")
        assert mgr.has_sufficient_data("NQU5", "5m", min_bars=3) is True
        assert mgr.has_sufficient_data("NQU5", "5m", min_bars=5) is False

    def test_load_with_date_filter(self, tmp_parquet_dir, sample_candle_data):
        mgr = HistoricalDataManager(tmp_parquet_dir)
        mgr.save_candles(sample_candle_data, "NQU5", "5m")

        # Filter by start date
        start = datetime(2025, 6, 15, 9, 35, tzinfo=timezone.utc)
        df = mgr.load_candles("NQU5", "5m", start=start)
        assert len(df) == 2  # 9:35 and 9:40 candles

    def test_load_nonexistent(self, tmp_parquet_dir):
        mgr = HistoricalDataManager(tmp_parquet_dir)
        df = mgr.load_candles("FAKE", "5m")
        assert len(df) == 0
        assert "timestamp" in df.columns

    def test_save_empty_candles(self, tmp_parquet_dir):
        mgr = HistoricalDataManager(tmp_parquet_dir)
        path = mgr.save_candles([], "NQU5", "5m")
        # Should not crash, just return path
        assert path is not None
