"""Shared test fixtures for the trading agent test suite."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Set test environment variables before importing config
os.environ.setdefault("TRADOVATE_USERNAME", "test_user")
os.environ.setdefault("TRADOVATE_PASSWORD", "test_pass")
os.environ.setdefault("TRADOVATE_APP_ID", "test_app")
os.environ.setdefault("TRADOVATE_CLIENT_ID", "test_client")
os.environ.setdefault("TRADOVATE_CLIENT_SECRET", "test_secret")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key")
os.environ.setdefault("TRADOVATE_ENVIRONMENT", "demo")


@pytest.fixture
def sample_candle_data():
    """Sample OHLCV data for testing."""
    from src.data.models import Candle

    return [
        Candle(
            timestamp=datetime(2025, 6, 15, 9, 30, tzinfo=timezone.utc),
            open=20100.0, high=20120.0, low=20090.0, close=20115.0,
            volume=1500, symbol="NQU5", timeframe="5m",
        ),
        Candle(
            timestamp=datetime(2025, 6, 15, 9, 35, tzinfo=timezone.utc),
            open=20115.0, high=20130.0, low=20110.0, close=20125.0,
            volume=1200, symbol="NQU5", timeframe="5m",
        ),
        Candle(
            timestamp=datetime(2025, 6, 15, 9, 40, tzinfo=timezone.utc),
            open=20125.0, high=20140.0, low=20100.0, close=20105.0,
            volume=1800, symbol="NQU5", timeframe="5m",
        ),
    ]


@pytest.fixture
def sample_quote():
    """Sample quote for testing."""
    from src.data.models import Quote

    return Quote(
        timestamp=datetime(2025, 6, 15, 10, 0, tzinfo=timezone.utc),
        symbol="NQU5",
        bid_price=20100.00,
        ask_price=20100.25,
        bid_size=50,
        ask_size=45,
        last_price=20100.25,
        last_size=5,
        total_volume=50000,
    )


@pytest.fixture
def tmp_parquet_dir(tmp_path):
    """Temporary directory for Parquet files."""
    return str(tmp_path / "candles")
