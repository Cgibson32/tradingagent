"""Historical data fetcher and storage.

Fetches historical OHLCV candle data via Tradovate Market Data WebSocket
and stores it as Parquet files partitioned by symbol and timeframe.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import structlog

from src.data.models import Candle

logger = structlog.get_logger(__name__)

# Tradovate chart description mappings
TIMEFRAME_TO_CHART_DESC = {
    "1m": {"underlyingType": "MinuteBar", "elementSize": 1, "elementSizeUnit": "UnderlyingUnits", "withHistogram": False},
    "5m": {"underlyingType": "MinuteBar", "elementSize": 5, "elementSizeUnit": "UnderlyingUnits", "withHistogram": False},
    "15m": {"underlyingType": "MinuteBar", "elementSize": 15, "elementSizeUnit": "UnderlyingUnits", "withHistogram": False},
    "30m": {"underlyingType": "MinuteBar", "elementSize": 30, "elementSizeUnit": "UnderlyingUnits", "withHistogram": False},
    "1h": {"underlyingType": "MinuteBar", "elementSize": 60, "elementSizeUnit": "UnderlyingUnits", "withHistogram": False},
    "4h": {"underlyingType": "MinuteBar", "elementSize": 240, "elementSizeUnit": "UnderlyingUnits", "withHistogram": False},
    "1d": {"underlyingType": "DailyBar", "elementSize": 1, "elementSizeUnit": "UnderlyingUnits", "withHistogram": False},
}


class HistoricalDataManager:
    """Manages fetching and storing historical candle data.

    Uses Tradovate's md/getchart endpoint via the Market Data WebSocket
    to fetch historical OHLCV data, then stores it as Parquet files
    for fast access during backtesting and indicator warm-up.
    """

    def __init__(self, parquet_dir: str = "data/candles"):
        self._parquet_dir = Path(parquet_dir)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)

    def _parquet_path(self, symbol: str, timeframe: str) -> Path:
        """Get the Parquet file path for a symbol/timeframe combo."""
        return self._parquet_dir / f"{symbol}_{timeframe}.parquet"

    async def fetch_candles(
        self,
        md_ws,
        symbol: str,
        timeframe: str,
        num_bars: int = 500,
    ) -> list[Candle]:
        """Fetch historical candles via WebSocket.

        Args:
            md_ws: MarketDataWebSocket instance (connected and authorized)
            symbol: Contract symbol (e.g., 'NQU5')
            timeframe: Candle timeframe (e.g., '5m', '30m', '1h')
            num_bars: Number of historical bars to fetch

        Returns:
            List of Candle objects, oldest first.
        """
        chart_desc = TIMEFRAME_TO_CHART_DESC.get(timeframe)
        if not chart_desc:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        candles: list[Candle] = []
        done_event = asyncio.Event()

        async def on_candle(candle: Candle):
            candle.timeframe = timeframe
            candle.symbol = symbol
            candles.append(candle)

        # Register callback and request chart
        req_id = await md_ws.get_chart(
            symbol=symbol,
            chart_description=chart_desc,
            time_range={"asMuchAsElements": num_bars},
        )
        md_ws.on_chart(req_id, on_candle)

        # Wait for data (with timeout)
        try:
            await asyncio.wait_for(self._wait_for_eoh(md_ws, req_id, done_event), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(
                "historical.fetch_timeout",
                symbol=symbol,
                timeframe=timeframe,
                bars_received=len(candles),
            )

        # Sort by timestamp
        candles.sort(key=lambda c: c.timestamp)

        logger.info(
            "historical.fetched",
            symbol=symbol,
            timeframe=timeframe,
            num_bars=len(candles),
        )

        return candles

    async def _wait_for_eoh(self, md_ws, req_id: int, done_event: asyncio.Event) -> None:
        """Wait for end-of-history marker from chart data."""
        # Poll until we stop receiving new bars (simple approach)
        last_count = 0
        stable_checks = 0
        while stable_checks < 3:
            await asyncio.sleep(1.0)
            current_count = len(md_ws._chart_callbacks.get(req_id, []))
            if current_count == last_count:
                stable_checks += 1
            else:
                stable_checks = 0
                last_count = current_count

    def save_candles(self, candles: list[Candle], symbol: str, timeframe: str) -> Path:
        """Save candles to a Parquet file.

        Merges with existing data if present, deduplicating by timestamp.
        """
        if not candles:
            logger.warning("historical.save_empty", symbol=symbol, timeframe=timeframe)
            return self._parquet_path(symbol, timeframe)

        # Convert to DataFrame
        new_df = pd.DataFrame([c.model_dump() for c in candles])
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True)

        path = self._parquet_path(symbol, timeframe)

        # Merge with existing if present
        if path.exists():
            existing_df = pd.read_parquet(path)
            existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], utc=True)
            combined = pd.concat([existing_df, new_df])
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
            combined = combined.sort_values("timestamp").reset_index(drop=True)
        else:
            combined = new_df.sort_values("timestamp").reset_index(drop=True)

        combined.to_parquet(path, index=False)
        logger.info(
            "historical.saved",
            symbol=symbol,
            timeframe=timeframe,
            total_bars=len(combined),
            path=str(path),
        )
        return path

    def load_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Load candles from Parquet file.

        Args:
            symbol: Contract symbol
            timeframe: Candle timeframe
            start: Optional start datetime filter
            end: Optional end datetime filter

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        path = self._parquet_path(symbol, timeframe)
        if not path.exists():
            logger.warning("historical.not_found", path=str(path))
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if start:
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            df = df[df["timestamp"] >= start]
        if end:
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            df = df[df["timestamp"] <= end]

        return df.sort_values("timestamp").reset_index(drop=True)

    def has_sufficient_data(
        self, symbol: str, timeframe: str, min_bars: int = 200
    ) -> bool:
        """Check if we have enough historical data for indicator warm-up."""
        path = self._parquet_path(symbol, timeframe)
        if not path.exists():
            return False
        df = pd.read_parquet(path)
        return len(df) >= min_bars

    async def ensure_warm_data(
        self,
        md_ws,
        symbol: str,
        timeframes: list[str],
        min_bars: int = 200,
    ) -> None:
        """Ensure we have enough data for all timeframes (indicator warm-up).

        Fetches missing data if needed.
        """
        for tf in timeframes:
            if not self.has_sufficient_data(symbol, tf, min_bars):
                logger.info(
                    "historical.warming_up",
                    symbol=symbol,
                    timeframe=tf,
                    min_bars=min_bars,
                )
                candles = await self.fetch_candles(md_ws, symbol, tf, num_bars=min_bars)
                self.save_candles(candles, symbol, tf)
