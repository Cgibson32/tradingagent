"""Key ICT reference levels engine.

Auto-calculates and maintains critical institutional reference levels:
- Previous day high/low/close/midpoint
- Previous week high/low/open
- Previous month high/low/open
- Quarterly opens
- HTF Fair Value Gaps (weekly/monthly)
"""

from __future__ import annotations

from datetime import datetime, date, timedelta, timezone

import pandas as pd

from src.data.models import KeyLevels


class KeyLevelsEngine:
    """Calculates and maintains critical ICT reference levels.

    These levels act as magnets, support/resistance, and liquidity targets.
    They are fed to both the signal aggregator and Claude for context.
    """

    def __init__(self):
        self._levels = KeyLevels()

    @property
    def levels(self) -> KeyLevels:
        return self._levels

    def update_from_daily(self, daily_df: pd.DataFrame) -> KeyLevels:
        """Update key levels from daily OHLCV data.

        Args:
            daily_df: DataFrame with daily candles, sorted by timestamp ascending.
                      Needs at least 30 bars for monthly levels.

        Returns:
            Updated KeyLevels.
        """
        if len(daily_df) < 2:
            return self._levels

        df = daily_df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp")
        else:
            df = df.sort_index()

        # Previous day levels (most recent completed day)
        prev_day = df.iloc[-2]
        self._levels.prev_day_high = float(prev_day["high"])
        self._levels.prev_day_low = float(prev_day["low"])
        self._levels.prev_day_close = float(prev_day["close"])
        self._levels.prev_day_midpoint = (self._levels.prev_day_high + self._levels.prev_day_low) / 2

        # Previous week levels
        self._update_weekly_levels(df)

        # Previous month levels
        self._update_monthly_levels(df)

        # Quarterly open
        self._update_quarterly_open(df)

        self._levels.last_updated = datetime.now(timezone.utc)
        return self._levels

    def _update_weekly_levels(self, df: pd.DataFrame) -> None:
        """Calculate previous week high/low/open."""
        if "timestamp" not in df.columns:
            return

        df = df.copy()
        df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
        df["year"] = df["timestamp"].dt.year

        # Group by year-week
        df["year_week"] = df["year"].astype(str) + "-" + df["week"].astype(str).str.zfill(2)
        weeks = df["year_week"].unique()

        if len(weeks) < 2:
            return

        # Previous week = second to last week
        prev_week_data = df[df["year_week"] == weeks[-2]]
        if len(prev_week_data) > 0:
            self._levels.prev_week_high = float(prev_week_data["high"].max())
            self._levels.prev_week_low = float(prev_week_data["low"].min())
            self._levels.prev_week_open = float(prev_week_data.iloc[0]["open"])

    def _update_monthly_levels(self, df: pd.DataFrame) -> None:
        """Calculate previous month high/low/open."""
        if "timestamp" not in df.columns:
            return

        df = df.copy()
        df["month"] = df["timestamp"].dt.to_period("M")
        months = df["month"].unique()

        if len(months) < 2:
            return

        prev_month_data = df[df["month"] == months[-2]]
        if len(prev_month_data) > 0:
            self._levels.prev_month_high = float(prev_month_data["high"].max())
            self._levels.prev_month_low = float(prev_month_data["low"].min())
            self._levels.prev_month_open = float(prev_month_data.iloc[0]["open"])

    def _update_quarterly_open(self, df: pd.DataFrame) -> None:
        """Calculate the current quarter's opening price.

        Quarters: Q1 (Jan 1), Q2 (Apr 1), Q3 (Jul 1), Q4 (Oct 1).
        """
        if "timestamp" not in df.columns:
            return

        df = df.copy()
        df["quarter"] = df["timestamp"].dt.quarter
        df["year"] = df["timestamp"].dt.year
        df["year_quarter"] = df["year"].astype(str) + "-Q" + df["quarter"].astype(str)

        # Current quarter
        current_yq = df["year_quarter"].iloc[-1]
        current_q_data = df[df["year_quarter"] == current_yq]

        if len(current_q_data) > 0:
            self._levels.quarterly_open = float(current_q_data.iloc[0]["open"])

    def get_nearest_levels(self, price: float, n: int = 5) -> list[tuple[str, float, float]]:
        """Get the N nearest key levels to the current price.

        Returns:
            List of (level_name, price_level, distance) tuples, sorted by distance.
        """
        all_levels = [
            ("prev_day_high", self._levels.prev_day_high),
            ("prev_day_low", self._levels.prev_day_low),
            ("prev_day_close", self._levels.prev_day_close),
            ("prev_day_midpoint", self._levels.prev_day_midpoint),
            ("prev_week_high", self._levels.prev_week_high),
            ("prev_week_low", self._levels.prev_week_low),
            ("prev_week_open", self._levels.prev_week_open),
            ("prev_month_high", self._levels.prev_month_high),
            ("prev_month_low", self._levels.prev_month_low),
            ("prev_month_open", self._levels.prev_month_open),
            ("quarterly_open", self._levels.quarterly_open),
        ]

        # Filter out zero/unset levels
        valid = [(name, lvl) for name, lvl in all_levels if lvl > 0]

        # Sort by distance to current price
        with_dist = [(name, lvl, abs(price - lvl)) for name, lvl in valid]
        with_dist.sort(key=lambda x: x[2])

        return with_dist[:n]

    def price_near_level(self, price: float, tolerance_points: float = 5.0) -> str | None:
        """Check if price is near any key level.

        Args:
            price: Current price
            tolerance_points: How close price needs to be (in points)

        Returns:
            Name of the nearest level if within tolerance, None otherwise.
        """
        nearest = self.get_nearest_levels(price, n=1)
        if nearest and nearest[0][2] <= tolerance_points:
            return nearest[0][0]
        return None
