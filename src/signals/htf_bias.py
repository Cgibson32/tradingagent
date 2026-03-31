"""Higher Timeframe (HTF) bias engine.

Analyzes 30m/1H chart structure to determine directional bias.
5m entries are only allowed in the direction of the HTF bias.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

import numpy as np
import pandas as pd

from src.signals.indicators import ema, swing_highs, swing_lows


class Bias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class HTFBiasEngine:
    """Determines directional bias from higher timeframe structure.

    Bias determined by:
    1. Swing structure: HH/HL = bullish, LH/LL = bearish
    2. EMA alignment: 9 > 21 > 50 = bullish, reverse = bearish
    3. Displacement direction: last displacement candle direction

    5m entries only allowed in HTF direction.
    Staleness timeout: re-evaluate every N candles or on structure break.
    """

    def __init__(self, staleness_bars: int = 6):
        self._bias = Bias.NEUTRAL
        self._confidence: float = 0.0
        self._last_update: datetime | None = None
        self._staleness_bars = staleness_bars
        self._bars_since_update = 0

    @property
    def bias(self) -> Bias:
        return self._bias

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def is_stale(self) -> bool:
        return self._bars_since_update >= self._staleness_bars

    def tick(self) -> None:
        """Increment bar counter. Call on each new candle close."""
        self._bars_since_update += 1

    def evaluate(self, df: pd.DataFrame) -> tuple[Bias, float]:
        """Evaluate HTF bias from a higher timeframe DataFrame (30m or 1H).

        Args:
            df: DataFrame with OHLCV data from the higher timeframe.
                Needs at least 50 bars for reliable assessment.

        Returns:
            Tuple of (Bias, confidence 0.0-1.0)
        """
        if len(df) < 50:
            return Bias.NEUTRAL, 0.0

        signals: list[tuple[Bias, float]] = []

        # 1. Swing structure analysis
        structure_bias, structure_conf = self._analyze_structure(df)
        signals.append((structure_bias, structure_conf * 0.45))

        # 2. EMA alignment
        ema_bias, ema_conf = self._analyze_ema_alignment(df)
        signals.append((ema_bias, ema_conf * 0.35))

        # 3. Last displacement direction
        disp_bias, disp_conf = self._analyze_displacement(df)
        signals.append((disp_bias, disp_conf * 0.20))

        # Combine signals
        bullish_score = sum(w for b, w in signals if b == Bias.BULLISH)
        bearish_score = sum(w for b, w in signals if b == Bias.BEARISH)

        if bullish_score > bearish_score and bullish_score > 0.3:
            self._bias = Bias.BULLISH
            self._confidence = min(bullish_score, 1.0)
        elif bearish_score > bullish_score and bearish_score > 0.3:
            self._bias = Bias.BEARISH
            self._confidence = min(bearish_score, 1.0)
        else:
            self._bias = Bias.NEUTRAL
            self._confidence = 0.0

        self._last_update = datetime.now(timezone.utc)
        self._bars_since_update = 0

        return self._bias, self._confidence

    def _analyze_structure(self, df: pd.DataFrame) -> tuple[Bias, float]:
        """Analyze swing structure for HH/HL or LH/LL pattern."""
        sh = swing_highs(df, lookback=3)
        sl = swing_lows(df, lookback=3)

        # Get last 4 swings
        recent_highs = sh.dropna().tail(4).values
        recent_lows = sl.dropna().tail(4).values

        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return Bias.NEUTRAL, 0.0

        # Check for higher highs and higher lows (bullish)
        hh = all(recent_highs[i] > recent_highs[i - 1] for i in range(1, len(recent_highs)))
        hl = all(recent_lows[i] > recent_lows[i - 1] for i in range(1, len(recent_lows)))

        # Check for lower highs and lower lows (bearish)
        lh = all(recent_highs[i] < recent_highs[i - 1] for i in range(1, len(recent_highs)))
        ll = all(recent_lows[i] < recent_lows[i - 1] for i in range(1, len(recent_lows)))

        if hh and hl:
            return Bias.BULLISH, 0.9
        if lh and ll:
            return Bias.BEARISH, 0.9
        if hh:
            return Bias.BULLISH, 0.6
        if ll:
            return Bias.BEARISH, 0.6

        return Bias.NEUTRAL, 0.0

    def _analyze_ema_alignment(self, df: pd.DataFrame) -> tuple[Bias, float]:
        """Check if EMAs are stacked in order."""
        ema_9 = ema(df["close"], 9).iloc[-1]
        ema_21 = ema(df["close"], 21).iloc[-1]
        ema_50 = ema(df["close"], 50).iloc[-1]

        if any(np.isnan(v) for v in [ema_9, ema_21, ema_50]):
            return Bias.NEUTRAL, 0.0

        if ema_9 > ema_21 > ema_50:
            return Bias.BULLISH, 0.8
        if ema_9 < ema_21 < ema_50:
            return Bias.BEARISH, 0.8

        # Partial alignment
        if ema_9 > ema_21:
            return Bias.BULLISH, 0.4
        if ema_9 < ema_21:
            return Bias.BEARISH, 0.4

        return Bias.NEUTRAL, 0.0

    def _analyze_displacement(self, df: pd.DataFrame) -> tuple[Bias, float]:
        """Check the direction of the most recent displacement candle."""
        from src.signals.indicators import atr as calc_atr

        if len(df) < 15:
            return Bias.NEUTRAL, 0.0

        atr_vals = calc_atr(df, 14)

        # Look at last 10 bars for a displacement
        for i in range(len(df) - 1, max(len(df) - 11, 0), -1):
            body = abs(df["close"].iloc[i] - df["open"].iloc[i])
            current_atr = atr_vals.iloc[i]

            if current_atr == 0 or np.isnan(current_atr):
                continue

            if body > current_atr * 1.5:
                if df["close"].iloc[i] > df["open"].iloc[i]:
                    return Bias.BULLISH, 0.7
                else:
                    return Bias.BEARISH, 0.7

        return Bias.NEUTRAL, 0.0

    def allows_entry(self, direction: str) -> bool:
        """Check if a proposed entry direction aligns with HTF bias.

        Args:
            direction: 'long' or 'short'

        Returns:
            True if the entry direction matches the bias or bias is neutral.
        """
        if self._bias == Bias.NEUTRAL:
            return False  # No trading without clear bias
        if direction == "long" and self._bias == Bias.BULLISH:
            return True
        if direction == "short" and self._bias == Bias.BEARISH:
            return True
        return False
