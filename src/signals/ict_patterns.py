"""ICT/SMC pattern detection — the core trading edge.

Detects institutional price action patterns:
- Liquidity sweeps (stop hunts)
- Displacement (institutional momentum)
- Fair Value Gaps (FVG)
- Break of Structure (BOS) / Change of Character (CHoCH)
- Order blocks (institutional supply/demand zones)
- SMT Divergence (NQ vs ES comparison)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.data.models import Direction, ICTSignal
from src.signals.indicators import atr, swing_highs, swing_lows


# ── Fair Value Gap (FVG) ──────────────────────────────


@dataclass
class FVGZone:
    """Tracks an active Fair Value Gap."""

    direction: Direction
    top: float
    bottom: float
    timestamp: datetime
    timeframe: str
    age_bars: int = 0
    filled: bool = False


def detect_fvg(
    df: pd.DataFrame,
    min_ticks: int = 4,
    tick_size: float = 0.25,
) -> list[FVGZone]:
    """Detect Fair Value Gaps in OHLCV data.

    Bullish FVG: candle[i] low > candle[i-2] high (gap up)
    Bearish FVG: candle[i] high < candle[i-2] low (gap down)

    Args:
        df: DataFrame with 'high', 'low', 'open', 'close' columns
        min_ticks: Minimum FVG size in ticks to qualify
        tick_size: Tick size of the instrument (NQ = 0.25)

    Returns:
        List of detected FVG zones.
    """
    min_size = min_ticks * tick_size
    fvgs: list[FVGZone] = []

    if len(df) < 3:
        return fvgs

    highs = df["high"].values
    lows = df["low"].values
    timestamps = df["timestamp"].values if "timestamp" in df.columns else df.index

    for i in range(2, len(df)):
        # Bullish FVG: current low > 2-bars-ago high
        gap_up = lows[i] - highs[i - 2]
        if gap_up >= min_size:
            fvgs.append(FVGZone(
                direction=Direction.LONG,
                top=lows[i],
                bottom=highs[i - 2],
                timestamp=pd.Timestamp(timestamps[i]).to_pydatetime().replace(tzinfo=timezone.utc)
                if not hasattr(timestamps[i], "tzinfo")
                else pd.Timestamp(timestamps[i]).to_pydatetime(),
                timeframe=df.get("timeframe", pd.Series([""])).iloc[0]
                if "timeframe" in df.columns else "",
            ))

        # Bearish FVG: current high < 2-bars-ago low
        gap_down = lows[i - 2] - highs[i]
        if gap_down >= min_size:
            fvgs.append(FVGZone(
                direction=Direction.SHORT,
                top=lows[i - 2],
                bottom=highs[i],
                timestamp=pd.Timestamp(timestamps[i]).to_pydatetime().replace(tzinfo=timezone.utc)
                if not hasattr(timestamps[i], "tzinfo")
                else pd.Timestamp(timestamps[i]).to_pydatetime(),
                timeframe=df.get("timeframe", pd.Series([""])).iloc[0]
                if "timeframe" in df.columns else "",
            ))

    return fvgs


def check_fvg_retest(
    price: float, fvgs: list[FVGZone], direction: Direction
) -> FVGZone | None:
    """Check if price is retesting an active FVG zone.

    For long entries: price enters a bullish FVG (support)
    For short entries: price enters a bearish FVG (resistance)
    """
    for fvg in fvgs:
        if fvg.filled or fvg.direction != direction:
            continue
        if fvg.bottom <= price <= fvg.top:
            return fvg
    return None


# ── Liquidity Sweep Detection ─────────────────────────


def detect_liquidity_sweeps(
    df: pd.DataFrame, lookback: int = 5
) -> list[ICTSignal]:
    """Detect liquidity sweeps (stop hunts).

    A sweep occurs when price wicks beyond a prior swing high/low
    and closes back inside — indicating a stop hunt by smart money.
    """
    signals: list[ICTSignal] = []

    if len(df) < lookback * 2 + 1:
        return signals

    sh = swing_highs(df, lookback)
    sl = swing_lows(df, lookback)
    timestamps = df["timestamp"].values if "timestamp" in df.columns else df.index

    # Track recent swing levels
    recent_highs: list[float] = []
    recent_lows: list[float] = []

    for i in range(len(df)):
        if not np.isnan(sh.iloc[i]):
            recent_highs.append(sh.iloc[i])
            if len(recent_highs) > 10:
                recent_highs.pop(0)

        if not np.isnan(sl.iloc[i]):
            recent_lows.append(sl.iloc[i])
            if len(recent_lows) > 10:
                recent_lows.pop(0)

        if i < lookback * 2:
            continue

        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        close = df["close"].iloc[i]
        open_price = df["open"].iloc[i]

        ts = pd.Timestamp(timestamps[i]).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Bearish sweep: wick above swing high, close below it
        for sh_level in recent_highs:
            if high > sh_level and close < sh_level and open_price < sh_level:
                signals.append(ICTSignal(
                    pattern="liquidity_sweep",
                    direction=Direction.SHORT,
                    price_level=sh_level,
                    confidence=0.7,
                    timeframe=df["timeframe"].iloc[0] if "timeframe" in df.columns else "",
                    timestamp=ts,
                    metadata={"swept_level": sh_level, "wick_high": high},
                ))
                break

        # Bullish sweep: wick below swing low, close above it
        for sl_level in recent_lows:
            if low < sl_level and close > sl_level and open_price > sl_level:
                signals.append(ICTSignal(
                    pattern="liquidity_sweep",
                    direction=Direction.LONG,
                    price_level=sl_level,
                    confidence=0.7,
                    timeframe=df["timeframe"].iloc[0] if "timeframe" in df.columns else "",
                    timestamp=ts,
                    metadata={"swept_level": sl_level, "wick_low": low},
                ))
                break

    return signals


# ── Displacement Detection ────────────────────────────


def detect_displacement(
    df: pd.DataFrame,
    atr_multiple: float = 2.0,
    atr_period: int = 14,
) -> list[ICTSignal]:
    """Detect displacement candles (institutional momentum).

    A displacement is a strong directional candle with body > Nx ATR,
    indicating institutional participation.
    """
    signals: list[ICTSignal] = []

    if len(df) < atr_period + 1:
        return signals

    atr_vals = atr(df, atr_period)
    timestamps = df["timestamp"].values if "timestamp" in df.columns else df.index

    for i in range(atr_period, len(df)):
        body = abs(df["close"].iloc[i] - df["open"].iloc[i])
        current_atr = atr_vals.iloc[i]

        if current_atr == 0 or np.isnan(current_atr):
            continue

        if body > current_atr * atr_multiple:
            is_bullish = df["close"].iloc[i] > df["open"].iloc[i]
            direction = Direction.LONG if is_bullish else Direction.SHORT

            ts = pd.Timestamp(timestamps[i]).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            signals.append(ICTSignal(
                pattern="displacement",
                direction=direction,
                price_level=df["close"].iloc[i],
                confidence=min(0.5 + (body / current_atr - atr_multiple) * 0.1, 0.9),
                timeframe=df["timeframe"].iloc[0] if "timeframe" in df.columns else "",
                timestamp=ts,
                metadata={
                    "body_size": body,
                    "atr": current_atr,
                    "body_atr_ratio": body / current_atr,
                },
            ))

    return signals


# ── Break of Structure (BOS) / Change of Character (CHoCH) ─


@dataclass
class SwingStructure:
    """Tracks market structure via swing highs and lows."""

    swing_highs: list[float] = field(default_factory=list)
    swing_lows: list[float] = field(default_factory=list)
    trend: str = "neutral"  # "bullish", "bearish", "neutral"


def detect_bos_choch(
    df: pd.DataFrame, lookback: int = 5
) -> list[ICTSignal]:
    """Detect Break of Structure and Change of Character.

    BOS: Trend continuation break (e.g., higher high in uptrend)
    CHoCH: First break against prevailing structure (trend reversal signal)
    """
    signals: list[ICTSignal] = []

    if len(df) < lookback * 3:
        return signals

    sh = swing_highs(df, lookback)
    sl = swing_lows(df, lookback)
    timestamps = df["timestamp"].values if "timestamp" in df.columns else df.index

    prev_sh: float | None = None
    prev_sl: float | None = None
    trend = "neutral"

    for i in range(len(df)):
        ts = pd.Timestamp(timestamps[i]).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        current_high = df["high"].iloc[i]
        current_low = df["low"].iloc[i]

        # Update swing levels
        if not np.isnan(sh.iloc[i]):
            new_sh = sh.iloc[i]

            if prev_sh is not None:
                if trend == "bullish" and new_sh > prev_sh:
                    # BOS: Higher high in uptrend (continuation)
                    signals.append(ICTSignal(
                        pattern="bos",
                        direction=Direction.LONG,
                        price_level=new_sh,
                        confidence=0.6,
                        timeframe=df["timeframe"].iloc[0] if "timeframe" in df.columns else "",
                        timestamp=ts,
                        metadata={"prev_high": prev_sh, "new_high": new_sh},
                    ))
                elif trend == "bearish" and new_sh > prev_sh:
                    # CHoCH: Higher high in downtrend (reversal signal)
                    signals.append(ICTSignal(
                        pattern="choch",
                        direction=Direction.LONG,
                        price_level=new_sh,
                        confidence=0.75,
                        timeframe=df["timeframe"].iloc[0] if "timeframe" in df.columns else "",
                        timestamp=ts,
                        metadata={"prev_high": prev_sh, "new_high": new_sh, "prev_trend": trend},
                    ))
                    trend = "bullish"

            prev_sh = new_sh

        if not np.isnan(sl.iloc[i]):
            new_sl = sl.iloc[i]

            if prev_sl is not None:
                if trend == "bearish" and new_sl < prev_sl:
                    # BOS: Lower low in downtrend (continuation)
                    signals.append(ICTSignal(
                        pattern="bos",
                        direction=Direction.SHORT,
                        price_level=new_sl,
                        confidence=0.6,
                        timeframe=df["timeframe"].iloc[0] if "timeframe" in df.columns else "",
                        timestamp=ts,
                        metadata={"prev_low": prev_sl, "new_low": new_sl},
                    ))
                elif trend == "bullish" and new_sl < prev_sl:
                    # CHoCH: Lower low in uptrend (reversal signal)
                    signals.append(ICTSignal(
                        pattern="choch",
                        direction=Direction.SHORT,
                        price_level=new_sl,
                        confidence=0.75,
                        timeframe=df["timeframe"].iloc[0] if "timeframe" in df.columns else "",
                        timestamp=ts,
                        metadata={"prev_low": prev_sl, "new_low": new_sl, "prev_trend": trend},
                    ))
                    trend = "bearish"

            prev_sl = new_sl

        # Determine initial trend from first two swings
        if trend == "neutral" and prev_sh is not None and prev_sl is not None:
            if not np.isnan(sh.iloc[i]) and prev_sl is not None:
                trend = "bullish" if sh.iloc[i] > prev_sl else "bearish"

    return signals


# ── Order Block Detection ─────────────────────────────


@dataclass
class OrderBlock:
    """An institutional order block (supply/demand zone)."""

    direction: Direction  # LONG = demand block, SHORT = supply block
    top: float
    bottom: float
    timestamp: datetime
    timeframe: str
    mitigated: bool = False


def detect_order_blocks(
    df: pd.DataFrame,
    atr_multiple: float = 2.0,
    atr_period: int = 14,
) -> list[OrderBlock]:
    """Detect order blocks — last opposing candle before displacement.

    A bullish order block is the last bearish candle before a bullish displacement.
    A bearish order block is the last bullish candle before a bearish displacement.
    """
    blocks: list[OrderBlock] = []

    if len(df) < atr_period + 2:
        return blocks

    atr_vals = atr(df, atr_period)
    timestamps = df["timestamp"].values if "timestamp" in df.columns else df.index

    for i in range(atr_period + 1, len(df)):
        body = abs(df["close"].iloc[i] - df["open"].iloc[i])
        current_atr = atr_vals.iloc[i]

        if current_atr == 0 or np.isnan(current_atr):
            continue

        # Is this a displacement candle?
        if body <= current_atr * atr_multiple:
            continue

        is_bullish_displacement = df["close"].iloc[i] > df["open"].iloc[i]

        # Find the last opposing candle before displacement
        for j in range(i - 1, max(i - 6, 0), -1):
            prev_is_bearish = df["close"].iloc[j] < df["open"].iloc[j]
            prev_is_bullish = df["close"].iloc[j] > df["open"].iloc[j]

            if is_bullish_displacement and prev_is_bearish:
                # Bullish OB: last bearish candle before bullish displacement
                ts = pd.Timestamp(timestamps[j]).to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                blocks.append(OrderBlock(
                    direction=Direction.LONG,
                    top=max(df["open"].iloc[j], df["close"].iloc[j]),
                    bottom=df["low"].iloc[j],
                    timestamp=ts,
                    timeframe=df["timeframe"].iloc[0] if "timeframe" in df.columns else "",
                ))
                break

            elif not is_bullish_displacement and prev_is_bullish:
                # Bearish OB: last bullish candle before bearish displacement
                ts = pd.Timestamp(timestamps[j]).to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                blocks.append(OrderBlock(
                    direction=Direction.SHORT,
                    top=df["high"].iloc[j],
                    bottom=min(df["open"].iloc[j], df["close"].iloc[j]),
                    timestamp=ts,
                    timeframe=df["timeframe"].iloc[0] if "timeframe" in df.columns else "",
                ))
                break

    return blocks


# ── SMT Divergence (NQ vs ES) ─────────────────────────


def detect_smt_divergence(
    nq_df: pd.DataFrame,
    es_df: pd.DataFrame,
    lookback: int = 5,
) -> list[ICTSignal]:
    """Detect Smart Money Technique divergence between NQ and ES.

    Bearish SMT: NQ makes higher high but ES does NOT
    Bullish SMT: NQ makes lower low but ES does NOT

    Both DataFrames must be aligned by timestamp and same timeframe.
    """
    signals: list[ICTSignal] = []

    if len(nq_df) < lookback * 2 + 1 or len(es_df) < lookback * 2 + 1:
        return signals

    # Ensure same length
    min_len = min(len(nq_df), len(es_df))
    nq = nq_df.iloc[:min_len].reset_index(drop=True)
    es = es_df.iloc[:min_len].reset_index(drop=True)

    nq_sh = swing_highs(nq, lookback)
    nq_sl = swing_lows(nq, lookback)
    es_sh = swing_highs(es, lookback)
    es_sl = swing_lows(es, lookback)

    timestamps = nq["timestamp"].values if "timestamp" in nq.columns else nq.index

    prev_nq_high: float | None = None
    prev_es_high: float | None = None
    prev_nq_low: float | None = None
    prev_es_low: float | None = None

    for i in range(len(nq)):
        ts = pd.Timestamp(timestamps[i]).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Track swing highs
        if not np.isnan(nq_sh.iloc[i]):
            curr_nq_high = nq_sh.iloc[i]
            curr_es_high = es_sh.iloc[i] if not np.isnan(es_sh.iloc[i]) else prev_es_high

            if prev_nq_high is not None and prev_es_high is not None and curr_es_high is not None:
                # Bearish SMT: NQ higher high, ES no higher high
                if curr_nq_high > prev_nq_high and curr_es_high <= prev_es_high:
                    signals.append(ICTSignal(
                        pattern="smt",
                        direction=Direction.SHORT,
                        price_level=curr_nq_high,
                        confidence=0.7,
                        timeframe=nq["timeframe"].iloc[0] if "timeframe" in nq.columns else "",
                        timestamp=ts,
                        metadata={
                            "nq_high": curr_nq_high,
                            "nq_prev_high": prev_nq_high,
                            "es_high": curr_es_high,
                            "es_prev_high": prev_es_high,
                        },
                    ))

            prev_nq_high = curr_nq_high
            if not np.isnan(es_sh.iloc[i]):
                prev_es_high = es_sh.iloc[i]

        # Track swing lows
        if not np.isnan(nq_sl.iloc[i]):
            curr_nq_low = nq_sl.iloc[i]
            curr_es_low = es_sl.iloc[i] if not np.isnan(es_sl.iloc[i]) else prev_es_low

            if prev_nq_low is not None and prev_es_low is not None and curr_es_low is not None:
                # Bullish SMT: NQ lower low, ES no lower low
                if curr_nq_low < prev_nq_low and curr_es_low >= prev_es_low:
                    signals.append(ICTSignal(
                        pattern="smt",
                        direction=Direction.LONG,
                        price_level=curr_nq_low,
                        confidence=0.7,
                        timeframe=nq["timeframe"].iloc[0] if "timeframe" in nq.columns else "",
                        timestamp=ts,
                        metadata={
                            "nq_low": curr_nq_low,
                            "nq_prev_low": prev_nq_low,
                            "es_low": curr_es_low,
                            "es_prev_low": prev_es_low,
                        },
                    ))

            prev_nq_low = curr_nq_low
            if not np.isnan(es_sl.iloc[i]):
                prev_es_low = es_sl.iloc[i]

    return signals
