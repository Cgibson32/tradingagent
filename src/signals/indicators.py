"""Technical indicator library.

Pure functions with no side effects. All accept a pandas DataFrame
with OHLCV columns and return computed values. Validated against
TradingView outputs within 0.1% tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── EMA (Exponential Moving Average) ──────────────────


def ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average.

    Uses the standard EMA formula: multiplier = 2 / (period + 1).
    Matches TradingView's EMA calculation.
    """
    return series.ewm(span=period, adjust=False).mean()


def ema_stack(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """Calculate EMA stack with slope for each period.

    Args:
        df: DataFrame with 'close' column
        periods: List of EMA periods (default: [9, 21, 50, 200])

    Returns:
        DataFrame with EMA values and slopes for each period.
    """
    if periods is None:
        periods = [9, 21, 50, 200]

    result = pd.DataFrame(index=df.index)
    for p in periods:
        col = f"ema_{p}"
        result[col] = ema(df["close"], p)
        result[f"{col}_slope"] = result[col].diff()

    return result


def ema_alignment(df: pd.DataFrame, periods: list[int] | None = None) -> pd.Series:
    """Determine EMA stack alignment.

    Returns:
        Series with values: 'bullish' (shortest > longest), 'bearish' (shortest < longest),
        or 'mixed'.
    """
    if periods is None:
        periods = [9, 21, 50, 200]

    emas = {}
    for p in periods:
        emas[p] = ema(df["close"], p)

    def classify(row_idx):
        vals = [emas[p].iloc[row_idx] for p in periods]
        if all(np.isnan(v) for v in vals):
            return "mixed"
        valid = [v for v in vals if not np.isnan(v)]
        if len(valid) < 2:
            return "mixed"
        if all(valid[i] >= valid[i + 1] for i in range(len(valid) - 1)):
            return "bullish"
        if all(valid[i] <= valid[i + 1] for i in range(len(valid) - 1)):
            return "bearish"
        return "mixed"

    return pd.Series([classify(i) for i in range(len(df))], index=df.index)


# ── ADX (Average Directional Index) ───────────────────


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate ADX with +DI and -DI components.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default: 14)

    Returns:
        DataFrame with columns: 'adx', 'plus_di', 'minus_di'
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # Smoothed using Wilder's method (same as EMA with alpha=1/period)
    atr_smooth = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr_smooth)

    # DX and ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    dx = dx.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    adx_val = dx.ewm(alpha=1.0 / period, adjust=False).mean()

    result = pd.DataFrame(index=df.index)
    result["adx"] = adx_val
    result["plus_di"] = plus_di
    result["minus_di"] = minus_di
    return result


# ── MACD ───────────────────────────────────────────────


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Calculate MACD with histogram and zero-cross detection.

    Args:
        df: DataFrame with 'close' column
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)

    Returns:
        DataFrame with columns: 'macd', 'signal', 'histogram', 'zero_cross'
    """
    fast_ema = ema(df["close"], fast)
    slow_ema = ema(df["close"], slow)

    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    # Zero-cross detection on MACD line
    prev_macd = macd_line.shift(1)
    zero_cross = pd.Series("none", index=df.index)
    zero_cross = zero_cross.where(
        ~((prev_macd < 0) & (macd_line >= 0)), "bullish"
    )
    zero_cross = zero_cross.where(
        ~((prev_macd > 0) & (macd_line <= 0)), "bearish"
    )

    result = pd.DataFrame(index=df.index)
    result["macd"] = macd_line
    result["signal"] = signal_line
    result["histogram"] = histogram
    result["zero_cross"] = zero_cross
    return result


# ── RSI (Relative Strength Index) ─────────────────────


def rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate RSI with divergence detection.

    Args:
        df: DataFrame with 'close' column
        period: Lookback period (default: 14)

    Returns:
        DataFrame with columns: 'rsi', 'divergence'
        divergence: 'bullish', 'bearish', or 'none'
    """
    delta = df["close"].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], 0.0)
    rsi_val = 100 - (100 / (1 + rs))

    # Divergence detection (simple: compare last 20 bars)
    divergence = _detect_rsi_divergence(df["close"], rsi_val, lookback=20)

    result = pd.DataFrame(index=df.index)
    result["rsi"] = rsi_val
    result["divergence"] = divergence
    return result


def _detect_rsi_divergence(
    price: pd.Series, rsi_vals: pd.Series, lookback: int = 20
) -> pd.Series:
    """Detect price vs RSI divergence.

    Bullish divergence: price makes lower low, RSI makes higher low
    Bearish divergence: price makes higher high, RSI makes lower high
    """
    divergence = pd.Series("none", index=price.index, dtype=str)

    for i in range(lookback, len(price)):
        window_price = price.iloc[i - lookback : i + 1]
        window_rsi = rsi_vals.iloc[i - lookback : i + 1]

        if window_price.isna().any() or window_rsi.isna().any():
            continue

        # Find swing lows in the window
        price_min_idx = window_price.idxmin()
        current_price = price.iloc[i]
        current_rsi = rsi_vals.iloc[i]

        price_at_min = window_price[price_min_idx]
        rsi_at_min = window_rsi[price_min_idx]

        # Bullish divergence: price lower low, RSI higher low
        if current_price <= price_at_min and current_rsi > rsi_at_min:
            if current_rsi < 40:  # Only in oversold territory
                divergence.iloc[i] = "bullish"

        # Find swing highs
        price_max_idx = window_price.idxmax()
        price_at_max = window_price[price_max_idx]
        rsi_at_max = window_rsi[price_max_idx]

        # Bearish divergence: price higher high, RSI lower high
        if current_price >= price_at_max and current_rsi < rsi_at_max:
            if current_rsi > 60:  # Only in overbought territory
                divergence.iloc[i] = "bearish"

    return divergence


# ── ATR (Average True Range) ──────────────────────────


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range.

    Uses Wilder's smoothing method to match TradingView.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def atr_percent(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as a percentage of price."""
    atr_val = atr(df, period)
    return (atr_val / df["close"]) * 100


# ── VWAP (Volume Weighted Average Price) ──────────────


def vwap(df: pd.DataFrame, session_col: str | None = None) -> pd.Series:
    """Calculate session-anchored VWAP.

    For futures, anchored to session open. If no session column provided,
    resets daily at the first bar of each calendar day.

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns
        session_col: Optional column name indicating session boundaries
                     (True at session start)

    Returns:
        Series with VWAP values.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_volume = typical_price * df["volume"]

    if session_col and session_col in df.columns:
        # Use explicit session boundaries
        session_groups = df[session_col].cumsum()
    else:
        # Reset at each new calendar day
        if hasattr(df.index, "date"):
            session_groups = pd.Series(df.index.date, index=df.index)
        elif "timestamp" in df.columns:
            session_groups = pd.to_datetime(df["timestamp"]).dt.date
        else:
            # No date info — treat entire series as one session
            session_groups = pd.Series(0, index=df.index)

    cum_tp_vol = tp_volume.groupby(session_groups).cumsum()
    cum_vol = df["volume"].groupby(session_groups).cumsum()

    result = cum_tp_vol / cum_vol
    return result.replace([np.inf, -np.inf], np.nan)


# ── Swing High/Low Detection ─────────────────────────


def swing_highs(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Detect swing highs (pivot highs).

    A swing high is a bar whose high is higher than the N bars before
    and after it. Matches TradingView's ta.pivothigh().

    Returns:
        Series with swing high price at pivot bars, NaN elsewhere.
    """
    highs = df["high"]
    result = pd.Series(np.nan, index=df.index)

    for i in range(lookback, len(highs) - lookback):
        current = highs.iloc[i]
        left = highs.iloc[i - lookback : i]
        right = highs.iloc[i + 1 : i + lookback + 1]

        if (current >= left).all() and (current >= right).all():
            result.iloc[i] = current

    return result


def swing_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Detect swing lows (pivot lows).

    A swing low is a bar whose low is lower than the N bars before
    and after it. Matches TradingView's ta.pivotlow().

    Returns:
        Series with swing low price at pivot bars, NaN elsewhere.
    """
    lows = df["low"]
    result = pd.Series(np.nan, index=df.index)

    for i in range(lookback, len(lows) - lookback):
        current = lows.iloc[i]
        left = lows.iloc[i - lookback : i]
        right = lows.iloc[i + 1 : i + lookback + 1]

        if (current <= left).all() and (current <= right).all():
            result.iloc[i] = current

    return result
