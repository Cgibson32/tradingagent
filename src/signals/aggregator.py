"""Signal aggregator with confidence scoring and time-awareness.

Combines all signal sources (ICT patterns, indicators, order flow,
key levels, kill zones) into a unified TradeSignal with confidence
scoring and time-adjusted targets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.data.models import (
    Direction,
    ICTSignal,
    KeyLevels,
    MarketState,
    OrderFlowState,
    TradeSignal,
)
from src.signals.htf_bias import Bias, HTFBiasEngine
from src.signals.kill_zones import KillZoneManager


class SignalAggregator:
    """Aggregates multiple signal sources into actionable trade signals.

    Signal weights (from config):
        ict_pattern: 0.35
        htf_bias: 0.20
        indicators: 0.15
        order_flow: 0.15
        key_levels: 0.10
        intermarket: 0.05

    Applies time-decay as a final confidence gate.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        min_confidence: float = 0.65,
        full_tp_r: float = 2.5,
        kill_zone_mgr: KillZoneManager | None = None,
        htf_bias_engine: HTFBiasEngine | None = None,
    ):
        self._weights = weights or {
            "ict_pattern": 0.35,
            "htf_bias": 0.20,
            "indicators": 0.15,
            "order_flow": 0.15,
            "key_levels": 0.10,
            "intermarket": 0.05,
        }
        self._min_confidence = min_confidence
        self._full_tp_r = full_tp_r
        self._kz = kill_zone_mgr or KillZoneManager()
        self._htf = htf_bias_engine or HTFBiasEngine()
        self._avg_trade_duration_min: float = 45.0  # Updated from trade history

    def update_avg_trade_duration(self, minutes: float) -> None:
        """Update the rolling average trade duration from historical data."""
        self._avg_trade_duration_min = minutes

    def evaluate(
        self,
        ict_signals: list[ICTSignal],
        htf_bias: Bias,
        htf_confidence: float,
        indicator_scores: dict[str, float],
        order_flow: OrderFlowState,
        key_levels: KeyLevels,
        intermarket_bias: str,
        current_price: float,
        atr_value: float,
        now: datetime | None = None,
    ) -> TradeSignal | None:
        """Evaluate all signals and produce a trade signal if conditions align.

        Returns None if no signal meets the minimum confidence threshold
        or if time/kill zone constraints prevent entry.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Gate 1: Can we enter at all?
        if not self._kz.can_enter_new_trade(now):
            return None

        # Gate 2: Enough time for a trade?
        if not self._kz.has_enough_time(now, self._avg_trade_duration_min):
            return None

        # Determine primary direction from ICT signals
        direction = self._determine_direction(ict_signals, htf_bias)
        if direction is None:
            return None

        # Calculate component scores
        scores: dict[str, float] = {}

        # ICT pattern score
        matching_ict = [s for s in ict_signals if s.direction == direction]
        if matching_ict:
            scores["ict_pattern"] = max(s.confidence for s in matching_ict)
        else:
            scores["ict_pattern"] = 0.0

        # HTF bias score
        if (direction == Direction.LONG and htf_bias == Bias.BULLISH) or \
           (direction == Direction.SHORT and htf_bias == Bias.BEARISH):
            scores["htf_bias"] = htf_confidence
        else:
            scores["htf_bias"] = 0.0

        # Indicator score
        scores["indicators"] = self._score_indicators(indicator_scores, direction)

        # Order flow score
        scores["order_flow"] = self._score_order_flow(order_flow, direction)

        # Key levels score
        scores["key_levels"] = self._score_key_levels(key_levels, current_price, direction)

        # Intermarket score
        scores["intermarket"] = self._score_intermarket(intermarket_bias, direction)

        # Weighted confidence
        raw_confidence = sum(
            scores.get(k, 0.0) * self._weights.get(k, 0.0)
            for k in self._weights
        )

        # Apply kill zone and time decay adjustments
        adjusted_confidence = self._kz.adjust_confidence(raw_confidence, now)

        # Gate 3: Minimum confidence threshold
        if adjusted_confidence < self._min_confidence:
            return None

        # Calculate entry, SL, TP
        entry_price, stop_loss, take_profit = self._calculate_levels(
            direction, current_price, atr_value, matching_ict, key_levels
        )

        # Time-adjusted TP
        time_adjusted_tp_r = self._kz.get_time_adjusted_tp(
            self._full_tp_r, now, self._avg_trade_duration_min
        )
        risk = abs(entry_price - stop_loss)
        if direction == Direction.LONG:
            time_adjusted_tp = entry_price + risk * time_adjusted_tp_r
        else:
            time_adjusted_tp = entry_price - risk * time_adjusted_tp_r

        mins_left = self._kz.minutes_until_close(now)

        return TradeSignal(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_adjusted_tp=time_adjusted_tp,
            confidence=adjusted_confidence,
            minutes_until_close=mins_left,
            estimated_duration_minutes=self._avg_trade_duration_min,
            strategy_name="ict_aggregated",
            contributing_signals=matching_ict,
            metadata={
                "component_scores": scores,
                "raw_confidence": raw_confidence,
                "kill_zone": self._kz.get_active_zone_label(now),
                "htf_bias": htf_bias.value,
                "time_adjusted_tp_r": time_adjusted_tp_r,
            },
        )

    def _determine_direction(
        self, ict_signals: list[ICTSignal], htf_bias: Bias
    ) -> Direction | None:
        """Determine trade direction from ICT signals aligned with HTF bias."""
        if not ict_signals:
            return None

        # Count directional signals
        long_signals = [s for s in ict_signals if s.direction == Direction.LONG]
        short_signals = [s for s in ict_signals if s.direction == Direction.SHORT]

        long_score = sum(s.confidence for s in long_signals)
        short_score = sum(s.confidence for s in short_signals)

        # Must align with HTF bias
        if htf_bias == Bias.BULLISH and long_score > short_score:
            return Direction.LONG
        if htf_bias == Bias.BEARISH and short_score > long_score:
            return Direction.SHORT

        # If bias is neutral, take the stronger side (with reduced confidence)
        if htf_bias == Bias.NEUTRAL:
            if long_score > short_score * 1.5:
                return Direction.LONG
            if short_score > long_score * 1.5:
                return Direction.SHORT

        return None

    def _score_indicators(
        self, indicator_scores: dict[str, float], direction: Direction
    ) -> float:
        """Score based on traditional indicator alignment."""
        score = 0.0
        count = 0

        # RSI
        rsi = indicator_scores.get("rsi", 50.0)
        if direction == Direction.LONG and rsi < 40:
            score += 0.7  # Oversold = bullish
        elif direction == Direction.SHORT and rsi > 60:
            score += 0.7  # Overbought = bearish
        elif direction == Direction.LONG and 40 <= rsi <= 50:
            score += 0.3
        elif direction == Direction.SHORT and 50 <= rsi <= 60:
            score += 0.3
        count += 1

        # MACD
        macd_hist = indicator_scores.get("macd_histogram", 0.0)
        if direction == Direction.LONG and macd_hist > 0:
            score += 0.6
        elif direction == Direction.SHORT and macd_hist < 0:
            score += 0.6
        count += 1

        # ADX (trend strength)
        adx_val = indicator_scores.get("adx", 20.0)
        if adx_val > 25:
            score += 0.5  # Strong trend = good for directional trades
        count += 1

        # EMA alignment
        ema_align = indicator_scores.get("ema_alignment", "mixed")
        if direction == Direction.LONG and ema_align == "bullish":
            score += 0.8
        elif direction == Direction.SHORT and ema_align == "bearish":
            score += 0.8
        count += 1

        return score / max(count, 1)

    def _score_order_flow(
        self, of: OrderFlowState, direction: Direction
    ) -> float:
        """Score based on order flow analysis."""
        score = 0.0

        # Delta trend alignment
        if direction == Direction.LONG and of.delta_trend == "rising":
            score += 0.4
        elif direction == Direction.SHORT and of.delta_trend == "falling":
            score += 0.4

        # Absorption (strong signal)
        if of.absorption_detected and of.absorption_direction == direction:
            score += 0.4

        # Stacked imbalances
        if of.imbalance_direction == direction and len(of.stacked_imbalances) >= 3:
            score += 0.3

        # Delta divergence (exhaustion = potential reversal)
        if of.delta_divergence:
            score += 0.2

        return min(score, 1.0)

    def _score_key_levels(
        self, levels: KeyLevels, price: float, direction: Direction
    ) -> float:
        """Score based on proximity to key ICT levels."""
        score = 0.0
        tolerance = 10.0  # points

        if direction == Direction.LONG:
            # Near support levels = good for longs
            support_levels = [
                levels.prev_day_low,
                levels.prev_week_low,
                levels.prev_month_low,
                levels.prev_day_midpoint,
            ]
            for lvl in support_levels:
                if lvl > 0 and abs(price - lvl) < tolerance:
                    score += 0.3
        else:
            # Near resistance levels = good for shorts
            resistance_levels = [
                levels.prev_day_high,
                levels.prev_week_high,
                levels.prev_month_high,
                levels.prev_day_midpoint,
            ]
            for lvl in resistance_levels:
                if lvl > 0 and abs(price - lvl) < tolerance:
                    score += 0.3

        return min(score, 1.0)

    def _score_intermarket(self, bias: str, direction: Direction) -> float:
        """Score based on intermarket context."""
        if direction == Direction.LONG and bias == "bullish":
            return 0.7
        if direction == Direction.SHORT and bias == "bearish":
            return 0.7
        if bias == "neutral":
            return 0.3
        return 0.0  # Misaligned

    def _calculate_levels(
        self,
        direction: Direction,
        price: float,
        atr_value: float,
        ict_signals: list[ICTSignal],
        key_levels: KeyLevels,
    ) -> tuple[float, float, float]:
        """Calculate entry price, stop loss, and take profit.

        Stop loss: Below/above the nearest order block or swing + ATR buffer
        Take profit: 2.5R from entry
        """
        # Default SL distance: 1.5x ATR
        sl_distance = atr_value * 1.5

        # Try to place SL beyond the signal level (order block, FVG, etc.)
        for sig in ict_signals:
            if sig.pattern in ("liquidity_sweep", "fvg", "order_block"):
                level_distance = abs(price - sig.price_level)
                if 0 < level_distance < sl_distance * 2:
                    sl_distance = level_distance + atr_value * 0.25  # Add buffer

        if direction == Direction.LONG:
            entry = price
            stop_loss = entry - sl_distance
            take_profit = entry + sl_distance * self._full_tp_r
        else:
            entry = price
            stop_loss = entry + sl_distance
            take_profit = entry - sl_distance * self._full_tp_r

        return entry, stop_loss, take_profit
