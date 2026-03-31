"""Order flow analysis from Tradovate DOM data.

Processes depth of market snapshots to compute:
- Cumulative delta (buy vs sell volume)
- Absorption detection (institutional support/resistance)
- Stacked imbalances (3+ consecutive bid/ask imbalances)
- Delta divergence (price vs delta disagreement)
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import structlog

from src.data.models import (
    DOMSnapshot,
    Direction,
    OrderFlowState,
    Quote,
)

logger = structlog.get_logger(__name__)

# Thresholds
IMBALANCE_RATIO = 3.0  # Bid or ask must be 3x the other side
ABSORPTION_VOLUME_THRESHOLD = 100  # Minimum volume to consider absorption
STACKED_IMBALANCE_COUNT = 3  # Consecutive levels needed


class OrderFlowAnalyzer:
    """Analyzes DOM and trade data to detect order flow patterns.

    This provides an institutional-level edge that pure candlestick
    analysis cannot detect. Feed the output to the signal aggregator.
    """

    def __init__(self):
        self._cumulative_delta: float = 0.0
        self._delta_history: deque[float] = deque(maxlen=100)  # Last 100 delta readings
        self._price_history: deque[float] = deque(maxlen=100)
        self._last_dom: DOMSnapshot | None = None
        self._absorption_events: deque[dict] = deque(maxlen=20)
        self._state = OrderFlowState()
        self._session_delta: float = 0.0  # Resets each session

    @property
    def state(self) -> OrderFlowState:
        return self._state

    def reset_session(self) -> None:
        """Reset delta tracking for a new trading session."""
        self._session_delta = 0.0
        self._cumulative_delta = 0.0
        self._delta_history.clear()
        self._price_history.clear()
        self._absorption_events.clear()

    def process_quote(self, quote: Quote) -> None:
        """Process a trade tick to update cumulative delta.

        Approximation: if last_price >= ask, count as buy (positive delta).
        If last_price <= bid, count as sell (negative delta).
        """
        if quote.last_size == 0:
            return

        if quote.last_price >= quote.ask_price:
            # Trade at ask = aggressive buy
            self._cumulative_delta += quote.last_size
        elif quote.last_price <= quote.bid_price:
            # Trade at bid = aggressive sell
            self._cumulative_delta -= quote.last_size

        self._delta_history.append(self._cumulative_delta)
        self._price_history.append(quote.last_price)
        self._session_delta = self._cumulative_delta

    def process_dom(self, dom: DOMSnapshot) -> OrderFlowState:
        """Process a DOM snapshot and return updated order flow state.

        Detects:
        1. Stacked imbalances
        2. Absorption patterns
        3. Delta divergence
        """
        self._last_dom = dom

        # Detect stacked imbalances
        stacked_levels, imbalance_dir = self._detect_stacked_imbalances(dom)

        # Detect absorption
        absorption, absorption_dir = self._detect_absorption(dom)

        # Detect delta divergence
        delta_divergence = self._detect_delta_divergence()

        # Determine delta trend
        delta_trend = self._compute_delta_trend()

        self._state = OrderFlowState(
            cumulative_delta=self._cumulative_delta,
            delta_trend=delta_trend,
            absorption_detected=absorption,
            absorption_direction=absorption_dir,
            stacked_imbalances=stacked_levels,
            imbalance_direction=imbalance_dir,
            delta_divergence=delta_divergence,
            timestamp=datetime.now(timezone.utc),
        )

        return self._state

    def _detect_stacked_imbalances(
        self, dom: DOMSnapshot
    ) -> tuple[list[float], Direction | None]:
        """Detect 3+ consecutive price levels with volume imbalance.

        Stacked bid imbalances (bid >> ask) = institutional buying = bullish
        Stacked ask imbalances (ask >> bid) = institutional selling = bearish
        """
        if len(dom.levels) < STACKED_IMBALANCE_COUNT:
            return [], None

        bid_streak: list[float] = []
        ask_streak: list[float] = []
        best_bid_streak: list[float] = []
        best_ask_streak: list[float] = []

        for level in dom.levels:
            total = level.bid_size + level.ask_size
            if total == 0:
                bid_streak = []
                ask_streak = []
                continue

            if level.bid_size >= level.ask_size * IMBALANCE_RATIO and level.bid_size > 0:
                bid_streak.append(level.price)
                ask_streak = []
            elif level.ask_size >= level.bid_size * IMBALANCE_RATIO and level.ask_size > 0:
                ask_streak.append(level.price)
                bid_streak = []
            else:
                bid_streak = []
                ask_streak = []

            if len(bid_streak) > len(best_bid_streak):
                best_bid_streak = bid_streak.copy()
            if len(ask_streak) > len(best_ask_streak):
                best_ask_streak = ask_streak.copy()

        if len(best_bid_streak) >= STACKED_IMBALANCE_COUNT:
            return best_bid_streak, Direction.LONG
        if len(best_ask_streak) >= STACKED_IMBALANCE_COUNT:
            return best_ask_streak, Direction.SHORT

        return [], None

    def _detect_absorption(
        self, dom: DOMSnapshot
    ) -> tuple[bool, Direction | None]:
        """Detect absorption: large resting orders that absorb aggressive flow.

        If there's a large bid that isn't moving despite sell pressure = bullish absorption.
        If there's a large ask that isn't moving despite buy pressure = bearish absorption.
        """
        if not self._last_dom or len(dom.levels) < 5:
            return False, None

        # Find the level with the largest bid
        max_bid_level = max(dom.levels, key=lambda l: l.bid_size, default=None)
        max_ask_level = max(dom.levels, key=lambda l: l.ask_size, default=None)

        if max_bid_level and max_bid_level.bid_size >= ABSORPTION_VOLUME_THRESHOLD:
            # Large bid resting — check if delta is negative (sellers hitting it)
            if self._cumulative_delta < 0 and len(self._delta_history) > 5:
                recent_delta_change = self._delta_history[-1] - self._delta_history[-5]
                if recent_delta_change < 0:
                    return True, Direction.LONG  # Bullish absorption

        if max_ask_level and max_ask_level.ask_size >= ABSORPTION_VOLUME_THRESHOLD:
            # Large ask resting — check if delta is positive (buyers hitting it)
            if self._cumulative_delta > 0 and len(self._delta_history) > 5:
                recent_delta_change = self._delta_history[-1] - self._delta_history[-5]
                if recent_delta_change > 0:
                    return True, Direction.SHORT  # Bearish absorption

        return False, None

    def _detect_delta_divergence(self) -> bool:
        """Detect divergence between price and cumulative delta.

        Price makes new high but delta doesn't → exhaustion (bearish divergence).
        Price makes new low but delta doesn't → exhaustion (bullish divergence).
        """
        if len(self._price_history) < 20 or len(self._delta_history) < 20:
            return False

        prices = list(self._price_history)
        deltas = list(self._delta_history)

        # Check last 20 readings
        recent_prices = prices[-20:]
        recent_deltas = deltas[-20:]

        # Price making new highs?
        price_high = max(recent_prices)
        price_high_idx = recent_prices.index(price_high)

        if price_high_idx >= 15:  # Recent high
            # Is delta also at its high?
            delta_at_price_high = recent_deltas[price_high_idx]
            delta_high = max(recent_deltas)
            if delta_at_price_high < delta_high * 0.8:
                return True  # Bearish divergence

        # Price making new lows?
        price_low = min(recent_prices)
        price_low_idx = recent_prices.index(price_low)

        if price_low_idx >= 15:  # Recent low
            delta_at_price_low = recent_deltas[price_low_idx]
            delta_low = min(recent_deltas)
            if delta_at_price_low > delta_low * 0.8:
                return True  # Bullish divergence

        return False

    def _compute_delta_trend(self) -> str:
        """Determine the trend of cumulative delta."""
        if len(self._delta_history) < 10:
            return "neutral"

        deltas = list(self._delta_history)
        recent = deltas[-10:]
        earlier = deltas[-20:-10] if len(deltas) >= 20 else deltas[:10]

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier) if earlier else 0

        diff = recent_avg - earlier_avg
        threshold = abs(earlier_avg) * 0.1 if earlier_avg != 0 else 10

        if diff > threshold:
            return "rising"
        elif diff < -threshold:
            return "falling"
        return "neutral"
