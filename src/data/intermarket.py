"""Intermarket correlation tracking for NQ context.

Monitors DXY (Dollar Index), VIX, and ZN (10-Year Treasury) to provide
macroeconomic context for NQ trading decisions. These instruments are
tracked only — never traded.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from src.data.models import IntermarketSnapshot, Quote

logger = structlog.get_logger(__name__)

# VIX thresholds
VIX_LOW = 15.0
VIX_ELEVATED = 25.0
VIX_HIGH = 30.0


class IntermarketTracker:
    """Tracks correlated instruments and computes an intermarket context snapshot.

    Fed to Claude for regime analysis. Key relationships:
    - Strong DXY (rising dollar) → typically bearish for NQ
    - Elevated VIX (>25) → high fear, reduce position sizes
    - Rising yields (falling ZN) → typically pressures NQ
    """

    def __init__(self):
        self._dxy_price: float | None = None
        self._dxy_prev_close: float | None = None
        self._vix_price: float | None = None
        self._zn_price: float | None = None
        self._zn_prev_close: float | None = None
        self._last_update: datetime | None = None

    @property
    def snapshot(self) -> IntermarketSnapshot:
        """Get the current intermarket context snapshot."""
        dxy_change = None
        if self._dxy_price and self._dxy_prev_close and self._dxy_prev_close > 0:
            dxy_change = ((self._dxy_price - self._dxy_prev_close) / self._dxy_prev_close) * 100

        zn_change = None
        if self._zn_price and self._zn_prev_close and self._zn_prev_close > 0:
            zn_change = ((self._zn_price - self._zn_prev_close) / self._zn_prev_close) * 100

        vix_level = "normal"
        if self._vix_price:
            if self._vix_price >= VIX_HIGH:
                vix_level = "high"
            elif self._vix_price >= VIX_ELEVATED:
                vix_level = "elevated"
            elif self._vix_price <= VIX_LOW:
                vix_level = "low"

        return IntermarketSnapshot(
            dxy_price=self._dxy_price,
            dxy_change_pct=dxy_change,
            vix_price=self._vix_price,
            vix_level=vix_level,
            zn_price=self._zn_price,
            zn_change_pct=zn_change,
            timestamp=self._last_update,
        )

    def process_quote(self, symbol: str, quote: Quote) -> None:
        """Process a quote update for a tracked intermarket instrument.

        Args:
            symbol: The instrument identifier (should contain 'DX', 'VX', or 'ZN')
            quote: The quote data
        """
        self._last_update = datetime.now(timezone.utc)

        if "DX" in symbol.upper():
            if self._dxy_price is None and self._dxy_prev_close is None:
                self._dxy_prev_close = quote.last_price
            self._dxy_price = quote.last_price

        elif "VX" in symbol.upper() or "VIX" in symbol.upper():
            self._vix_price = quote.last_price

        elif "ZN" in symbol.upper():
            if self._zn_price is None and self._zn_prev_close is None:
                self._zn_prev_close = quote.last_price
            self._zn_price = quote.last_price

    def set_previous_closes(
        self,
        dxy_close: float | None = None,
        zn_close: float | None = None,
    ) -> None:
        """Set previous day closes for change calculation.

        Call this once at session start with historical data.
        """
        if dxy_close is not None:
            self._dxy_prev_close = dxy_close
        if zn_close is not None:
            self._zn_prev_close = zn_close

    def should_reduce_size(self) -> bool:
        """Check if intermarket conditions suggest reducing position size.

        Returns True if VIX is elevated or high.
        """
        if self._vix_price and self._vix_price >= VIX_ELEVATED:
            return True
        return False

    def get_nq_bias(self) -> str:
        """Get directional bias for NQ based on intermarket context.

        Returns 'bullish', 'bearish', or 'neutral'.
        """
        signals = []

        # DXY: strong dollar = bearish for NQ
        if self._dxy_price and self._dxy_prev_close:
            dxy_change = ((self._dxy_price - self._dxy_prev_close) / self._dxy_prev_close) * 100
            if dxy_change > 0.3:
                signals.append("bearish")
            elif dxy_change < -0.3:
                signals.append("bullish")

        # VIX: high fear = bearish
        if self._vix_price:
            if self._vix_price >= VIX_ELEVATED:
                signals.append("bearish")
            elif self._vix_price <= VIX_LOW:
                signals.append("bullish")

        # ZN: falling ZN (rising yields) = bearish for NQ
        if self._zn_price and self._zn_prev_close:
            zn_change = ((self._zn_price - self._zn_prev_close) / self._zn_prev_close) * 100
            if zn_change < -0.2:
                signals.append("bearish")
            elif zn_change > 0.2:
                signals.append("bullish")

        if not signals:
            return "neutral"

        bullish_count = signals.count("bullish")
        bearish_count = signals.count("bearish")

        if bullish_count > bearish_count:
            return "bullish"
        elif bearish_count > bullish_count:
            return "bearish"
        return "neutral"
