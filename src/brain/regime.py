"""Market regime classifier using Claude API.

Determines whether the market is trending (bullish/bearish),
ranging, or volatile/choppy. Caches the result and only
re-evaluates on a timer or significant structure change.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import structlog

from src.brain.claude_client import ClaudeClient
from src.brain.prompts import regime_classification_prompt
from src.data.models import IntermarketSnapshot, MarketRegime, OrderFlowState

logger = structlog.get_logger(__name__)


class RegimeClassifier:
    """Classifies market regime using Claude + technical data.

    Caches the regime and re-evaluates:
    - Every N minutes (configurable, default 30)
    - When explicitly requested (e.g., on structure break)

    Falls back to technical-only classification if Claude is unavailable.
    """

    def __init__(
        self,
        claude_client: ClaudeClient,
        update_interval_minutes: int = 30,
    ):
        self._claude = claude_client
        self._interval = update_interval_minutes * 60  # Convert to seconds
        self._last_update: float = 0
        self._current_regime = MarketRegime.RANGING
        self._confidence: float = 0.0
        self._reasoning: str = ""

    @property
    def regime(self) -> MarketRegime:
        return self._current_regime

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def reasoning(self) -> str:
        return self._reasoning

    @property
    def is_stale(self) -> bool:
        return (time.time() - self._last_update) > self._interval

    async def classify(
        self,
        df: pd.DataFrame,
        adx_value: float,
        atr_value: float,
        atr_20d_avg: float,
        ema_alignment: str,
        order_flow: OrderFlowState,
        intermarket: IntermarketSnapshot,
        force: bool = False,
    ) -> tuple[MarketRegime, float]:
        """Classify the current market regime.

        Args:
            df: Recent OHLCV data (100+ bars)
            adx_value: Current ADX(14) value
            atr_value: Current ATR(14) value
            atr_20d_avg: 20-day average ATR for percentile calculation
            ema_alignment: 'bullish', 'bearish', or 'mixed'
            order_flow: Current order flow state
            intermarket: Current intermarket snapshot
            force: Force re-evaluation even if cache is fresh

        Returns:
            Tuple of (MarketRegime, confidence)
        """
        if not force and not self.is_stale:
            return self._current_regime, self._confidence

        # Calculate ATR percentile
        atr_percentile = (atr_value / atr_20d_avg * 100) if atr_20d_avg > 0 else 50

        # Build candle summary
        candle_summary = self._summarize_candles(df)

        # Build volume profile
        volume_profile = self._summarize_volume(df)

        # Build intermarket context
        im_context = self._format_intermarket(intermarket)

        # Build order flow summary
        of_summary = self._format_order_flow(order_flow)

        # Try Claude
        system, user = regime_classification_prompt(
            candle_summary=candle_summary,
            adx_value=adx_value,
            atr_percentile=atr_percentile,
            ema_alignment=ema_alignment,
            volume_profile=volume_profile,
            intermarket_context=im_context,
            order_flow_summary=of_summary,
        )

        result = await self._claude.ask(user, system=system, mode="realtime")

        if result.get("fallback"):
            # Claude unavailable — use technical fallback
            logger.warning("regime.claude_unavailable", error=result.get("error"))
            return self._technical_fallback(adx_value, ema_alignment, atr_percentile)

        # Parse Claude's response
        try:
            regime_str = result.get("regime", "ranging")
            regime_map = {
                "trending_bullish": MarketRegime.TRENDING_BULLISH,
                "trending_bearish": MarketRegime.TRENDING_BEARISH,
                "ranging": MarketRegime.RANGING,
                "volatile_choppy": MarketRegime.VOLATILE_CHOPPY,
            }
            self._current_regime = regime_map.get(regime_str, MarketRegime.RANGING)
            self._confidence = float(result.get("confidence", 0.5))
            self._reasoning = result.get("reasoning", "")
            self._last_update = time.time()

            logger.info(
                "regime.classified",
                regime=self._current_regime.value,
                confidence=self._confidence,
                reasoning=self._reasoning[:100],
            )
        except (KeyError, ValueError) as e:
            logger.warning("regime.parse_error", error=str(e))
            return self._technical_fallback(adx_value, ema_alignment, atr_percentile)

        return self._current_regime, self._confidence

    def _technical_fallback(
        self, adx_value: float, ema_alignment: str, atr_percentile: float
    ) -> tuple[MarketRegime, float]:
        """Classify regime using only technical indicators (no LLM)."""
        if atr_percentile > 150:
            self._current_regime = MarketRegime.VOLATILE_CHOPPY
            self._confidence = 0.6
        elif adx_value > 25:
            if ema_alignment == "bullish":
                self._current_regime = MarketRegime.TRENDING_BULLISH
                self._confidence = 0.7
            elif ema_alignment == "bearish":
                self._current_regime = MarketRegime.TRENDING_BEARISH
                self._confidence = 0.7
            else:
                self._current_regime = MarketRegime.RANGING
                self._confidence = 0.4
        elif adx_value < 20:
            self._current_regime = MarketRegime.RANGING
            self._confidence = 0.6
        else:
            self._current_regime = MarketRegime.RANGING
            self._confidence = 0.4

        self._reasoning = f"Technical fallback: ADX={adx_value:.1f}, alignment={ema_alignment}"
        self._last_update = time.time()

        logger.info(
            "regime.technical_fallback",
            regime=self._current_regime.value,
            confidence=self._confidence,
        )
        return self._current_regime, self._confidence

    def _summarize_candles(self, df: pd.DataFrame) -> str:
        """Create a text summary of recent price action."""
        if len(df) == 0:
            return "No data"

        last_n = min(100, len(df))
        recent = df.tail(last_n)

        high = recent["high"].max()
        low = recent["low"].min()
        close = recent["close"].iloc[-1]
        open_first = recent["open"].iloc[0]
        change = close - open_first
        change_pct = (change / open_first) * 100

        return (
            f"Range: {low:.2f} - {high:.2f} ({high - low:.2f} points)\n"
            f"Open: {open_first:.2f}, Close: {close:.2f}, Change: {change:+.2f} ({change_pct:+.2f}%)\n"
            f"Bars analyzed: {last_n}"
        )

    def _summarize_volume(self, df: pd.DataFrame) -> str:
        if len(df) == 0:
            return "No volume data"
        recent = df.tail(20)
        avg_vol = recent["volume"].mean()
        last_vol = recent["volume"].iloc[-1]
        return f"Avg volume (20 bars): {avg_vol:.0f}, Last bar: {last_vol}"

    def _format_intermarket(self, im: IntermarketSnapshot) -> str:
        parts = []
        if im.dxy_price:
            parts.append(f"DXY: {im.dxy_price:.2f} ({im.dxy_change_pct:+.2f}%)" if im.dxy_change_pct else f"DXY: {im.dxy_price:.2f}")
        if im.vix_price:
            parts.append(f"VIX: {im.vix_price:.1f} ({im.vix_level})")
        if im.zn_price:
            parts.append(f"ZN: {im.zn_price:.3f} ({im.zn_change_pct:+.2f}%)" if im.zn_change_pct else f"ZN: {im.zn_price:.3f}")
        return ", ".join(parts) if parts else "No intermarket data"

    def _format_order_flow(self, of: OrderFlowState) -> str:
        parts = [
            f"Cumulative delta: {of.cumulative_delta:.0f} ({of.delta_trend})",
        ]
        if of.absorption_detected:
            parts.append(f"Absorption: {of.absorption_direction.value if of.absorption_direction else 'unknown'}")
        if of.stacked_imbalances:
            parts.append(f"Stacked imbalances: {len(of.stacked_imbalances)} levels ({of.imbalance_direction.value if of.imbalance_direction else 'unknown'})")
        if of.delta_divergence:
            parts.append("Delta divergence detected")
        return ", ".join(parts)
