"""Shadow/paper comparison runner for A/B testing strategy variants.

Runs two strategy configurations simultaneously:
- Live: executes real orders
- Shadow: paper-only, logged but not executed

Compares performance to identify improvements before deploying.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from src.feedback.analytics import PerformanceAnalytics

logger = structlog.get_logger(__name__)


class ShadowRunner:
    """A/B testing framework for strategy parameter variants.

    The shadow strategy receives identical market data but uses different
    parameters. Its signals are evaluated and logged but never executed.
    """

    def __init__(self, shadow_params: dict | None = None):
        self._shadow_params = shadow_params or {}
        self._shadow_trades: list[dict] = []
        self._analytics = PerformanceAnalytics()
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def shadow_trades(self) -> list[dict]:
        return self._shadow_trades

    def enable(self, params: dict) -> None:
        """Enable shadow runner with alternative parameters."""
        self._shadow_params = params
        self._enabled = True
        logger.info("shadow_runner.enabled", params=params)

    def disable(self) -> None:
        self._enabled = False
        logger.info("shadow_runner.disabled")

    def record_shadow_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        strategy_variant: str = "shadow",
    ) -> None:
        """Record a shadow signal (would-be trade)."""
        self._shadow_trades.append({
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "signal_confidence": confidence,
            "strategy_name": strategy_variant,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "status": "open",
            "pnl_r": None,
            "pnl_dollars": None,
        })

    def close_shadow_trade(self, index: int, exit_price: float, pnl_r: float, pnl_dollars: float) -> None:
        """Close a shadow trade with its outcome."""
        if 0 <= index < len(self._shadow_trades):
            trade = self._shadow_trades[index]
            trade["exit_price"] = exit_price
            trade["exit_time"] = datetime.now(timezone.utc).isoformat()
            trade["pnl_r"] = pnl_r
            trade["pnl_dollars"] = pnl_dollars
            trade["status"] = "win" if pnl_r > 0 else "loss"

    def compare(self, live_trades: list[dict]) -> dict:
        """Compare shadow vs live performance.

        Returns comparison metrics and a recommendation.
        """
        live_metrics = self._analytics.calculate_metrics(live_trades)
        shadow_metrics = self._analytics.calculate_metrics(
            [t for t in self._shadow_trades if t.get("status") in ("win", "loss")]
        )

        live_wr = live_metrics.get("win_rate", 0)
        shadow_wr = shadow_metrics.get("win_rate", 0)
        live_pf = live_metrics.get("profit_factor", 0)
        shadow_pf = shadow_metrics.get("profit_factor", 0)

        # Determine recommendation
        if shadow_metrics.get("total_trades", 0) < 10:
            recommendation = "insufficient_data"
        elif shadow_pf > live_pf * 1.2 and shadow_wr > live_wr:
            recommendation = "shadow_outperforms"
        elif live_pf > shadow_pf * 1.2:
            recommendation = "live_outperforms"
        else:
            recommendation = "no_significant_difference"

        return {
            "live": live_metrics,
            "shadow": shadow_metrics,
            "shadow_params": self._shadow_params,
            "recommendation": recommendation,
            "summary": (
                f"Live: {live_wr}% WR, {live_pf} PF | "
                f"Shadow: {shadow_wr}% WR, {shadow_pf} PF | "
                f"Recommendation: {recommendation}"
            ),
        }

    def reset(self) -> None:
        """Reset shadow trades for a new comparison period."""
        self._shadow_trades = []
