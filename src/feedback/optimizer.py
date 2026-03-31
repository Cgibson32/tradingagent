"""Walk-forward parameter optimization.

Suggests bounded parameter changes based on recent trade performance.
Never auto-deploys — changes are logged for human or LLM review.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import aiosqlite
import structlog

from src.feedback.analytics import PerformanceAnalytics

logger = structlog.get_logger(__name__)

MAX_PARAM_DELTA = {
    "min_confidence_threshold": 0.1,
    "risk_per_trade_pct": 0.005,
    "trailing_stop_atr_multiple": 0.5,
    "partial_tp_pct": 0.1,
}


class ParameterOptimizer:
    """Suggests bounded parameter adjustments based on trade performance."""

    def __init__(self, db_path: str = "data/trading.db"):
        self._db_path = db_path
        self._analytics = PerformanceAnalytics()

    async def suggest_adjustments(self, trades: list[dict], current_params: dict) -> list[dict]:
        if len(trades) < 20:
            return [{"note": "Insufficient trades for optimization (need 20+)"}]

        metrics = self._analytics.calculate_metrics(trades)
        suggestions = []

        # Analyze confidence buckets
        low_conf_losing = self._check_low_confidence(trades)
        if low_conf_losing:
            current = current_params.get("min_confidence_threshold", 0.65)
            suggestions.append({
                "parameter": "min_confidence_threshold",
                "current": current,
                "suggested": round(min(current + 0.05, current + MAX_PARAM_DELTA["min_confidence_threshold"]), 2),
                "reasoning": "Low-confidence trades are losing — raise threshold",
            })

        if metrics.get("max_drawdown", 0) > metrics.get("total_pnl", 0) * 0.5:
            current = current_params.get("risk_per_trade_pct", 0.01)
            suggestions.append({
                "parameter": "risk_per_trade_pct",
                "current": current,
                "suggested": round(max(current - 0.002, 0.005), 4),
                "reasoning": "Drawdown is high relative to P&L — reduce risk",
            })

        if not suggestions:
            suggestions.append({"note": "No adjustments suggested — parameters performing well"})

        return suggestions

    async def log_adjustment(self, review_type: str, adjustment_text: str, parameters_json: dict | None = None) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO strategy_adjustments (created_at, review_type, adjustment_text, parameters_json, applied)
                   VALUES (?, ?, ?, ?, 0)""",
                (datetime.now(timezone.utc).isoformat(), review_type, adjustment_text, json.dumps(parameters_json or {})),
            )
            await db.commit()

    def _check_low_confidence(self, trades: list[dict]) -> bool:
        low = [t for t in trades if (t.get("signal_confidence") or 0) < 0.7]
        if len(low) >= 5:
            wins = sum(1 for t in low if (t.get("pnl_dollars") or 0) > 0)
            return (wins / len(low)) < 0.4
        return False
