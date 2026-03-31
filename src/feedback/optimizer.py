"""Walk-forward parameter optimization.

Optimizes strategy parameters (confidence threshold, indicator weights,
ATR multiplier, etc.) using bounded changes. Monthly frequency.
Never auto-deploys — presents suggestions for review.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import aiosqlite
import structlog

logger = structlog.get_logger(__name__)

# Maximum change per optimization cycle (prevents wild swings)
MAX_PARAM_CHANGE = 0.1


class ParameterOptimizer:
    """Walk-forward parameter optimizer.

    Analyzes recent trade performance and suggests bounded parameter
    adjustments. All changes are logged and reversible.
    """

    def __init__(self, db_path: str = "data/trading.db"):
        self._db_path = db_path

    async def suggest_adjustments(self, trades: list[dict], current_params: dict) -> list[dict]:
        """Analyze trades and suggest parameter adjustments.

        Args:
            trades: Recent trade history
            current_params: Current parameter values

        Returns:
            List of suggested adjustments with reasoning.
        """
        if len(trades) < 10:
            return [{"message": "Insufficient trades for optimization (need 10+)"}]

        suggestions = []

        # Analyze win rate by confidence bucket
        conf_analysis = self._analyze_confidence_buckets(trades)
        if conf_analysis:
            suggestions.append(conf_analysis)

        # Analyze R:R outcomes
        rr_analysis = self._analyze_rr_outcomes(trades, current_params)
        if rr_analysis:
            suggestions.append(rr_analysis)

        # Analyze performance by time of day
        time_analysis = self._analyze_time_performance(trades)
        if time_analysis:
            suggestions.append(time_analysis)

        return suggestions

    def _analyze_confidence_buckets(self, trades: list[dict]) -> dict | None:
        """Check if the confidence threshold should be adjusted."""
        buckets = {"low": [], "mid": [], "high": []}

        for t in trades:
            conf = t.get("signal_confidence", 0.65)
            if conf is None:
                continue
            if conf < 0.70:
                buckets["low"].append(t)
            elif conf < 0.80:
                buckets["mid"].append(t)
            else:
                buckets["high"].append(t)

        # Check if low-confidence trades are losing
        low_trades = buckets["low"]
        if len(low_trades) >= 3:
            low_wins = sum(1 for t in low_trades if (t.get("pnl_r") or 0) > 0)
            low_wr = low_wins / len(low_trades)
            if low_wr < 0.40:
                return {
                    "parameter": "min_confidence_threshold",
                    "current": 0.65,
                    "suggested": min(0.65 + MAX_PARAM_CHANGE, 0.75),
                    "reasoning": f"Low-confidence trades ({len(low_trades)}) have {low_wr:.0%} win rate — raise threshold",
                }

        return None

    def _analyze_rr_outcomes(self, trades: list[dict], params: dict) -> dict | None:
        """Check if R:R targets are realistic."""
        closed = [t for t in trades if t.get("status") in ("win", "loss")]
        if not closed:
            return None

        wins = [t for t in closed if (t.get("pnl_r") or 0) > 0]
        if not wins:
            return None

        avg_win_r = sum(t.get("pnl_r", 0) for t in wins) / len(wins)
        current_tp_r = params.get("full_tp_r", 2.5)

        # If average win is significantly less than target, consider reducing TP
        if avg_win_r < current_tp_r * 0.6 and len(wins) >= 5:
            suggested = max(current_tp_r - MAX_PARAM_CHANGE, 1.5)
            return {
                "parameter": "full_tp_r",
                "current": current_tp_r,
                "suggested": suggested,
                "reasoning": f"Avg win is {avg_win_r:.1f}R vs {current_tp_r}R target — wins aren't reaching TP",
            }

        return None

    def _analyze_time_performance(self, trades: list[dict]) -> dict | None:
        """Check if certain times of day should be avoided."""
        time_buckets: dict[str, list] = {}

        for t in trades:
            entry = t.get("entry_time", "")
            if not entry:
                continue
            try:
                dt = datetime.fromisoformat(entry) if isinstance(entry, str) else entry
                bucket = "morning" if dt.hour < 12 else "afternoon"
            except (ValueError, AttributeError):
                continue

            if bucket not in time_buckets:
                time_buckets[bucket] = []
            time_buckets[bucket].append(t)

        if "afternoon" in time_buckets and len(time_buckets["afternoon"]) >= 5:
            pm_trades = time_buckets["afternoon"]
            pm_wins = sum(1 for t in pm_trades if (t.get("pnl_r") or 0) > 0)
            pm_wr = pm_wins / len(pm_trades)

            if pm_wr < 0.35:
                return {
                    "parameter": "no_entry_cutoff",
                    "current": "14:30",
                    "suggested": "14:00",
                    "reasoning": f"Afternoon trades ({len(pm_trades)}) have only {pm_wr:.0%} win rate — consider earlier cutoff",
                }

        return None

    async def save_suggestion(self, suggestion: dict, review_type: str = "monthly") -> None:
        """Save an optimization suggestion to the database."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO strategy_adjustments
                   (created_at, review_type, adjustment_text, parameters_json, applied)
                   VALUES (?, ?, ?, ?, 0)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    review_type,
                    suggestion.get("reasoning", ""),
                    json.dumps(suggestion),
                ),
            )
            await db.commit()
