"""LLM-powered weekly strategy review.

Every Sunday, compiles the week's trading data and sends to Claude Opus
for deep analysis. Results stored for longitudinal learning.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import aiosqlite
import structlog

from src.brain.claude_client import ClaudeClient
from src.brain.prompts import weekly_review_prompt
from src.feedback.analytics import PerformanceAnalytics

logger = structlog.get_logger(__name__)


class WeeklyReviewer:
    """Conducts weekly performance reviews using Claude Opus."""

    def __init__(self, claude_client: ClaudeClient, db_path: str = "data/trading.db"):
        self._claude = claude_client
        self._db_path = db_path
        self._analytics = PerformanceAnalytics()

    async def run_review(self, trades: list[dict], shadow_trades: list[dict] | None = None) -> dict:
        """Run the weekly review and return Claude's analysis.

        Args:
            trades: This week's trades
            shadow_trades: Shadow strategy trades for comparison (optional)

        Returns:
            Claude's review as a dict
        """
        if not trades:
            return {"summary": "No trades this week."}

        # Calculate metrics
        metrics = self._analytics.calculate_metrics(trades)
        regime_breakdown = self._analytics.segment_by_regime(trades)

        # Build context strings
        trades_summary = self._format_trades(trades)
        metrics_summary = json.dumps(metrics, indent=2)
        equity_desc = self._describe_equity_curve(trades)
        regime_desc = json.dumps(
            {k: {"trades": v.get("total_trades", 0), "win_rate": v.get("win_rate", 0)}
             for k, v in regime_breakdown.items()},
            indent=2,
        )

        shadow_comparison = None
        if shadow_trades:
            shadow_metrics = self._analytics.calculate_metrics(shadow_trades)
            shadow_comparison = (
                f"Shadow strategy: {shadow_metrics.get('total_trades', 0)} trades, "
                f"{shadow_metrics.get('win_rate', 0)}% win rate, "
                f"{shadow_metrics.get('total_pnl_r', 0)}R total P&L\n"
                f"Live strategy: {metrics.get('total_trades', 0)} trades, "
                f"{metrics.get('win_rate', 0)}% win rate, "
                f"{metrics.get('total_pnl_r', 0)}R total P&L"
            )

        system, user = weekly_review_prompt(
            trades_summary=trades_summary,
            aggregate_metrics=metrics_summary,
            equity_curve_description=equity_desc,
            regime_breakdown=regime_desc,
            shadow_comparison=shadow_comparison,
        )

        result = await self._claude.ask(user, system=system, mode="analysis")

        # Save review to database
        await self._save_review(result)

        logger.info("weekly_review.complete", trades=len(trades))
        return result

    def _format_trades(self, trades: list[dict]) -> str:
        lines = []
        for i, t in enumerate(trades[:20], 1):  # Cap at 20 for prompt size
            lines.append(
                f"{i}. {t.get('direction', '?')} {t.get('symbol', '?')} | "
                f"Entry: {t.get('entry_price', '?')} | "
                f"P&L: {t.get('pnl_r', '?')}R (${t.get('pnl_dollars', '?')}) | "
                f"Status: {t.get('status', '?')} | "
                f"Strategy: {t.get('strategy_name', '?')}"
            )
        if len(trades) > 20:
            lines.append(f"... and {len(trades) - 20} more trades")
        return "\n".join(lines)

    def _describe_equity_curve(self, trades: list[dict]) -> str:
        pnls = [t.get("pnl_dollars", 0) for t in trades if t.get("pnl_dollars") is not None]
        if not pnls:
            return "No P&L data"

        cumulative = []
        total = 0
        for p in pnls:
            total += p
            cumulative.append(total)

        peak = max(cumulative)
        trough = min(cumulative)
        final = cumulative[-1]

        return (
            f"Starting: $0, Peak: ${peak:,.2f}, Trough: ${trough:,.2f}, Final: ${final:,.2f}. "
            f"{'Positive week' if final > 0 else 'Negative week'}."
        )

    async def _save_review(self, review: dict) -> None:
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """INSERT INTO strategy_adjustments
                       (created_at, review_type, adjustment_text, parameters_json, applied)
                       VALUES (?, 'weekly', ?, ?, 0)""",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        review.get("performance_summary", ""),
                        json.dumps(review.get("parameter_adjustments", [])),
                    ),
                )
                await db.commit()
        except Exception as e:
            logger.warning("weekly_review.save_failed", error=str(e))
