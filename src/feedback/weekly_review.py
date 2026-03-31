"""LLM-powered weekly strategy review.

Every Sunday, compiles the week's trades and sends to Claude Opus
for deep analysis. Stores reviews for longitudinal analysis.
"""

from __future__ import annotations

import structlog

from src.brain.claude_client import ClaudeClient
from src.brain.prompts import weekly_review_prompt
from src.feedback.analytics import PerformanceAnalytics

logger = structlog.get_logger(__name__)


class WeeklyReviewer:
    """Conducts weekly performance reviews using Claude Opus."""

    def __init__(self, claude_client: ClaudeClient, analytics: PerformanceAnalytics | None = None):
        self._claude = claude_client
        self._analytics = analytics or PerformanceAnalytics()

    async def conduct_review(self, trades: list[dict], shadow_trades: list[dict] | None = None) -> dict:
        if not trades:
            return {"error": "No trades this week"}

        metrics = self._analytics.calculate_metrics(trades)
        metrics_text = self._analytics.format_summary(metrics)

        trades_summary = "\n".join(
            f"{i}. {t.get('direction','?').upper()} | {t.get('status','?')} | "
            f"${(t.get('pnl_dollars') or 0):+.2f} ({(t.get('pnl_r') or 0):+.2f}R) | "
            f"Strategy: {t.get('strategy_name','?')} | Regime: {t.get('regime','?')}"
            for i, t in enumerate(trades[:30], 1)
        ) or "No trades"

        pnls = [t.get("pnl_dollars", 0) or 0 for t in trades if t.get("status") != "open"]
        cum = []
        total = 0
        for p in pnls:
            total += p
            cum.append(total)
        equity_desc = f"Final: ${total:.2f}, Peak: ${max(cum):.2f}, Trough: ${min(cum):.2f}" if cum else "No data"

        regime_breakdown = self._analytics.segment_by(trades, "regime")
        regime_text = "\n".join(
            f"- {r}: {m.get('total_trades',0)} trades, WR {m.get('win_rate',0)}%, PF {m.get('profit_factor',0)}"
            for r, m in regime_breakdown.items()
        ) or "No regime data"

        shadow_text = None
        if shadow_trades:
            sm = self._analytics.calculate_metrics(shadow_trades)
            shadow_text = f"Shadow: {sm.get('total_trades',0)} trades, WR {sm.get('win_rate',0)}%, PF {sm.get('profit_factor',0)}"

        system, user = weekly_review_prompt(trades_summary, metrics_text, equity_desc, regime_text, shadow_text)
        result = await self._claude.ask(user, system=system, mode="analysis")

        if result.get("fallback"):
            return {"performance_summary": metrics_text, "error": "Claude unavailable", "metrics": metrics}

        result["metrics"] = metrics
        logger.info("weekly_review.complete", trades=len(trades))
        return result
