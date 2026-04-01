"""LLM-powered weekly strategy review with auto-tuning integration.

Every Sunday, compiles the week's trades and sends to Claude Opus
for deep analysis. If auto-tune is enabled, applies Claude's
parameter suggestions through the optimizer.
"""

from __future__ import annotations

import json

import structlog

from src.brain.claude_client import ClaudeClient
from src.brain.prompts import weekly_review_prompt
from src.feedback.analytics import PerformanceAnalytics
from src.feedback.optimizer import ParameterOptimizer

logger = structlog.get_logger(__name__)


class WeeklyReviewer:
    """Conducts weekly performance reviews using Claude Opus.

    When auto-tune is enabled, feeds Claude's parameter_adjustments
    to the optimizer for validation and application.
    """

    def __init__(
        self,
        claude_client: ClaudeClient,
        optimizer: ParameterOptimizer | None = None,
        analytics: PerformanceAnalytics | None = None,
        auto_tune_enabled: bool = False,
    ):
        self._claude = claude_client
        self._optimizer = optimizer
        self._analytics = analytics or PerformanceAnalytics()
        self._auto_tune = auto_tune_enabled

    async def conduct_review(
        self,
        trades: list[dict],
        shadow_trades: list[dict] | None = None,
    ) -> dict:
        """Conduct weekly review and optionally auto-apply improvements."""
        if not trades:
            return {"error": "No trades this week"}

        metrics = self._analytics.calculate_metrics(trades)
        metrics_text = self._analytics.format_summary(metrics)

        trades_summary = "\n".join(
            f"{i}. {t.get('direction', '?').upper()} | {t.get('status', '?')} | "
            f"${(t.get('pnl_dollars') or 0):+.2f} ({(t.get('pnl_r') or 0):+.2f}R) | "
            f"Strategy: {t.get('strategy_name', '?')} | Regime: {t.get('regime', '?')}"
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
            f"- {r}: {m.get('total_trades', 0)} trades, WR {m.get('win_rate', 0)}%, PF {m.get('profit_factor', 0)}"
            for r, m in regime_breakdown.items()
        ) or "No regime data"

        shadow_text = None
        if shadow_trades:
            sm = self._analytics.calculate_metrics(shadow_trades)
            shadow_text = f"Shadow: {sm.get('total_trades', 0)} trades, WR {sm.get('win_rate', 0)}%, PF {sm.get('profit_factor', 0)}"

        # Include current params and adjustment history for Claude's context
        current_params_text = ""
        adjustment_history_text = ""
        if self._optimizer:
            current_params = self._optimizer.get_current_params_from_yaml()
            if current_params:
                current_params_text = "\n## Current Parameter Values\n" + "\n".join(
                    f"- {k}: {v}" for k, v in current_params.items()
                )

            history = await self._optimizer.get_adjustment_history(5)
            if history:
                adjustment_history_text = "\n## Recent Parameter Adjustments\n" + "\n".join(
                    f"- [{h.get('created_at', '')[:10]}] {h.get('adjustment_text', '')}"
                    for h in history
                )

        system, user = weekly_review_prompt(
            trades_summary, metrics_text, equity_desc, regime_text, shadow_text
        )

        # Append params context to the user prompt
        if current_params_text:
            user += current_params_text
        if adjustment_history_text:
            user += adjustment_history_text

        result = await self._claude.ask(user, system=system, mode="analysis")

        if result.get("fallback"):
            return {"performance_summary": metrics_text, "error": "Claude unavailable", "metrics": metrics}

        result["metrics"] = metrics

        # Auto-apply Claude's parameter suggestions if enabled
        applied_changes = []
        if self._auto_tune and self._optimizer:
            claude_adjustments = result.get("parameter_adjustments", [])
            if claude_adjustments and isinstance(claude_adjustments, list):
                applied_changes = await self._optimizer.apply_claude_suggestions(claude_adjustments)
                result["auto_applied"] = applied_changes
                logger.info(
                    "weekly_review.auto_tuned",
                    suggestions=len(claude_adjustments),
                    applied=len([c for c in applied_changes if "parameter" in c]),
                )

        logger.info("weekly_review.complete", trades=len(trades))
        return result
