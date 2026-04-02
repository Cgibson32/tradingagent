"""Signal evaluation engine — Claude approves/rejects/modifies trade signals.

The evaluator is the final AI gate before execution. It receives a
TradeSignal from the aggregator and either approves, rejects, or
modifies it based on full market context.

Key design: LLM is ADVISORY ONLY. Risk guardrails cannot be overridden.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import structlog

from src.brain.claude_client import ClaudeClient
from src.brain.prompts import signal_evaluation_prompt, trade_journal_prompt
from src.data.models import MarketRegime, MarketState, TradeSignal

logger = structlog.get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result from Claude's signal evaluation."""

    decision: str  # "approve", "reject", "modify"
    adjusted_size: int = 1
    adjusted_tp_r: float | None = None
    time_feasibility: str = "feasible"
    confidence: float = 0.0
    reasoning: str = ""
    risk_notes: str = ""
    cautions: list[str] | None = None
    raw_response: dict | None = None
    used_fallback: bool = False


@dataclass
class JournalEntry:
    """Result from Claude's post-trade journal."""

    grade: str = "C"
    entry_quality: str = "fair"
    exit_quality: str = "fair"
    time_management: str = "fair"
    key_lesson: str = ""
    what_went_well: str = ""
    what_to_improve: str = ""
    should_repeat: bool = False
    tags: list[str] | None = None
    raw_response: dict | None = None


class SignalEvaluator:
    """Evaluates trade signals using Claude API.

    Falls back to rule-based approval if Claude is unavailable.
    Never blocks trade exits (stop losses) waiting for LLM response.
    """

    def __init__(self, claude_client: ClaudeClient):
        self._claude = claude_client

    async def evaluate(
        self,
        signal: TradeSignal,
        market_state: MarketState,
        recent_trades_summary: str = "No recent trades",
    ) -> EvaluationResult:
        """Evaluate a trade signal with full market context.

        Args:
            signal: The aggregated trade signal to evaluate
            market_state: Complete market state snapshot
            recent_trades_summary: Text summary of last 5 trades

        Returns:
            EvaluationResult with decision and reasoning
        """
        # Build context strings
        ta_tp = f"{signal.time_adjusted_tp:.2f}" if signal.time_adjusted_tp else "N/A"
        signal_summary = (
            f"Direction: {signal.direction.value}\n"
            f"Entry: {signal.entry_price:.2f}\n"
            f"Stop Loss: {signal.stop_loss:.2f} (risk: {signal.risk_points:.2f} pts)\n"
            f"Take Profit: {signal.take_profit:.2f} ({signal.risk_reward_ratio:.1f}R)\n"
            f"Time-adjusted TP: {ta_tp}\n"
            f"Confidence: {signal.confidence:.0%}\n"
            f"Strategy: {signal.strategy_name}\n"
            f"Contributing signals: {', '.join(s.pattern for s in signal.contributing_signals)}"
        )

        account_state = (
            f"Equity: ${market_state.equity:,.2f}\n"
            f"Daily P&L: ${market_state.daily_pnl:,.2f}\n"
            f"Drawdown floor: ${market_state.drawdown_floor:,.2f}\n"
            f"Trades today: {market_state.trades_today}\n"
            f"Consecutive losses: {market_state.consecutive_losses}"
        )

        kill_zone = market_state.active_kill_zone or "None (outside kill zones)"

        key_levels_proximity = "Near key levels" if market_state.key_levels.prev_day_high > 0 else "No level data"

        of = market_state.order_flow
        order_flow_str = (
            f"Delta: {of.cumulative_delta:.0f} ({of.delta_trend}), "
            f"Absorption: {of.absorption_detected}, "
            f"Imbalances: {len(of.stacked_imbalances)}"
        )

        tp_r = signal.risk_reward_ratio
        if signal.time_adjusted_tp and signal.risk_points > 0:
            tp_r = abs(signal.time_adjusted_tp - signal.entry_price) / signal.risk_points

        system, user = signal_evaluation_prompt(
            signal_summary=signal_summary,
            regime=market_state.regime.value,
            regime_confidence=market_state.regime_confidence,
            account_state=account_state,
            recent_trades=recent_trades_summary,
            kill_zone=kill_zone,
            key_levels_proximity=key_levels_proximity,
            order_flow_confirmation=order_flow_str,
            minutes_until_close=signal.minutes_until_close,
            estimated_trade_duration=signal.estimated_duration_minutes,
            time_adjusted_tp=tp_r,
        )

        result = await self._claude.ask(user, system=system, mode="realtime")

        if result.get("fallback"):
            logger.warning("evaluator.claude_unavailable", error=result.get("error"))
            return self._rule_based_fallback(signal, market_state)

        # Parse response
        try:
            return EvaluationResult(
                decision=result.get("decision", "reject"),
                adjusted_size=int(result.get("adjusted_size", 1)),
                adjusted_tp_r=result.get("adjusted_tp_r"),
                time_feasibility=result.get("time_feasibility", "feasible"),
                confidence=float(result.get("confidence", 0.0)),
                reasoning=result.get("reasoning", ""),
                risk_notes=result.get("risk_notes", ""),
                cautions=result.get("cautions"),
                raw_response=result,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("evaluator.parse_error", error=str(e))
            return self._rule_based_fallback(signal, market_state)

    def _rule_based_fallback(
        self, signal: TradeSignal, state: MarketState
    ) -> EvaluationResult:
        """Simple rule-based approval when Claude is unavailable.

        Conservative: only approve high-confidence signals with good R:R.
        """
        # Reject if time is insufficient
        if signal.minutes_until_close < signal.estimated_duration_minutes * 1.5:
            return EvaluationResult(
                decision="reject",
                reasoning="Insufficient time remaining",
                time_feasibility="reject",
                used_fallback=True,
            )

        # Reject if confidence is borderline
        if signal.confidence < 0.70:
            return EvaluationResult(
                decision="reject",
                reasoning=f"Confidence too low for fallback mode: {signal.confidence:.0%}",
                used_fallback=True,
            )

        # Reject if R:R is too low
        if signal.risk_reward_ratio < 2.0:
            return EvaluationResult(
                decision="reject",
                reasoning=f"R:R too low: {signal.risk_reward_ratio:.1f}",
                used_fallback=True,
            )

        # Reject if too many consecutive losses
        if state.consecutive_losses >= 2:
            return EvaluationResult(
                decision="reject",
                reasoning="Consecutive losses — waiting for conditions to improve",
                used_fallback=True,
            )

        # Approve with conservative sizing
        return EvaluationResult(
            decision="approve",
            adjusted_size=1,  # Minimum size in fallback mode
            confidence=signal.confidence * 0.8,  # Discount confidence without AI
            reasoning="Rule-based approval (Claude unavailable)",
            used_fallback=True,
        )

    async def journal_trade(
        self,
        trade_details: str,
        entry_context: str,
        exit_context: str,
        pnl: float,
        pnl_r: float,
        duration_minutes: float,
    ) -> JournalEntry:
        """Generate a post-trade journal entry using Claude.

        Args:
            trade_details: Text description of the trade
            entry_context: Market conditions at entry
            exit_context: Market conditions at exit
            pnl: P&L in dollars
            pnl_r: P&L in R-multiples
            duration_minutes: Trade duration

        Returns:
            JournalEntry with grade, lessons, and tags
        """
        system, user = trade_journal_prompt(
            trade_details=trade_details,
            entry_context=entry_context,
            exit_context=exit_context,
            pnl=pnl,
            pnl_r=pnl_r,
            duration_minutes=duration_minutes,
        )

        result = await self._claude.ask(user, system=system, mode="realtime")

        if result.get("fallback"):
            return JournalEntry(
                grade="C" if pnl >= 0 else "D",
                key_lesson="Claude unavailable for reflection",
                raw_response=result,
            )

        try:
            return JournalEntry(
                grade=result.get("grade", "C"),
                entry_quality=result.get("entry_quality", "fair"),
                exit_quality=result.get("exit_quality", "fair"),
                time_management=result.get("time_management", "fair"),
                key_lesson=result.get("key_lesson", ""),
                what_went_well=result.get("what_went_well", ""),
                what_to_improve=result.get("what_to_improve", ""),
                should_repeat=result.get("should_repeat_setup", False),
                tags=result.get("tags"),
                raw_response=result,
            )
        except (KeyError, ValueError) as e:
            logger.warning("evaluator.journal_parse_error", error=str(e))
            return JournalEntry(grade="C", raw_response=result)
