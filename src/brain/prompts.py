"""Structured prompt templates for Claude API.

Each prompt returns a system message and user message pair.
All prompts demand JSON output in a specific schema.
"""

from __future__ import annotations

from typing import Any


def regime_classification_prompt(
    candle_summary: str,
    adx_value: float,
    atr_percentile: float,
    ema_alignment: str,
    volume_profile: str,
    intermarket_context: str,
    order_flow_summary: str,
) -> tuple[str, str]:
    """Build the market regime classification prompt.

    Called every 30 minutes or on significant structure change.
    """
    system = (
        "You are an expert NQ futures market analyst. Your job is to classify the current "
        "market regime based on the data provided. You must respond ONLY with valid JSON "
        "matching the exact schema specified. No additional text."
    )

    user = f"""Analyze the following NQ futures market data and classify the current regime.

## Price Action Summary (last 100 candles)
{candle_summary}

## Technical Indicators
- ADX(14): {adx_value:.1f} (>25 = trending, <20 = ranging)
- ATR percentile (vs 20-day): {atr_percentile:.0f}%
- EMA alignment (9/21/50/200): {ema_alignment}

## Volume Profile
{volume_profile}

## Intermarket Context
{intermarket_context}

## Order Flow
{order_flow_summary}

## Respond with this exact JSON schema:
{{
  "regime": "trending_bullish" | "trending_bearish" | "ranging" | "volatile_choppy",
  "confidence": 0.0-1.0,
  "key_observation": "one sentence summary of the most important observation",
  "support_levels": [float],
  "resistance_levels": [float],
  "reasoning": "2-3 sentence explanation"
}}"""

    return system, user


def signal_evaluation_prompt(
    signal_summary: str,
    regime: str,
    regime_confidence: float,
    account_state: str,
    recent_trades: str,
    kill_zone: str,
    key_levels_proximity: str,
    order_flow_confirmation: str,
    minutes_until_close: float,
    estimated_trade_duration: float,
    time_adjusted_tp: float,
) -> tuple[str, str]:
    """Build the signal evaluation prompt.

    Called on every signal that passes the aggregator threshold.
    """
    system = (
        "You are a conservative NQ futures trading advisor for a Tradeify 150K prop firm account. "
        "Capital preservation is priority #1. You must respond ONLY with valid JSON. "
        "CRITICAL RULE: A trade entered with insufficient time to reach its target is a BAD trade "
        "regardless of setup quality. If minutes_until_close < estimated_trade_duration * 1.5, "
        "you MUST reject."
    )

    user = f"""Evaluate this trade signal and decide whether to approve, reject, or modify it.

## Signal
{signal_summary}

## Market Regime
- Current: {regime} (confidence: {regime_confidence:.0%})

## Account State
{account_state}

## Recent Trades (last 5)
{recent_trades}

## Time Context
- Kill zone: {kill_zone}
- Minutes until mandatory 3:45 PM close: {minutes_until_close:.0f}
- Estimated trade duration (avg): {estimated_trade_duration:.0f} minutes
- Time-adjusted TP target: {time_adjusted_tp:.1f}R
- Time feasibility: {"FEASIBLE" if minutes_until_close >= estimated_trade_duration * 1.5 else "INSUFFICIENT"}

## Confluence Factors
- Key levels: {key_levels_proximity}
- Order flow: {order_flow_confirmation}

## Respond with this exact JSON schema:
{{
  "decision": "approve" | "reject" | "modify",
  "adjusted_size": 1-4,
  "adjusted_tp_r": float | null,
  "time_feasibility": "feasible" | "marginal" | "reject",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentences",
  "risk_notes": "any specific risks to watch",
  "cautions": ["list of concerns"]
}}"""

    return system, user


def trade_journal_prompt(
    trade_details: str,
    entry_context: str,
    exit_context: str,
    pnl: float,
    pnl_r: float,
    duration_minutes: float,
) -> tuple[str, str]:
    """Build the post-trade journal prompt.

    Called on every trade close for reflection and learning.
    """
    system = (
        "You are a trading coach reviewing a completed NQ futures trade. "
        "Be honest and constructive. Focus on what can be improved. "
        "Respond ONLY with valid JSON."
    )

    user = f"""Review this completed trade and provide a journal entry.

## Trade Details
{trade_details}

## Market Context at Entry
{entry_context}

## Market Context at Exit
{exit_context}

## Outcome
- P&L: ${pnl:.2f} ({pnl_r:.2f}R)
- Duration: {duration_minutes:.0f} minutes

## Respond with this exact JSON schema:
{{
  "grade": "A" | "B" | "C" | "D" | "F",
  "entry_quality": "excellent" | "good" | "fair" | "poor",
  "exit_quality": "excellent" | "good" | "fair" | "poor",
  "time_management": "excellent" | "good" | "fair" | "poor",
  "key_lesson": "one sentence lesson to remember",
  "what_went_well": "brief description",
  "what_to_improve": "brief description",
  "should_repeat_setup": true | false,
  "tags": ["momentum", "reversal", "trend_following", etc.]
}}"""

    return system, user


def weekly_review_prompt(
    trades_summary: str,
    aggregate_metrics: str,
    equity_curve_description: str,
    regime_breakdown: str,
    shadow_comparison: str | None = None,
) -> tuple[str, str]:
    """Build the weekly strategy review prompt.

    Called every Sunday. Uses Opus for deeper analysis.
    """
    system = (
        "You are a senior trading strategist conducting a weekly performance review "
        "of an NQ futures trading agent. Provide actionable insights. "
        "Respond ONLY with valid JSON."
    )

    shadow_section = ""
    if shadow_comparison:
        shadow_section = f"\n## Shadow Strategy Comparison\n{shadow_comparison}\n"

    user = f"""Conduct a weekly review of this NQ futures trading agent's performance.

## This Week's Trades
{trades_summary}

## Aggregate Metrics
{aggregate_metrics}

## Equity Curve
{equity_curve_description}

## Regime Breakdown
{regime_breakdown}
{shadow_section}
## Respond with this exact JSON schema:
{{
  "performance_summary": "3-4 sentence overview",
  "win_rate_trend": "improving" | "declining" | "stable",
  "patterns_noticed": ["pattern 1", "pattern 2", ...],
  "best_performing_setup": "description",
  "worst_performing_setup": "description",
  "parameter_adjustments": [
    {{"parameter": "name", "current": value, "suggested": value, "reasoning": "why"}}
  ],
  "strategy_notes": "forward-looking guidance for next week",
  "risk_management_notes": "any concerns about risk"
}}"""

    return system, user
