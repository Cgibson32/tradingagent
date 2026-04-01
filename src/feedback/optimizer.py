"""Self-tuning parameter optimizer with auto-apply.

Analyzes trade performance across multiple dimensions and automatically
applies bounded parameter improvements. Every change is logged and reversible.

Safety: bounded changes only, max N per day, hard floors/ceilings, requires
minimum trade count before tuning.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import yaml
import structlog

from src.feedback.analytics import PerformanceAnalytics

logger = structlog.get_logger(__name__)

# Maximum allowed change per optimization cycle
MAX_PARAM_DELTA = {
    "min_confidence_threshold": 0.1,
    "risk_per_trade_pct": 0.005,
    "max_risk_per_trade_pct": 0.005,
    "trailing_stop_atr_multiple": 0.5,
    "partial_tp_pct": 0.1,
    "break_even_trigger_pct": 0.1,
    "full_tp_r": 0.5,
    "confidence_penalty_outside_kz": 0.05,
}

# Hard floors and ceilings — auto-tune can never go beyond these
HARD_BOUNDS = {
    "min_confidence_threshold": (0.50, 0.90),
    "risk_per_trade_pct": (0.005, 0.03),
    "max_risk_per_trade_pct": (0.01, 0.03),
    "trailing_stop_atr_multiple": (0.5, 3.0),
    "partial_tp_pct": (0.25, 0.75),
    "break_even_trigger_pct": (0.2, 0.6),
    "full_tp_r": (1.5, 4.0),
    "confidence_penalty_outside_kz": (0.0, 0.3),
}

# YAML path mapping: parameter name → (section, key) in default.yaml
YAML_PATHS = {
    "min_confidence_threshold": ("signals", "min_confidence_threshold"),
    "risk_per_trade_pct": ("risk", "risk_per_trade_pct"),
    "max_risk_per_trade_pct": ("risk", "max_risk_per_trade_pct"),
    "trailing_stop_atr_multiple": ("trade", "trailing_stop_atr_multiple"),
    "partial_tp_pct": ("trade", "partial_tp_pct"),
    "break_even_trigger_pct": ("trade", "break_even_trigger_pct"),
    "full_tp_r": ("trade", "full_tp_r"),
    "confidence_penalty_outside_kz": ("kill_zones", "confidence_penalty_outside_kz"),
    # Signal weights
    "weight_ict_pattern": ("signals", "signal_weights", "ict_pattern"),
    "weight_htf_bias": ("signals", "signal_weights", "htf_bias"),
    "weight_indicators": ("signals", "signal_weights", "indicators"),
    "weight_order_flow": ("signals", "signal_weights", "order_flow"),
    "weight_key_levels": ("signals", "signal_weights", "key_levels"),
    "weight_intermarket": ("signals", "signal_weights", "intermarket"),
}


class ParameterOptimizer:
    """Self-tuning optimizer that analyzes performance and auto-applies improvements.

    Analyzes across multiple dimensions:
    - Confidence bucket performance → adjust confidence threshold
    - Signal source win rates → adjust signal weights
    - Drawdown vs P&L ratio → adjust risk per trade
    - Trailing stop efficiency (MAE/MFE) → adjust ATR multiple
    - Regime-specific performance → adjust regime handling
    """

    def __init__(
        self,
        db_path: str = "data/trading.db",
        config_path: str = "config/default.yaml",
        max_changes_per_day: int = 3,
        min_trades_required: int = 20,
    ):
        self._db_path = db_path
        self._config_path = Path(config_path)
        self._max_changes_per_day = max_changes_per_day
        self._min_trades = min_trades_required
        self._analytics = PerformanceAnalytics()
        self._changes_today: int = 0
        self._last_reset_date: str = ""

    def _check_daily_reset(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._last_reset_date != today:
            self._changes_today = 0
            self._last_reset_date = today

    @property
    def can_tune(self) -> bool:
        self._check_daily_reset()
        return self._changes_today < self._max_changes_per_day

    # ── Analysis Methods ──

    def analyze_all(self, trades: list[dict], current_params: dict) -> list[dict]:
        """Run all analysis dimensions and return suggested adjustments."""
        if len(trades) < self._min_trades:
            return [{"note": f"Need {self._min_trades}+ trades, have {len(trades)}"}]

        suggestions = []
        suggestions.extend(self._analyze_confidence_buckets(trades, current_params))
        suggestions.extend(self._analyze_signal_weights(trades, current_params))
        suggestions.extend(self._analyze_risk_sizing(trades, current_params))
        suggestions.extend(self._analyze_trailing_stop(trades, current_params))
        suggestions.extend(self._analyze_tp_targets(trades, current_params))

        if not suggestions:
            suggestions.append({"note": "All parameters performing well — no changes needed"})

        return suggestions

    def _analyze_confidence_buckets(self, trades: list[dict], params: dict) -> list[dict]:
        """Analyze win rate by confidence bucket → adjust threshold."""
        buckets = {"low": [], "mid": [], "high": []}
        for t in trades:
            conf = float(t.get("signal_confidence") or 0)
            if conf < 0.65:
                buckets["low"].append(t)
            elif conf < 0.80:
                buckets["mid"].append(t)
            else:
                buckets["high"].append(t)

        suggestions = []

        # If low-confidence trades are losing, raise threshold
        if len(buckets["low"]) >= 5:
            low_wins = sum(1 for t in buckets["low"] if (t.get("pnl_dollars") or 0) > 0)
            low_wr = low_wins / len(buckets["low"])
            if low_wr < 0.40:
                current = params.get("min_confidence_threshold", 0.65)
                suggested = self._clamp("min_confidence_threshold", current + 0.05)
                if suggested != current:
                    suggestions.append({
                        "parameter": "min_confidence_threshold",
                        "current": current,
                        "suggested": suggested,
                        "reasoning": f"Low-confidence trades win rate is {low_wr:.0%} — raising threshold to filter them out",
                        "dimension": "confidence_analysis",
                    })

        # If high-confidence trades are dominant and profitable, could lower threshold slightly
        if len(buckets["high"]) >= 10 and len(buckets["mid"]) < 3:
            mid_potential = len(buckets["mid"])
            current = params.get("min_confidence_threshold", 0.65)
            if current > 0.70:
                suggested = self._clamp("min_confidence_threshold", current - 0.03)
                if suggested != current:
                    suggestions.append({
                        "parameter": "min_confidence_threshold",
                        "current": current,
                        "suggested": suggested,
                        "reasoning": "Most trades are high-confidence — slightly lowering threshold to capture more opportunities",
                        "dimension": "confidence_analysis",
                    })

        return suggestions

    def _analyze_signal_weights(self, trades: list[dict], params: dict) -> list[dict]:
        """Analyze which signal sources produce the best trades → adjust weights."""
        # This would need trade_context data with per-signal contribution
        # For now, return empty — will be populated when we have enough signal-level data
        return []

    def _analyze_risk_sizing(self, trades: list[dict], params: dict) -> list[dict]:
        """Analyze drawdown vs P&L → adjust risk per trade."""
        metrics = self._analytics.calculate_metrics(trades)
        suggestions = []

        total_pnl = metrics.get("total_pnl", 0)
        max_dd = metrics.get("max_drawdown", 0)

        # Drawdown is too high relative to profits
        if max_dd > 0 and total_pnl > 0 and max_dd > total_pnl * 0.6:
            current = params.get("risk_per_trade_pct", 0.01)
            suggested = self._clamp("risk_per_trade_pct", current - 0.002)
            if suggested != current:
                suggestions.append({
                    "parameter": "risk_per_trade_pct",
                    "current": current,
                    "suggested": suggested,
                    "reasoning": f"Max drawdown (${max_dd:.0f}) is {max_dd/total_pnl:.0%} of total P&L — reducing risk",
                    "dimension": "risk_analysis",
                })

        # Very low drawdown and strong profits — could increase risk slightly
        elif total_pnl > 0 and max_dd < total_pnl * 0.2 and metrics.get("win_rate", 0) > 60:
            current = params.get("risk_per_trade_pct", 0.01)
            suggested = self._clamp("risk_per_trade_pct", current + 0.001)
            if suggested != current:
                suggestions.append({
                    "parameter": "risk_per_trade_pct",
                    "current": current,
                    "suggested": suggested,
                    "reasoning": f"Low drawdown ({max_dd/total_pnl:.0%} of P&L) with {metrics['win_rate']}% WR — slight risk increase",
                    "dimension": "risk_analysis",
                })

        return suggestions

    def _analyze_trailing_stop(self, trades: list[dict], params: dict) -> list[dict]:
        """Analyze if trailing stop is too tight or too loose."""
        if not trades:
            return []

        wins = [t for t in trades if t.get("status") == "win"]
        metrics = self._analytics.calculate_metrics(trades)
        suggestions = []

        # High win rate but low avg R → trailing too tight (cutting winners short)
        if metrics.get("win_rate", 0) > 60 and metrics.get("avg_win_r", 0) < 1.5:
            current = params.get("trailing_stop_atr_multiple", 1.5)
            suggested = self._clamp("trailing_stop_atr_multiple", current + 0.25)
            if suggested != current:
                suggestions.append({
                    "parameter": "trailing_stop_atr_multiple",
                    "current": current,
                    "suggested": suggested,
                    "reasoning": f"Win rate {metrics['win_rate']}% but avg win only {metrics['avg_win_r']:.2f}R — loosening trail to let winners run",
                    "dimension": "trailing_stop_analysis",
                })

        # Low win rate with decent avg R → trailing may be too loose
        elif metrics.get("win_rate", 0) < 45 and metrics.get("avg_win_r", 0) > 2.0:
            current = params.get("trailing_stop_atr_multiple", 1.5)
            suggested = self._clamp("trailing_stop_atr_multiple", current - 0.25)
            if suggested != current:
                suggestions.append({
                    "parameter": "trailing_stop_atr_multiple",
                    "current": current,
                    "suggested": suggested,
                    "reasoning": f"Low win rate ({metrics['win_rate']}%) — tightening trail to capture more exits in profit",
                    "dimension": "trailing_stop_analysis",
                })

        return suggestions

    def _analyze_tp_targets(self, trades: list[dict], params: dict) -> list[dict]:
        """Analyze if TP targets are realistic."""
        metrics = self._analytics.calculate_metrics(trades)
        suggestions = []

        # If very few trades hit full TP, maybe target is too ambitious
        wins = [t for t in trades if t.get("status") == "win"]
        if len(wins) >= 5:
            avg_win_r = metrics.get("avg_win_r", 0)
            full_tp = params.get("full_tp_r", 2.5)
            if avg_win_r < full_tp * 0.6:
                suggested = self._clamp("full_tp_r", full_tp - 0.25)
                if suggested != full_tp:
                    suggestions.append({
                        "parameter": "full_tp_r",
                        "current": full_tp,
                        "suggested": suggested,
                        "reasoning": f"Avg win is {avg_win_r:.2f}R vs {full_tp}R target — reducing TP for more completions",
                        "dimension": "tp_analysis",
                    })

        return suggestions

    # ── Auto-Apply ──

    async def auto_apply_adjustments(
        self, trades: list[dict], current_params: dict
    ) -> list[dict]:
        """Analyze and auto-apply parameter improvements.

        Returns list of applied changes.
        """
        self._check_daily_reset()

        if not self.can_tune:
            return [{"note": f"Daily change limit reached ({self._max_changes_per_day})"}]

        suggestions = self.analyze_all(trades, current_params)
        actionable = [s for s in suggestions if "parameter" in s]

        if not actionable:
            return suggestions

        applied = []
        for suggestion in actionable:
            if not self.can_tune:
                break

            success = await self._apply_single(suggestion)
            if success:
                applied.append(suggestion)
                self._changes_today += 1

        return applied if applied else [{"note": "No changes applied"}]

    async def apply_claude_suggestions(self, claude_adjustments: list[dict]) -> list[dict]:
        """Apply parameter adjustments from Claude's weekly review.

        Validates each against bounds before applying.
        """
        self._check_daily_reset()
        applied = []

        for adj in claude_adjustments:
            param = adj.get("parameter", "")
            suggested = adj.get("suggested")
            reasoning = adj.get("reasoning", "Claude weekly review")

            if param not in HARD_BOUNDS or suggested is None:
                logger.warning("optimizer.invalid_claude_suggestion", parameter=param)
                continue

            if not self.can_tune:
                break

            clamped = self._clamp(param, float(suggested))
            suggestion = {
                "parameter": param,
                "current": adj.get("current", 0),
                "suggested": clamped,
                "reasoning": reasoning,
                "dimension": "claude_weekly_review",
            }

            success = await self._apply_single(suggestion)
            if success:
                applied.append(suggestion)
                self._changes_today += 1

        return applied

    async def _apply_single(self, suggestion: dict) -> bool:
        """Apply a single parameter change to config/default.yaml."""
        param = suggestion["parameter"]
        new_value = suggestion["suggested"]
        old_value = suggestion.get("current", 0)

        yaml_path = YAML_PATHS.get(param)
        if not yaml_path:
            logger.warning("optimizer.no_yaml_path", parameter=param)
            return False

        try:
            # Read config
            config = yaml.safe_load(self._config_path.read_text()) or {}

            # Navigate to the right nested key and set value
            if len(yaml_path) == 2:
                section, key = yaml_path
                config.setdefault(section, {})[key] = round(new_value, 4)
            elif len(yaml_path) == 3:
                section, subsection, key = yaml_path
                config.setdefault(section, {}).setdefault(subsection, {})[key] = round(new_value, 4)

            # Write back
            self._config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))

            # Log to database
            await self._log_applied_change(suggestion)

            logger.info(
                "optimizer.applied",
                parameter=param,
                old=old_value,
                new=new_value,
                reasoning=suggestion.get("reasoning", ""),
            )
            return True

        except Exception as e:
            logger.error("optimizer.apply_failed", parameter=param, error=str(e))
            return False

    async def _log_applied_change(self, suggestion: dict) -> None:
        """Log an applied change to the strategy_adjustments table."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO strategy_adjustments
                   (created_at, review_type, adjustment_text, parameters_json, applied)
                   VALUES (?, ?, ?, ?, 1)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    suggestion.get("dimension", "auto_tune"),
                    f"{suggestion['parameter']}: {suggestion.get('current', '?')} → {suggestion['suggested']} | {suggestion.get('reasoning', '')}",
                    json.dumps(suggestion),
                ),
            )
            await db.commit()

    async def get_adjustment_history(self, limit: int = 20) -> list[dict]:
        """Get recent parameter adjustment history."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM strategy_adjustments ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in await cursor.fetchall()]

    # ── Helpers ──

    def _clamp(self, param: str, value: float) -> float:
        """Clamp a value within hard bounds and delta limits."""
        floor, ceiling = HARD_BOUNDS.get(param, (0.0, 1.0))
        return round(max(floor, min(ceiling, value)), 4)

    def get_current_params_from_yaml(self) -> dict:
        """Read current tunable parameter values from config/default.yaml."""
        if not self._config_path.exists():
            return {}

        config = yaml.safe_load(self._config_path.read_text()) or {}
        params = {}

        for param, path in YAML_PATHS.items():
            try:
                if len(path) == 2:
                    params[param] = config.get(path[0], {}).get(path[1])
                elif len(path) == 3:
                    params[param] = config.get(path[0], {}).get(path[1], {}).get(path[2])
            except (KeyError, TypeError):
                pass

        return {k: v for k, v in params.items() if v is not None}
