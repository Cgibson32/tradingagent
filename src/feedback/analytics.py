"""Performance analytics engine.

Calculates comprehensive metrics from trade history:
- Per-trade: R-multiple, duration, MAE, MFE
- Rolling: win rate, avg R, profit factor, Sharpe, Sortino
- Segmented: by regime, time of day, day of week, signal type
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class PerformanceAnalytics:
    """Calculates and tracks performance metrics from trade history."""

    def calculate_metrics(self, trades: list[dict]) -> dict:
        """Calculate comprehensive metrics from a list of trade dicts.

        Args:
            trades: List of trade dicts with keys: pnl_r, pnl_dollars,
                   entry_time, exit_time, direction, status, etc.

        Returns:
            Dict of performance metrics.
        """
        if not trades:
            return {"total_trades": 0, "message": "No trades to analyze"}

        df = pd.DataFrame(trades)

        # Filter to closed trades only
        closed = df[df["status"].isin(["win", "loss", "breakeven"])].copy()
        if len(closed) == 0:
            return {"total_trades": 0, "message": "No closed trades"}

        # Ensure numeric types
        closed["pnl_r"] = pd.to_numeric(closed["pnl_r"], errors="coerce").fillna(0)
        closed["pnl_dollars"] = pd.to_numeric(closed["pnl_dollars"], errors="coerce").fillna(0)

        wins = closed[closed["pnl_r"] > 0]
        losses = closed[closed["pnl_r"] <= 0]

        total = len(closed)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total if total > 0 else 0

        avg_win = wins["pnl_r"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["pnl_r"].mean()) if len(losses) > 0 else 0

        gross_profit = wins["pnl_dollars"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl_dollars"].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Equity curve and drawdown
        equity = closed["pnl_dollars"].cumsum()
        max_dd = self._max_drawdown(equity)

        # Sharpe and Sortino (annualized)
        returns = closed["pnl_r"]
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        downside = returns[returns < 0]
        sortino = (returns.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0

        # Streaks
        max_win_streak, max_loss_streak = self._calculate_streaks(closed["pnl_r"])

        return {
            "total_trades": total,
            "wins": win_count,
            "losses": loss_count,
            "win_rate": round(win_rate * 100, 1),
            "avg_win_r": round(float(avg_win), 2),
            "avg_loss_r": round(float(avg_loss), 2),
            "profit_factor": round(float(profit_factor), 2),
            "total_pnl_r": round(float(returns.sum()), 2),
            "total_pnl_dollars": round(float(closed["pnl_dollars"].sum()), 2),
            "max_drawdown_dollars": round(float(max_dd), 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "sortino_ratio": round(float(sortino), 2),
            "max_win_r": round(float(returns.max()), 2),
            "max_loss_r": round(float(returns.min()), 2),
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "avg_trade_r": round(float(returns.mean()), 2),
        }

    def rolling_metrics(self, trades: list[dict], window: int = 20) -> dict:
        """Calculate rolling metrics over the last N trades."""
        if len(trades) < window:
            return self.calculate_metrics(trades)
        return self.calculate_metrics(trades[-window:])

    def segment_by_regime(self, trades: list[dict]) -> dict[str, dict]:
        """Segment performance by market regime."""
        result = {}
        for trade in trades:
            regime = trade.get("regime", "unknown")
            if regime not in result:
                result[regime] = []
            result[regime].append(trade)

        return {regime: self.calculate_metrics(t) for regime, t in result.items()}

    def segment_by_time_of_day(self, trades: list[dict]) -> dict[str, dict]:
        """Segment performance by time of day (entry hour)."""
        buckets: dict[str, list] = {}
        for trade in trades:
            entry_time = trade.get("entry_time", "")
            if entry_time:
                try:
                    dt = datetime.fromisoformat(entry_time) if isinstance(entry_time, str) else entry_time
                    hour = dt.hour
                    bucket = f"{hour:02d}:00-{hour:02d}:59"
                except (ValueError, AttributeError):
                    bucket = "unknown"
            else:
                bucket = "unknown"

            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(trade)

        return {bucket: self.calculate_metrics(t) for bucket, t in buckets.items()}

    def segment_by_strategy(self, trades: list[dict]) -> dict[str, dict]:
        """Segment performance by strategy name."""
        buckets: dict[str, list] = {}
        for trade in trades:
            strategy = trade.get("strategy_name", "unknown")
            if strategy not in buckets:
                buckets[strategy] = []
            buckets[strategy].append(trade)

        return {s: self.calculate_metrics(t) for s, t in buckets.items()}

    def _max_drawdown(self, equity: pd.Series) -> float:
        peak = 0.0
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _calculate_streaks(self, pnl_r: pd.Series) -> tuple[int, int]:
        max_win_streak = 0
        max_loss_streak = 0
        current_win = 0
        current_loss = 0

        for r in pnl_r:
            if r > 0:
                current_win += 1
                current_loss = 0
                max_win_streak = max(max_win_streak, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_loss_streak = max(max_loss_streak, current_loss)

        return max_win_streak, max_loss_streak

    def generate_report(self, trades: list[dict]) -> str:
        """Generate a markdown performance report."""
        metrics = self.calculate_metrics(trades)
        if metrics.get("total_trades", 0) == 0:
            return "# Performance Report\nNo trades to analyze."

        report = f"""# Performance Report

## Overview
- **Total Trades**: {metrics['total_trades']}
- **Win Rate**: {metrics['win_rate']}%
- **Profit Factor**: {metrics['profit_factor']}
- **Total P&L**: ${metrics['total_pnl_dollars']:,.2f} ({metrics['total_pnl_r']}R)

## Risk Metrics
- **Sharpe Ratio**: {metrics['sharpe_ratio']}
- **Sortino Ratio**: {metrics['sortino_ratio']}
- **Max Drawdown**: ${metrics['max_drawdown_dollars']:,.2f}

## Trade Quality
- **Avg Win**: {metrics['avg_win_r']}R
- **Avg Loss**: {metrics['avg_loss_r']}R
- **Max Win**: {metrics['max_win_r']}R
- **Max Loss**: {metrics['max_loss_r']}R
- **Max Win Streak**: {metrics['max_win_streak']}
- **Max Loss Streak**: {metrics['max_loss_streak']}
"""
        return report
