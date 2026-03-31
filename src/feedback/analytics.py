"""Performance analytics engine.

Calculates comprehensive trading metrics from the trade journal:
- Per-trade: R-multiple, duration, MAE, MFE
- Rolling: win rate, profit factor, Sharpe, Sortino, max drawdown
- Segmented: by regime, time of day, day of week, signal type, confidence
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class PerformanceAnalytics:
    """Calculates and reports trading performance metrics."""

    def calculate_metrics(self, trades: list[dict]) -> dict[str, Any]:
        if not trades:
            return {"error": "No trades", "total_trades": 0}

        df = pd.DataFrame(trades)
        closed = df[df["status"].isin(["win", "loss", "breakeven"])].copy()
        if len(closed) == 0:
            return {"error": "No closed trades", "total_trades": 0}

        for col in ["pnl_dollars", "pnl_r", "fees"]:
            if col in closed.columns:
                closed[col] = pd.to_numeric(closed[col], errors="coerce").fillna(0)

        wins = closed[closed["pnl_dollars"] > 0]
        losses = closed[closed["pnl_dollars"] < 0]
        total = len(closed)

        avg_win = wins["pnl_dollars"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["pnl_dollars"].mean()) if len(losses) > 0 else 0
        avg_win_r = wins["pnl_r"].mean() if len(wins) > 0 else 0
        avg_loss_r = abs(losses["pnl_r"].mean()) if len(losses) > 0 else 0

        gross_profit = wins["pnl_dollars"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl_dollars"].sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        equity = closed["pnl_dollars"].cumsum()
        peak = equity.cummax()
        max_drawdown = (peak - equity).max()

        returns = closed["pnl_r"]
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        downside = returns[returns < 0]
        sortino = (returns.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

        durations = []
        if "entry_time" in closed.columns and "exit_time" in closed.columns:
            for _, row in closed.iterrows():
                if row.get("entry_time") and row.get("exit_time"):
                    try:
                        durations.append((pd.to_datetime(row["exit_time"]) - pd.to_datetime(row["entry_time"])).total_seconds() / 60)
                    except Exception:
                        pass

        return {
            "total_trades": total, "wins": len(wins), "losses": len(losses),
            "win_rate": round(len(wins) / total * 100, 1),
            "avg_win_dollars": round(avg_win, 2), "avg_loss_dollars": round(avg_loss, 2),
            "avg_win_r": round(avg_win_r, 2), "avg_loss_r": round(avg_loss_r, 2),
            "max_win": round(closed["pnl_dollars"].max(), 2),
            "max_loss": round(closed["pnl_dollars"].min(), 2),
            "profit_factor": round(profit_factor, 2),
            "total_pnl": round(closed["pnl_dollars"].sum(), 2),
            "total_fees": round(closed["fees"].sum() if "fees" in closed.columns else 0, 2),
            "net_pnl": round(closed["pnl_dollars"].sum() - (closed["fees"].sum() if "fees" in closed.columns else 0), 2),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 2), "sortino_ratio": round(sortino, 2),
            "max_win_streak": self._max_streak(closed["pnl_dollars"] > 0),
            "max_loss_streak": self._max_streak(closed["pnl_dollars"] <= 0),
            "avg_duration_minutes": round(np.mean(durations), 1) if durations else 0,
        }

    def segment_by(self, trades: list[dict], field: str) -> dict[str, dict]:
        if not trades:
            return {}
        df = pd.DataFrame(trades)
        if field not in df.columns:
            return {}
        result = {}
        for value, group in df.groupby(field):
            if pd.isna(value) or value == "":
                continue
            result[str(value)] = self.calculate_metrics(group.to_dict("records"))
        return result

    def rolling_metrics(self, trades: list[dict], window: int = 20) -> list[dict]:
        if len(trades) < window:
            return []
        return [
            {**self.calculate_metrics(trades[i - window:i]), "trade_index": i}
            for i in range(window, len(trades) + 1)
        ]

    def format_summary(self, metrics: dict) -> str:
        if metrics.get("error"):
            return f"No data: {metrics['error']}"
        return (
            f"Trades: {metrics['total_trades']} ({metrics['wins']}W / {metrics['losses']}L)\n"
            f"Win Rate: {metrics['win_rate']}%\n"
            f"Avg Win: ${metrics['avg_win_dollars']:.2f} ({metrics['avg_win_r']:.2f}R) | "
            f"Avg Loss: ${metrics['avg_loss_dollars']:.2f} ({metrics['avg_loss_r']:.2f}R)\n"
            f"Profit Factor: {metrics['profit_factor']}\n"
            f"Total P&L: ${metrics['total_pnl']:.2f} | Max DD: ${metrics['max_drawdown']:.2f}\n"
            f"Sharpe: {metrics['sharpe_ratio']} | Sortino: {metrics['sortino_ratio']}"
        )

    @staticmethod
    def _max_streak(condition: pd.Series) -> int:
        if len(condition) == 0:
            return 0
        groups = (condition != condition.shift()).cumsum()
        return int(condition.groupby(groups).sum().max())
