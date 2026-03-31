"""A/B testing framework — shadow strategy vs live strategy.

Runs two strategy configs simultaneously. Shadow trades are logged
but never executed. Compares performance to validate changes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import aiosqlite
import structlog

from src.feedback.analytics import PerformanceAnalytics

logger = structlog.get_logger(__name__)


class ShadowRunner:
    """Runs a shadow strategy alongside live for A/B testing."""

    def __init__(self, db_path: str = "data/trading.db"):
        self._db_path = db_path
        self._analytics = PerformanceAnalytics()

    async def log_shadow_trade(
        self, symbol: str, direction: str, entry_price: float, exit_price: float,
        stop_loss: float, take_profit: float, position_size: int,
        pnl_dollars: float, pnl_r: float, strategy_variant: str, confidence: float,
    ) -> None:
        trade_id = f"shadow_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO shadow_trades
                   (id, symbol, direction, entry_time, exit_time, entry_price,
                    exit_price, position_size, stop_loss, take_profit,
                    pnl_dollars, pnl_r, strategy_variant, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (trade_id, symbol, direction, now, now, entry_price, exit_price,
                 position_size, stop_loss, take_profit, pnl_dollars, pnl_r,
                 strategy_variant, confidence),
            )
            await db.commit()
        logger.info("shadow.logged", variant=strategy_variant, pnl=round(pnl_dollars, 2))

    async def get_shadow_trades(self, limit: int = 50) -> list[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM shadow_trades ORDER BY entry_time DESC LIMIT ?", (limit,))
            return [dict(row) for row in await cursor.fetchall()]

    async def compare_performance(self, live_trades: list[dict]) -> dict:
        shadow_trades = await self.get_shadow_trades(100)
        if not live_trades or not shadow_trades:
            return {"error": "Insufficient data for comparison"}

        live_m = self._analytics.calculate_metrics(live_trades)
        shadow_m = self._analytics.calculate_metrics(shadow_trades)

        shadow_better = (
            shadow_m.get("profit_factor", 0) > live_m.get("profit_factor", 0) * 1.2
            and shadow_m.get("win_rate", 0) > live_m.get("win_rate", 0)
            and shadow_m.get("total_trades", 0) >= 10
        )

        return {
            "live": {"trades": live_m.get("total_trades", 0), "win_rate": live_m.get("win_rate", 0),
                     "profit_factor": live_m.get("profit_factor", 0), "total_pnl": live_m.get("total_pnl", 0)},
            "shadow": {"trades": shadow_m.get("total_trades", 0), "win_rate": shadow_m.get("win_rate", 0),
                       "profit_factor": shadow_m.get("profit_factor", 0), "total_pnl": shadow_m.get("total_pnl", 0)},
            "shadow_outperforms": shadow_better,
            "recommendation": "Consider switching to shadow parameters" if shadow_better else "Keep current parameters",
        }
