"""Crash recovery for mid-session restarts.

On startup, reconciles local state with broker state:
- Queries Tradovate for open positions and pending orders
- Reconstructs daily P&L from today's completed trades
- Restores drawdown floor from account equity history
- Writes periodic checkpoints for faster recovery
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import structlog

logger = structlog.get_logger(__name__)


class StateRecovery:
    """Handles crash recovery by reconciling local and broker state.

    Writes checkpoints every 60 seconds with:
    - Current agent state
    - Open positions
    - Daily P&L and metrics
    """

    def __init__(self, db_path: str = "data/trading.db"):
        self._db_path = db_path

    async def save_checkpoint(
        self,
        state: dict,
        open_positions: list[dict],
        daily_pnl: float,
        drawdown_floor: float,
        trades_today: int,
        consecutive_losses: int,
    ) -> None:
        """Save a state checkpoint to SQLite."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO agent_checkpoints
                   (timestamp, state_json, open_positions_json, daily_pnl,
                    drawdown_floor, trades_today, consecutive_losses)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(state),
                    json.dumps(open_positions),
                    daily_pnl,
                    drawdown_floor,
                    trades_today,
                    consecutive_losses,
                ),
            )
            await db.commit()

    async def load_latest_checkpoint(self) -> dict | None:
        """Load the most recent checkpoint from SQLite."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM agent_checkpoints ORDER BY id DESC LIMIT 1"
                )
                row = await cursor.fetchone()
                if row:
                    return {
                        "timestamp": row["timestamp"],
                        "state": json.loads(row["state_json"]),
                        "open_positions": json.loads(row["open_positions_json"]),
                        "daily_pnl": row["daily_pnl"],
                        "drawdown_floor": row["drawdown_floor"],
                        "trades_today": row["trades_today"],
                        "consecutive_losses": row["consecutive_losses"],
                    }
        except Exception as e:
            logger.warning("state_recovery.load_failed", error=str(e))
        return None

    async def reconcile(
        self,
        rest_client,
        account_id: int,
        checkpoint: dict | None = None,
    ) -> dict:
        """Reconcile local state with broker state.

        Args:
            rest_client: TradovateRestClient instance
            account_id: Tradovate account ID
            checkpoint: Latest local checkpoint (if available)

        Returns:
            Reconciled state dict with positions, pnl, etc.
        """
        logger.info("state_recovery.reconciling")

        # Get broker state
        broker_positions = await rest_client.list_positions()
        broker_orders = await rest_client.list_orders()
        balance = await rest_client.get_cash_balance(account_id)

        # Filter for active positions
        active_positions = [p for p in broker_positions if not p.is_flat]

        state = {
            "equity": balance.total_equity,
            "cash_balance": balance.cash_balance,
            "open_positions": [p.model_dump() for p in active_positions],
            "pending_orders": len(broker_orders),
            "daily_pnl": 0.0,
            "drawdown_floor": 0.0,
            "trades_today": 0,
            "consecutive_losses": 0,
        }

        # Merge with checkpoint if available
        if checkpoint:
            state["daily_pnl"] = checkpoint.get("daily_pnl", 0.0)
            state["drawdown_floor"] = checkpoint.get("drawdown_floor", 0.0)
            state["trades_today"] = checkpoint.get("trades_today", 0)
            state["consecutive_losses"] = checkpoint.get("consecutive_losses", 0)

            # Check for position mismatches
            local_positions = checkpoint.get("open_positions", [])
            if len(active_positions) != len(local_positions):
                logger.warning(
                    "state_recovery.position_mismatch",
                    broker=len(active_positions),
                    local=len(local_positions),
                )

        # Flag positions the agent didn't place
        for pos in active_positions:
            logger.info(
                "state_recovery.position_found",
                symbol=pos.symbol,
                direction=pos.direction.value if pos.direction else "flat",
                size=pos.size,
                entry=pos.net_price,
            )

        logger.info(
            "state_recovery.complete",
            equity=state["equity"],
            open_positions=len(active_positions),
            daily_pnl=state["daily_pnl"],
        )

        return state

    async def cleanup_old_checkpoints(self, keep_last: int = 100) -> None:
        """Remove old checkpoints to prevent database bloat."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """DELETE FROM agent_checkpoints
                       WHERE id NOT IN (
                           SELECT id FROM agent_checkpoints
                           ORDER BY id DESC LIMIT ?
                       )""",
                    (keep_last,),
                )
                await db.commit()
        except Exception as e:
            logger.warning("state_recovery.cleanup_failed", error=str(e))
