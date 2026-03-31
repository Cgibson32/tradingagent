"""Trade logging — full context capture for every trade.

Logs every trade with complete context (indicators, regime, news,
AI reasoning) to SQLite within 5 seconds of close.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import structlog

logger = structlog.get_logger(__name__)


class TradeLogger:
    """Logs trades and AI decisions to SQLite for learning and analysis."""

    def __init__(self, db_path: str = "data/trading.db"):
        self._db_path = db_path
        self._initialized = False

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        migrations_path = Path(__file__).parent.parent.parent / "migrations" / "001_initial.sql"

        async with aiosqlite.connect(self._db_path) as db:
            if migrations_path.exists():
                sql = migrations_path.read_text()
                await db.executescript(sql)
                await db.commit()
            self._initialized = True
            logger.info("trade_logger.initialized", db_path=self._db_path)

    async def log_trade(
        self,
        symbol: str,
        direction: str,
        entry_time: datetime,
        exit_time: datetime | None,
        entry_price: float,
        exit_price: float | None,
        position_size: int,
        stop_loss: float,
        take_profit: float,
        pnl_dollars: float | None = None,
        pnl_r: float | None = None,
        fees: float = 0.0,
        status: str = "open",
        is_shadow: bool = False,
        # Context
        regime: str = "",
        htf_bias: str = "",
        signal_confidence: float = 0.0,
        strategy_name: str = "",
        indicators_json: str = "{}",
        ict_patterns_json: str = "{}",
        key_levels_json: str = "{}",
        order_flow_json: str = "{}",
        intermarket_json: str = "{}",
        kill_zone: str = "",
        minutes_until_close: float = 0.0,
        llm_reasoning: str = "",
        llm_journal: str = "",
        news_events: str = "",
    ) -> str:
        """Log a trade with full context.

        Returns:
            The trade ID (UUID).
        """
        trade_id = str(uuid.uuid4())[:12]

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO trades
                   (id, symbol, direction, entry_time, exit_time, entry_price, exit_price,
                    position_size, stop_loss, take_profit, pnl_dollars, pnl_r, fees, status, is_shadow)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade_id, symbol, direction,
                    entry_time.isoformat() if entry_time else None,
                    exit_time.isoformat() if exit_time else None,
                    entry_price, exit_price, position_size,
                    stop_loss, take_profit, pnl_dollars, pnl_r, fees, status,
                    1 if is_shadow else 0,
                ),
            )

            await db.execute(
                """INSERT INTO trade_context
                   (trade_id, regime, htf_bias, signal_confidence, strategy_name,
                    indicators_json, ict_patterns_json, key_levels_json,
                    order_flow_json, intermarket_json, kill_zone, minutes_until_close,
                    llm_reasoning, llm_journal, news_events)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade_id, regime, htf_bias, signal_confidence, strategy_name,
                    indicators_json, ict_patterns_json, key_levels_json,
                    order_flow_json, intermarket_json, kill_zone, minutes_until_close,
                    llm_reasoning, llm_journal, news_events,
                ),
            )
            await db.commit()

        logger.info("trade_logger.logged", trade_id=trade_id, symbol=symbol, direction=direction, status=status)
        return trade_id

    async def update_trade(
        self, trade_id: str, exit_price: float, exit_time: datetime,
        pnl_dollars: float, pnl_r: float, status: str, fees: float = 0.0,
    ) -> None:
        """Update a trade on close."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """UPDATE trades SET exit_price=?, exit_time=?, pnl_dollars=?,
                   pnl_r=?, status=?, fees=? WHERE id=?""",
                (exit_price, exit_time.isoformat(), pnl_dollars, pnl_r, status, fees, trade_id),
            )
            await db.commit()

    async def update_journal(self, trade_id: str, journal_text: str) -> None:
        """Add LLM journal entry to a trade."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "UPDATE trade_context SET llm_journal=? WHERE trade_id=?",
                (journal_text, trade_id),
            )
            await db.commit()

    async def log_ai_decision(
        self, decision_type: str, input_context: str, output_decision: str,
        model: str, tokens: int = 0, cost_usd: float = 0.0, latency_ms: int = 0,
    ) -> None:
        """Log an AI decision for audit trail."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO ai_decisions
                   (timestamp, decision_type, input_context, output_decision,
                    model_used, tokens_used, cost_usd, latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    decision_type, input_context, output_decision,
                    model, tokens, cost_usd, latency_ms,
                ),
            )
            await db.commit()

    async def log_daily_stats(
        self, date: str, starting_equity: float, ending_equity: float,
        high_water_mark: float, drawdown_floor: float, total_pnl: float,
        trades_taken: int, wins: int, losses: int,
    ) -> None:
        """Log daily performance summary."""
        win_rate = wins / trades_taken if trades_taken > 0 else 0
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO daily_stats
                   (date, starting_equity, ending_equity, high_water_mark,
                    drawdown_floor, total_pnl, trades_taken, wins, losses, win_rate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (date, starting_equity, ending_equity, high_water_mark,
                 drawdown_floor, total_pnl, trades_taken, wins, losses, win_rate),
            )
            await db.commit()

    async def get_recent_trades(self, n: int = 20, shadow: bool = False) -> list[dict]:
        """Get the N most recent trades."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT t.*, tc.regime, tc.strategy_name, tc.signal_confidence
                   FROM trades t LEFT JOIN trade_context tc ON t.id = tc.trade_id
                   WHERE t.is_shadow = ? ORDER BY t.entry_time DESC LIMIT ?""",
                (1 if shadow else 0, n),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
