"""Trade logging with full context to SQLite.

Captures every trade with complete context (indicators, regime, news,
AI reasoning) within 5 seconds of close. Also logs AI decisions
for cost tracking and debugging.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite
import structlog

logger = structlog.get_logger(__name__)


class TradeLogger:
    """Logs trades, context, and AI decisions to SQLite."""

    def __init__(self, db_path: str = "data/trading.db"):
        self._db_path = db_path
        self._initialized = False

    async def _ensure_db(self) -> None:
        if self._initialized:
            return
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        migrations_path = Path(__file__).parent.parent.parent / "migrations" / "001_initial.sql"
        if migrations_path.exists():
            sql = migrations_path.read_text()
            async with aiosqlite.connect(self._db_path) as db:
                await db.executescript(sql)
                await db.commit()
        self._initialized = True

    async def log_trade_open(
        self, symbol: str, direction: str, entry_price: float,
        stop_loss: float, take_profit: float, position_size: int,
        strategy_name: str = "", signal_confidence: float = 0.0,
        regime: str = "", htf_bias: str = "", indicators: dict | None = None,
        ict_patterns: list | None = None, key_levels: dict | None = None,
        order_flow: dict | None = None, intermarket: dict | None = None,
        kill_zone: str = "", minutes_until_close: float = 0.0,
        llm_reasoning: str = "",
    ) -> str:
        await self._ensure_db()
        trade_id = str(uuid.uuid4())[:12]
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO trades (id, symbol, direction, entry_time, entry_price,
                   stop_loss, take_profit, position_size, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
                (trade_id, symbol, direction, now, entry_price, stop_loss,
                 take_profit, position_size),
            )
            await db.execute(
                """INSERT INTO trade_context (trade_id, regime, htf_bias, signal_confidence,
                   strategy_name, indicators_json, ict_patterns_json, key_levels_json,
                   order_flow_json, intermarket_json, kill_zone, minutes_until_close, llm_reasoning)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (trade_id, regime, htf_bias, signal_confidence, strategy_name,
                 json.dumps(indicators or {}), json.dumps(ict_patterns or []),
                 json.dumps(key_levels or {}), json.dumps(order_flow or {}),
                 json.dumps(intermarket or {}), kill_zone, minutes_until_close, llm_reasoning),
            )
            await db.commit()
        logger.info("trade_logger.opened", trade_id=trade_id, symbol=symbol, direction=direction)
        return trade_id

    async def log_trade_close(
        self, trade_id: str, exit_price: float, pnl_dollars: float,
        pnl_r: float, fees: float = 0.0, llm_journal: str = "",
    ) -> None:
        await self._ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        status = "win" if pnl_dollars > 0 else ("loss" if pnl_dollars < 0 else "breakeven")
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """UPDATE trades SET exit_time=?, exit_price=?, pnl_dollars=?, pnl_r=?,
                   fees=?, status=? WHERE id=?""",
                (now, exit_price, pnl_dollars, pnl_r, fees, status, trade_id),
            )
            if llm_journal:
                await db.execute(
                    "UPDATE trade_context SET llm_journal=? WHERE trade_id=?",
                    (llm_journal, trade_id),
                )
            await db.commit()
        logger.info("trade_logger.closed", trade_id=trade_id, pnl=round(pnl_dollars, 2), status=status)

    async def log_ai_decision(
        self, decision_type: str, input_context: dict, output_decision: dict,
        model_used: str = "", tokens_used: int = 0, cost_usd: float = 0.0, latency_ms: int = 0,
    ) -> None:
        await self._ensure_db()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO ai_decisions (timestamp, decision_type, input_context,
                   output_decision, model_used, tokens_used, cost_usd, latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now(timezone.utc).isoformat(), decision_type,
                 json.dumps(input_context), json.dumps(output_decision),
                 model_used, tokens_used, cost_usd, latency_ms),
            )
            await db.commit()

    async def log_daily_stats(
        self, date: str, starting_equity: float, ending_equity: float,
        total_pnl: float, trades_taken: int, wins: int, losses: int,
        dominant_regime: str = "",
    ) -> None:
        await self._ensure_db()
        win_rate = wins / trades_taken if trades_taken > 0 else 0.0
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO daily_stats
                   (date, starting_equity, ending_equity, total_pnl,
                    trades_taken, wins, losses, win_rate, dominant_regime)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (date, starting_equity, ending_equity, total_pnl,
                 trades_taken, wins, losses, win_rate, dominant_regime),
            )
            await db.commit()

    async def get_recent_trades(self, limit: int = 20) -> list[dict]:
        await self._ensure_db()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT t.*, tc.regime, tc.strategy_name, tc.signal_confidence,
                          tc.llm_reasoning, tc.llm_journal
                   FROM trades t LEFT JOIN trade_context tc ON t.id = tc.trade_id
                   ORDER BY t.entry_time DESC LIMIT ?""", (limit,))
            return [dict(row) for row in await cursor.fetchall()]

    async def get_trades_since(self, since: datetime) -> list[dict]:
        await self._ensure_db()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT t.*, tc.regime, tc.strategy_name, tc.signal_confidence
                   FROM trades t LEFT JOIN trade_context tc ON t.id = tc.trade_id
                   WHERE t.entry_time >= ? ORDER BY t.entry_time ASC""",
                (since.isoformat(),))
            return [dict(row) for row in await cursor.fetchall()]

    async def get_daily_stats(self, days: int = 30) -> list[dict]:
        await self._ensure_db()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM daily_stats ORDER BY date DESC LIMIT ?", (days,))
            return [dict(row) for row in await cursor.fetchall()]
