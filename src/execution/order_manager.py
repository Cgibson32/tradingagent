"""Order state machine for trade lifecycle management.

State machine:
IDLE → SIGNAL_RECEIVED → ENTRY_PLACED → ENTRY_FILLED → MANAGING → PARTIAL_TP → TRAILING → CLOSED

Handles entry, stop loss, take profit, partial fills, trailing stops,
and mandatory EOD close.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from src.data.models import Direction, OrderAction, OrderType

logger = structlog.get_logger(__name__)


class OrderState(str, Enum):
    IDLE = "idle"
    SIGNAL_RECEIVED = "signal_received"
    ENTRY_PLACED = "entry_placed"
    ENTRY_FILLED = "entry_filled"
    MANAGING = "managing"
    PARTIAL_TP = "partial_tp"
    TRAILING = "trailing"
    CLOSED = "closed"


@dataclass
class ManagedTrade:
    """Tracks a trade through its full lifecycle."""

    trade_id: str
    symbol: str
    direction: Direction
    target_entry: float
    stop_loss: float
    take_profit: float
    time_adjusted_tp: float | None = None
    size: int = 1
    state: OrderState = OrderState.IDLE

    # Fill info
    entry_price: float = 0.0
    entry_time: datetime | None = None
    entry_order_id: int | None = None
    sl_order_id: int | None = None
    tp_order_id: int | None = None

    # Partial TP
    partial_tp_filled: bool = False
    remaining_size: int = 0

    # Exit info
    exit_price: float = 0.0
    exit_time: datetime | None = None
    exit_reason: str = ""

    # Trailing stop
    trailing_stop: float = 0.0
    break_even_triggered: bool = False

    # Context
    signal_confidence: float = 0.0
    strategy_name: str = ""
    ai_reasoning: str = ""

    @property
    def is_active(self) -> bool:
        return self.state not in (OrderState.IDLE, OrderState.CLOSED)

    @property
    def unrealized_pnl_points(self) -> float:
        """Calculate unrealized P&L in points given current entry."""
        if self.entry_price == 0:
            return 0.0
        if self.direction == Direction.LONG:
            return self.exit_price - self.entry_price if self.exit_price else 0
        return self.entry_price - self.exit_price if self.exit_price else 0


class OrderManager:
    """Manages the complete lifecycle of trades.

    Responsibilities:
    - Place entry orders (limit at FVG or market)
    - Immediately place SL on fill
    - Manage partial TP at 1R
    - Trail remainder with ATR stop
    - Close all positions at 3:45 PM ET
    - Handle partial fills, rejections, disconnects
    """

    def __init__(
        self,
        rest_client,
        account_id: int,
        account_spec: str,
        point_value: float = 20.0,
        partial_tp_pct: float = 0.50,
        partial_tp_r: float = 1.0,
        trailing_atr_multiple: float = 1.5,
        break_even_trigger_pct: float = 0.40,
        limit_timeout_minutes: int = 30,
    ):
        self._rest = rest_client
        self._account_id = account_id
        self._account_spec = account_spec
        self._point_value = point_value
        self._partial_tp_pct = partial_tp_pct
        self._partial_tp_r = partial_tp_r
        self._trailing_atr_mult = trailing_atr_multiple
        self._be_trigger_pct = break_even_trigger_pct
        self._limit_timeout = limit_timeout_minutes

        self._active_trades: dict[str, ManagedTrade] = {}
        self._trade_counter = 0

    @property
    def active_trades(self) -> dict[str, ManagedTrade]:
        return self._active_trades

    @property
    def has_open_positions(self) -> bool:
        return any(t.is_active for t in self._active_trades.values())

    @property
    def open_position_count(self) -> int:
        return sum(1 for t in self._active_trades.values() if t.is_active)

    def create_trade(
        self,
        symbol: str,
        direction: Direction,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        time_adjusted_tp: float | None,
        size: int,
        confidence: float = 0.0,
        strategy: str = "",
        reasoning: str = "",
    ) -> ManagedTrade:
        """Create a new managed trade (does not place orders yet)."""
        self._trade_counter += 1
        trade_id = f"trade_{self._trade_counter:04d}_{datetime.now(timezone.utc).strftime('%H%M%S')}"

        trade = ManagedTrade(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            target_entry=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_adjusted_tp=time_adjusted_tp,
            size=size,
            remaining_size=size,
            signal_confidence=confidence,
            strategy_name=strategy,
            ai_reasoning=reasoning,
            state=OrderState.SIGNAL_RECEIVED,
        )

        self._active_trades[trade_id] = trade
        logger.info("order_manager.trade_created", trade_id=trade_id, direction=direction.value, size=size)
        return trade

    async def place_entry(self, trade: ManagedTrade, use_limit: bool = True) -> bool:
        """Place the entry order for a trade.

        Args:
            trade: The managed trade to enter
            use_limit: If True, place limit order at target; if False, market order

        Returns:
            True if order was placed successfully
        """
        action = OrderAction.BUY if trade.direction == Direction.LONG else OrderAction.SELL
        order_type = OrderType.LIMIT if use_limit else OrderType.MARKET

        try:
            result = await self._rest.place_order(
                account_id=self._account_id,
                account_spec=self._account_spec,
                symbol=trade.symbol,
                action=action,
                qty=trade.size,
                order_type=order_type,
                price=trade.target_entry if use_limit else None,
            )

            trade.entry_order_id = result.order_id
            trade.state = OrderState.ENTRY_PLACED

            logger.info(
                "order_manager.entry_placed",
                trade_id=trade.trade_id,
                order_id=result.order_id,
                type=order_type.value,
            )

            # Schedule timeout for limit orders
            if use_limit:
                asyncio.create_task(self._limit_timeout_check(trade))

            return True

        except Exception as e:
            logger.error("order_manager.entry_failed", trade_id=trade.trade_id, error=str(e))
            trade.state = OrderState.CLOSED
            trade.exit_reason = f"Entry failed: {e}"
            return False

    async def on_fill(self, trade: ManagedTrade, fill_price: float, fill_qty: int) -> None:
        """Handle an entry fill — immediately place stop loss."""
        trade.entry_price = fill_price
        trade.entry_time = datetime.now(timezone.utc)
        trade.state = OrderState.ENTRY_FILLED

        logger.info(
            "order_manager.entry_filled",
            trade_id=trade.trade_id,
            fill_price=fill_price,
            qty=fill_qty,
        )

        # Place stop loss IMMEDIATELY
        await self._place_stop_loss(trade)
        trade.state = OrderState.MANAGING

    async def _place_stop_loss(self, trade: ManagedTrade) -> None:
        """Place the stop loss order. Must succeed."""
        action = OrderAction.SELL if trade.direction == Direction.LONG else OrderAction.BUY

        try:
            result = await self._rest.place_order(
                account_id=self._account_id,
                account_spec=self._account_spec,
                symbol=trade.symbol,
                action=action,
                qty=trade.remaining_size,
                order_type=OrderType.STOP,
                stop_price=trade.stop_loss,
            )
            trade.sl_order_id = result.order_id
            logger.info("order_manager.sl_placed", trade_id=trade.trade_id, sl=trade.stop_loss)
        except Exception as e:
            logger.error("order_manager.sl_failed", trade_id=trade.trade_id, error=str(e))
            # SL placement failure is critical — close position at market
            await self.close_trade_market(trade, reason="SL placement failed")

    async def check_management(self, trade: ManagedTrade, current_price: float, atr_value: float) -> None:
        """Check and execute trade management logic.

        Called on each new price update for active trades.
        """
        if trade.state not in (OrderState.MANAGING, OrderState.PARTIAL_TP, OrderState.TRAILING):
            return

        if trade.direction == Direction.LONG:
            pnl_points = current_price - trade.entry_price
        else:
            pnl_points = trade.entry_price - current_price

        risk_points = abs(trade.entry_price - trade.stop_loss)

        if risk_points == 0:
            return

        r_multiple = pnl_points / risk_points

        # Break-even trigger
        if not trade.break_even_triggered and r_multiple >= self._be_trigger_pct * (abs(trade.take_profit - trade.entry_price) / risk_points):
            await self._move_to_break_even(trade)

        # Partial TP at 1R
        if not trade.partial_tp_filled and r_multiple >= self._partial_tp_r:
            await self._take_partial_profit(trade)

        # Update trailing stop
        if trade.state == OrderState.TRAILING:
            await self._update_trailing_stop(trade, current_price, atr_value)

    async def _move_to_break_even(self, trade: ManagedTrade) -> None:
        """Move stop loss to break-even (entry price)."""
        if trade.break_even_triggered:
            return

        trade.break_even_triggered = True
        new_sl = trade.entry_price

        if trade.sl_order_id:
            try:
                await self._rest.modify_order(
                    order_id=trade.sl_order_id,
                    stop_price=new_sl,
                )
                trade.stop_loss = new_sl
                logger.info("order_manager.break_even", trade_id=trade.trade_id, new_sl=new_sl)
            except Exception as e:
                logger.error("order_manager.be_failed", trade_id=trade.trade_id, error=str(e))

    async def _take_partial_profit(self, trade: ManagedTrade) -> None:
        """Close partial position at 1R profit."""
        partial_qty = max(1, int(trade.size * self._partial_tp_pct))
        action = OrderAction.SELL if trade.direction == Direction.LONG else OrderAction.BUY

        try:
            await self._rest.place_order(
                account_id=self._account_id,
                account_spec=self._account_spec,
                symbol=trade.symbol,
                action=action,
                qty=partial_qty,
                order_type=OrderType.MARKET,
            )
            trade.partial_tp_filled = True
            trade.remaining_size = trade.size - partial_qty
            trade.state = OrderState.TRAILING

            # Update SL quantity
            if trade.sl_order_id and trade.remaining_size > 0:
                await self._rest.modify_order(
                    order_id=trade.sl_order_id,
                    qty=trade.remaining_size,
                )

            logger.info(
                "order_manager.partial_tp",
                trade_id=trade.trade_id,
                closed=partial_qty,
                remaining=trade.remaining_size,
            )
        except Exception as e:
            logger.error("order_manager.partial_tp_failed", trade_id=trade.trade_id, error=str(e))

    async def _update_trailing_stop(self, trade: ManagedTrade, price: float, atr_value: float) -> None:
        """Update trailing stop based on ATR."""
        trail_distance = atr_value * self._trailing_atr_mult

        if trade.direction == Direction.LONG:
            new_trail = price - trail_distance
            if new_trail > trade.stop_loss:
                trade.stop_loss = new_trail
                trade.trailing_stop = new_trail
        else:
            new_trail = price + trail_distance
            if new_trail < trade.stop_loss:
                trade.stop_loss = new_trail
                trade.trailing_stop = new_trail

        # Update the SL order
        if trade.sl_order_id:
            try:
                await self._rest.modify_order(
                    order_id=trade.sl_order_id,
                    stop_price=trade.stop_loss,
                )
            except Exception as e:
                logger.error("order_manager.trail_update_failed", error=str(e))

    async def close_trade_market(self, trade: ManagedTrade, reason: str = "manual") -> None:
        """Close a trade at market price immediately."""
        if trade.state == OrderState.CLOSED:
            return

        action = OrderAction.SELL if trade.direction == Direction.LONG else OrderAction.BUY

        try:
            # Cancel existing SL/TP orders
            if trade.sl_order_id:
                try:
                    await self._rest.cancel_order(trade.sl_order_id)
                except Exception:
                    pass

            # Place market close
            await self._rest.place_order(
                account_id=self._account_id,
                account_spec=self._account_spec,
                symbol=trade.symbol,
                action=action,
                qty=trade.remaining_size,
                order_type=OrderType.MARKET,
            )

            trade.state = OrderState.CLOSED
            trade.exit_time = datetime.now(timezone.utc)
            trade.exit_reason = reason

            logger.info(
                "order_manager.trade_closed",
                trade_id=trade.trade_id,
                reason=reason,
                remaining_closed=trade.remaining_size,
            )
        except Exception as e:
            logger.error("order_manager.close_failed", trade_id=trade.trade_id, error=str(e))

    async def close_all_positions(self, reason: str = "eod_close") -> None:
        """Close ALL active positions at market. Used for EOD and emergencies."""
        active = [t for t in self._active_trades.values() if t.is_active]
        for trade in active:
            await self.close_trade_market(trade, reason=reason)
        logger.info("order_manager.all_closed", count=len(active), reason=reason)

    async def _limit_timeout_check(self, trade: ManagedTrade) -> None:
        """Cancel unfilled limit orders after timeout."""
        await asyncio.sleep(self._limit_timeout * 60)
        if trade.state == OrderState.ENTRY_PLACED and trade.entry_order_id:
            try:
                await self._rest.cancel_order(trade.entry_order_id)
                trade.state = OrderState.CLOSED
                trade.exit_reason = "Limit order timeout"
                logger.info("order_manager.limit_timeout", trade_id=trade.trade_id)
            except Exception as e:
                logger.error("order_manager.cancel_failed", trade_id=trade.trade_id, error=str(e))
