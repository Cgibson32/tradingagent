"""Paper trading client — same interface as live Tradovate client.

Simulates order fills with realistic slippage and commission
models for NQ futures. Must run for 2 weeks before live deployment.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

import structlog

from src.data.models import (
    AccountBalance,
    OrderAction,
    OrderResult,
    OrderType,
    Position,
    TimeInForce,
)

logger = structlog.get_logger(__name__)


class PaperTradingClient:
    """Simulates Tradovate order execution for paper trading.

    Same interface as TradovateRestClient so it can be swapped in.
    Models realistic slippage and commission for NQ futures.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission_per_contract: float = 0.62,
        slippage_ticks: int = 2,
        tick_value: float = 0.50,
        point_value: float = 2.0,
    ):
        self._balance = initial_balance
        self._initial_balance = initial_balance
        self._commission = commission_per_contract
        self._slippage_ticks = slippage_ticks
        self._tick_value = tick_value
        self._point_value = point_value

        self._positions: dict[str, Position] = {}
        self._orders: dict[int, dict] = {}
        self._order_counter = 0
        self._fills: list[dict] = []
        self._daily_pnl: float = 0.0
        self._total_pnl: float = 0.0

        # Current market prices (updated externally)
        self._current_prices: dict[str, float] = {}

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    def update_price(self, symbol: str, price: float) -> None:
        """Update the current market price for a symbol."""
        self._current_prices[symbol] = price

        # Check stop orders
        self._check_stop_orders(symbol, price)

    def _check_stop_orders(self, symbol: str, price: float) -> None:
        """Check if any stop orders should be triggered."""
        triggered = []
        for order_id, order in self._orders.items():
            if order["symbol"] != symbol or order["status"] != "working":
                continue
            if order["order_type"] != "Stop":
                continue

            stop_price = order["stop_price"]
            if order["action"] == "Sell" and price <= stop_price:
                triggered.append(order_id)
            elif order["action"] == "Buy" and price >= stop_price:
                triggered.append(order_id)

        for order_id in triggered:
            self._fill_order(order_id)

    def _apply_slippage(self, price: float, action: str) -> float:
        """Apply realistic slippage to a fill price."""
        slippage_points = random.uniform(0, self._slippage_ticks) * 0.25  # tick_size = 0.25
        if action == "Buy":
            return price + slippage_points  # Worse fill for buys
        return price - slippage_points  # Worse fill for sells

    def _fill_order(self, order_id: int) -> None:
        """Simulate filling an order."""
        order = self._orders[order_id]
        symbol = order["symbol"]
        action = order["action"]
        qty = order["qty"]

        # Determine fill price
        if order["order_type"] == "Market":
            base_price = self._current_prices.get(symbol, order.get("price", 0))
        elif order["order_type"] == "Stop":
            base_price = order["stop_price"]
        elif order["order_type"] == "Limit":
            base_price = order["price"]
        else:
            base_price = self._current_prices.get(symbol, 0)

        fill_price = self._apply_slippage(base_price, action)
        commission = self._commission * qty

        # Update position
        current_pos = self._positions.get(symbol)
        if current_pos and not current_pos.is_flat:
            # Closing or adding to position
            if (action == "Sell" and current_pos.net_pos > 0) or \
               (action == "Buy" and current_pos.net_pos < 0):
                # Closing position
                close_qty = min(qty, abs(current_pos.net_pos))
                if current_pos.net_pos > 0:
                    pnl = (fill_price - current_pos.net_price) * self._point_value * close_qty
                else:
                    pnl = (current_pos.net_price - fill_price) * self._point_value * close_qty
                pnl -= commission
                self._balance += pnl
                self._daily_pnl += pnl
                self._total_pnl += pnl

                # Update position
                remaining = current_pos.net_pos + (qty if action == "Buy" else -qty)
                if remaining == 0:
                    current_pos.net_pos = 0
                else:
                    current_pos.net_pos = remaining
            else:
                # Adding to position
                if action == "Buy":
                    current_pos.net_pos += qty
                else:
                    current_pos.net_pos -= qty
                current_pos.net_price = fill_price
                self._balance -= commission
        else:
            # New position
            net_pos = qty if action == "Buy" else -qty
            self._positions[symbol] = Position(
                netPos=net_pos,
                netPrice=fill_price,
                symbol=symbol,
            )
            self._balance -= commission

        order["status"] = "filled"
        order["fill_price"] = fill_price

        self._fills.append({
            "order_id": order_id,
            "symbol": symbol,
            "action": action,
            "qty": qty,
            "fill_price": fill_price,
            "commission": commission,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        logger.info(
            "paper.fill",
            order_id=order_id,
            symbol=symbol,
            action=action,
            qty=qty,
            fill_price=round(fill_price, 2),
            balance=round(self._balance, 2),
        )

    # ── Same interface as TradovateRestClient ──

    async def find_contract(self, name: str) -> dict:
        return {"id": 1, "name": name, "description": f"Paper {name}"}

    async def list_accounts(self) -> list[dict]:
        return [{"id": 1, "name": "Paper Account", "netLiq": self._balance}]

    async def get_cash_balance(self, account_id: int) -> AccountBalance:
        open_pnl = sum(
            self._unrealized_pnl(symbol, pos)
            for symbol, pos in self._positions.items()
            if not pos.is_flat
        )
        return AccountBalance(
            account_id=account_id,
            cash_balance=self._balance,
            open_trade_equity=open_pnl,
            total_equity=self._balance + open_pnl,
            realized_pnl=self._total_pnl,
        )

    def _unrealized_pnl(self, symbol: str, pos: Position) -> float:
        current = self._current_prices.get(symbol, pos.net_price)
        if pos.net_pos > 0:
            return (current - pos.net_price) * self._point_value * pos.net_pos
        elif pos.net_pos < 0:
            return (pos.net_price - current) * self._point_value * abs(pos.net_pos)
        return 0.0

    async def list_positions(self) -> list[Position]:
        return list(self._positions.values())

    async def place_order(
        self,
        account_id: int,
        account_spec: str,
        symbol: str,
        action: OrderAction,
        qty: int,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
    ) -> OrderResult:
        self._order_counter += 1
        order_id = self._order_counter

        order = {
            "order_id": order_id,
            "symbol": symbol,
            "action": action.value,
            "qty": qty,
            "order_type": order_type.value,
            "price": price,
            "stop_price": stop_price,
            "status": "working",
        }
        self._orders[order_id] = order

        # Immediately fill market orders
        if order_type == OrderType.MARKET:
            self._fill_order(order_id)

        return OrderResult(
            orderId=order_id,
            accountId=account_id,
            action=action.value,
            symbol=symbol,
            orderQty=qty,
            orderType=order_type.value,
            status=order["status"],
            fillPrice=order.get("fill_price"),
        )

    async def cancel_order(self, order_id: int) -> dict:
        if order_id in self._orders:
            self._orders[order_id]["status"] = "cancelled"
        return {"orderId": order_id, "status": "cancelled"}

    async def modify_order(
        self, order_id: int, qty: int | None = None,
        price: float | None = None, stop_price: float | None = None,
    ) -> dict:
        if order_id in self._orders:
            if qty is not None:
                self._orders[order_id]["qty"] = qty
            if price is not None:
                self._orders[order_id]["price"] = price
            if stop_price is not None:
                self._orders[order_id]["stop_price"] = stop_price
        return {"orderId": order_id, "status": "modified"}

    async def list_orders(self) -> list[dict]:
        return [o for o in self._orders.values() if o["status"] == "working"]

    def reset_daily(self) -> None:
        """Reset daily P&L tracking."""
        self._daily_pnl = 0.0

    async def close(self) -> None:
        """No-op for paper client."""
        pass
