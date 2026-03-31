"""Tradovate REST API client.

Wraps all REST endpoints with typed methods, automatic authentication,
retry logic with exponential backoff, and rate limit handling.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import structlog

from src.data.auth import TokenManager
from src.data.models import (
    AccountBalance,
    OrderAction,
    OrderResult,
    OrderType,
    PlaceOrderRequest,
    Position,
    TimeInForce,
)

logger = structlog.get_logger(__name__)

# Tradovate rate limits: 120 requests per 5 seconds for order endpoints
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds


class TradovateRestClient:
    """Async REST client for Tradovate API.

    All methods return typed Pydantic models. Handles authentication,
    retries, and rate limiting automatically.
    """

    def __init__(self, token_manager: TokenManager, base_url: str):
        self._token_manager = token_manager
        self._base_url = base_url
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated request with retry logic."""
        client = await self._get_client()
        url = f"{self._base_url}{endpoint}"

        for attempt in range(MAX_RETRIES + 1):
            token = await self._token_manager.ensure_valid()
            headers = {"Authorization": f"Bearer {token}"}

            try:
                response = await client.request(
                    method, url, headers=headers, json=json_data, params=params
                )

                if response.status_code == 429:
                    # Rate limited — back off and retry
                    wait = RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "tradovate.rest.rate_limited",
                        endpoint=endpoint,
                        retry_in=wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(
                    "tradovate.rest.error",
                    endpoint=endpoint,
                    status=e.response.status_code,
                    body=e.response.text,
                    attempt=attempt,
                )
                if attempt < MAX_RETRIES and e.response.status_code >= 500:
                    await asyncio.sleep(RETRY_BACKOFF_BASE * (2**attempt))
                    continue
                raise
            except httpx.RequestError as e:
                logger.error(
                    "tradovate.rest.request_error",
                    endpoint=endpoint,
                    error=str(e),
                    attempt=attempt,
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BACKOFF_BASE * (2**attempt))
                    continue
                raise

        raise RuntimeError(f"Max retries exceeded for {endpoint}")

    # ── Contract endpoints ──

    async def find_contract(self, name: str) -> dict[str, Any]:
        """Look up a contract by name (e.g., 'NQU5')."""
        return await self._request("GET", "/contract/find", params={"name": name})

    async def get_contract(self, contract_id: int) -> dict[str, Any]:
        """Get contract details by ID."""
        return await self._request("GET", f"/contract/item?id={contract_id}")

    # ── Account endpoints ──

    async def list_accounts(self) -> list[dict[str, Any]]:
        """List all accounts."""
        return await self._request("GET", "/account/list")

    async def get_cash_balance(self, account_id: int) -> AccountBalance:
        """Get cash balance for an account."""
        data = await self._request(
            "GET", "/cashBalance/getCashBalanceSnapshot", params={"accountId": account_id}
        )
        return AccountBalance(
            account_id=account_id,
            cash_balance=data.get("cashBalance", 0.0),
            open_trade_equity=data.get("openTradeEquity", 0.0),
            total_equity=data.get("totalEquity", 0.0),
            realized_pnl=data.get("realizedPnl", 0.0),
            margin_used=data.get("marginUsed", 0.0),
        )

    # ── Position endpoints ──

    async def list_positions(self) -> list[Position]:
        """List all open positions."""
        data = await self._request("GET", "/position/list")
        positions = []
        for item in data:
            positions.append(
                Position(
                    accountId=item.get("accountId", 0),
                    contractId=item.get("contractId", 0),
                    netPos=item.get("netPos", 0),
                    netPrice=item.get("netPrice", 0.0),
                    timestamp=item.get("timestamp"),
                )
            )
        return positions

    # ── Order endpoints ──

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
        """Place a new order."""
        payload = {
            "accountSpec": account_spec,
            "accountId": account_id,
            "action": action.value,
            "symbol": symbol,
            "orderQty": qty,
            "orderType": order_type.value,
            "timeInForce": time_in_force.value,
            "isAutomated": True,
        }
        if price is not None:
            payload["price"] = price
        if stop_price is not None:
            payload["stopPrice"] = stop_price

        logger.info(
            "tradovate.order.placing",
            symbol=symbol,
            action=action.value,
            qty=qty,
            order_type=order_type.value,
        )

        data = await self._request("POST", "/order/placeorder", json_data=payload)

        result = OrderResult(
            orderId=data.get("orderId", 0),
            accountId=account_id,
            action=action.value,
            symbol=symbol,
            orderQty=qty,
            orderType=order_type.value,
            price=price,
            stopPrice=stop_price,
            status=data.get("orderStatus", ""),
        )

        logger.info(
            "tradovate.order.placed",
            order_id=result.order_id,
            status=result.status,
        )
        return result

    async def cancel_order(self, order_id: int) -> dict[str, Any]:
        """Cancel an existing order."""
        logger.info("tradovate.order.cancelling", order_id=order_id)
        return await self._request(
            "POST", "/order/cancelorder", json_data={"orderId": order_id}
        )

    async def modify_order(
        self,
        order_id: int,
        qty: int | None = None,
        price: float | None = None,
        stop_price: float | None = None,
    ) -> dict[str, Any]:
        """Modify an existing order."""
        payload: dict[str, Any] = {"orderId": order_id}
        if qty is not None:
            payload["orderQty"] = qty
        if price is not None:
            payload["price"] = price
        if stop_price is not None:
            payload["stopPrice"] = stop_price

        logger.info("tradovate.order.modifying", order_id=order_id, changes=payload)
        return await self._request("POST", "/order/modifyorder", json_data=payload)

    async def list_orders(self) -> list[dict[str, Any]]:
        """List all orders."""
        return await self._request("GET", "/order/list")

    # ── Leverage ──

    async def set_leverage(self, account_id: int, contract_id: int, leverage: int) -> None:
        """Set leverage for a contract (if applicable)."""
        # Note: Tradovate futures don't use leverage the same way as crypto.
        # This is a placeholder for future instrument support.
        logger.info(
            "tradovate.leverage.set",
            account_id=account_id,
            contract_id=contract_id,
            leverage=leverage,
        )

    async def close(self) -> None:
        """Clean up HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
