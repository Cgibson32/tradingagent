"""Tradovate Market Data WebSocket client.

Connects to the market data endpoint for real-time quotes, chart data,
and depth of market (DOM). Uses the same custom text protocol as the
trading WebSocket but on a separate connection.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable

import structlog

from src.data.models import Candle, DOMLevel, DOMSnapshot, Quote
from src.data.websocket import TradovateWebSocket

logger = structlog.get_logger(__name__)


class MarketDataWebSocket(TradovateWebSocket):
    """Market data WebSocket client for Tradovate.

    Provides methods for subscribing to:
    - Real-time quotes (bid/ask/last)
    - Chart data (OHLCV candles at various timeframes)
    - Depth of market (DOM)
    """

    def __init__(self, url: str, token: str):
        super().__init__(url, token)
        self._quote_callbacks: dict[str, list[Callable]] = {}
        self._chart_callbacks: dict[int, list[Callable]] = {}
        self._dom_callbacks: dict[str, list[Callable]] = {}
        self._chart_request_map: dict[int, str] = {}  # req_id → symbol

    def on_quote(self, symbol: str, callback: Callable) -> None:
        """Register a callback for quote updates on a symbol.

        Callback signature: async def callback(quote: Quote)
        """
        self._quote_callbacks.setdefault(symbol, []).append(callback)

    def on_chart(self, request_id: int, callback: Callable) -> None:
        """Register a callback for chart data updates.

        Callback signature: async def callback(candle: Candle)
        """
        self._chart_callbacks.setdefault(request_id, []).append(callback)

    def on_dom(self, symbol: str, callback: Callable) -> None:
        """Register a callback for DOM updates.

        Callback signature: async def callback(dom: DOMSnapshot)
        """
        self._dom_callbacks.setdefault(symbol, []).append(callback)

    async def subscribe_quotes(self, symbol: str) -> None:
        """Subscribe to real-time quote updates for a symbol."""
        logger.info("tradovate.md.subscribing_quotes", symbol=symbol)
        await self.send_request(
            "md/subscribequote",
            body={"symbol": symbol},
            timeout=10.0,
        )
        logger.info("tradovate.md.subscribed_quotes", symbol=symbol)

    async def unsubscribe_quotes(self, symbol: str) -> None:
        """Unsubscribe from quote updates."""
        await self.send_request(
            "md/unsubscribequote",
            body={"symbol": symbol},
            timeout=10.0,
        )
        logger.info("tradovate.md.unsubscribed_quotes", symbol=symbol)

    async def subscribe_dom(self, symbol: str) -> None:
        """Subscribe to depth of market updates for a symbol."""
        logger.info("tradovate.md.subscribing_dom", symbol=symbol)
        await self.send_request(
            "md/subscribeDOM",
            body={"symbol": symbol},
            timeout=10.0,
        )
        logger.info("tradovate.md.subscribed_dom", symbol=symbol)

    async def unsubscribe_dom(self, symbol: str) -> None:
        """Unsubscribe from DOM updates."""
        await self.send_request(
            "md/unsubscribeDOM",
            body={"symbol": symbol},
            timeout=10.0,
        )

    async def get_chart(
        self,
        symbol: str,
        chart_description: dict[str, Any],
        time_range: dict[str, Any],
    ) -> int:
        """Request chart data (historical + streaming).

        Args:
            symbol: Contract symbol (e.g., 'NQU5')
            chart_description: Dict with 'underlyingType', 'elementSize',
                              'elementSizeUnit', 'withHistogram'
            time_range: Dict with 'asFarAsTimestamp' and/or 'closestTimestamp',
                       or 'asMuchAsElements'

        Returns:
            Request ID for tracking chart data callbacks.

        Example chart_description for 5-minute candles:
            {"underlyingType": "MinuteBar", "elementSize": 5,
             "elementSizeUnit": "UnderlyingUnits", "withHistogram": False}

        Example time_range for last 200 bars:
            {"asMuchAsElements": 200}
        """
        req_id = self._next_request_id()
        self._chart_request_map[req_id] = symbol

        body = {
            "symbol": symbol,
            "chartDescription": chart_description,
            "timeRange": time_range,
        }

        logger.info(
            "tradovate.md.requesting_chart",
            symbol=symbol,
            request_id=req_id,
        )

        if not self._ws or not self._connected:
            raise ConnectionError("WebSocket not connected")

        frame = f"md/getchart\n{req_id}\n\n{json.dumps(body)}"
        await self._ws.send(frame)
        return req_id

    async def _handle_response(self, item: dict) -> None:
        """Override to also dispatch quote, chart, and DOM events."""
        # Handle standard responses first
        await super()._handle_response(item)

        # Dispatch market data events
        event_name = item.get("e")
        data = item.get("d")

        if not data:
            return

        if event_name == "md" and isinstance(data, dict):
            await self._dispatch_market_data(data)

    async def _dispatch_market_data(self, data: dict) -> None:
        """Dispatch market data based on the data content."""
        # Quote data
        if "entries" in data and "contractId" in data:
            await self._handle_quote_data(data)

        # Chart/candle data
        if "charts" in data:
            for chart in data["charts"]:
                await self._handle_chart_data(chart)

        # DOM data
        if "doms" in data:
            for dom in data["doms"]:
                await self._handle_dom_data(dom)

    async def _handle_quote_data(self, data: dict) -> None:
        """Parse and dispatch quote updates."""
        entries = data.get("entries", {})

        # Build Quote from entries
        bid = entries.get("Bid", {})
        ask = entries.get("Offer", {})  # Tradovate uses "Offer" not "Ask"
        trade = entries.get("Trade", {})
        total_vol = entries.get("TotalTradeVolume", {})

        # Find symbol from contract mapping or use contractId
        symbol = data.get("symbol", str(data.get("contractId", "")))

        quote = Quote(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            bid_price=bid.get("price", 0.0),
            ask_price=ask.get("price", 0.0),
            bid_size=bid.get("size", 0),
            ask_size=ask.get("size", 0),
            last_price=trade.get("price", 0.0),
            last_size=trade.get("size", 0),
            total_volume=total_vol.get("size", 0),
        )

        # Dispatch to all callbacks for this symbol
        for cb_symbol, callbacks in self._quote_callbacks.items():
            if cb_symbol in symbol or symbol in cb_symbol:
                for callback in callbacks:
                    try:
                        await callback(quote)
                    except Exception as e:
                        logger.error(
                            "tradovate.md.quote_callback_error",
                            symbol=symbol,
                            error=str(e),
                        )

    async def _handle_chart_data(self, chart_data: dict) -> None:
        """Parse and dispatch chart/candle data."""
        req_id = chart_data.get("id")
        bars = chart_data.get("bars", [])
        symbol = self._chart_request_map.get(req_id, "")

        # Check for end-of-history marker
        eoh = chart_data.get("eoh", False)

        for bar in bars:
            candle = Candle(
                timestamp=datetime.fromtimestamp(
                    bar.get("timestamp", 0) / 1000, tz=timezone.utc
                ),
                open=bar.get("open", 0.0),
                high=bar.get("high", 0.0),
                low=bar.get("low", 0.0),
                close=bar.get("close", 0.0),
                volume=bar.get("upVolume", 0) + bar.get("downVolume", 0),
                symbol=symbol,
            )

            if req_id in self._chart_callbacks:
                for callback in self._chart_callbacks[req_id]:
                    try:
                        await callback(candle)
                    except Exception as e:
                        logger.error(
                            "tradovate.md.chart_callback_error",
                            symbol=symbol,
                            error=str(e),
                        )

        if eoh:
            logger.info("tradovate.md.chart_eoh", symbol=symbol, request_id=req_id)

    async def _handle_dom_data(self, dom_data: dict) -> None:
        """Parse and dispatch DOM updates."""
        contract_id = dom_data.get("contractId")
        bids = dom_data.get("bids", [])
        asks = dom_data.get("offers", [])

        levels = []
        for bid in bids:
            levels.append(
                DOMLevel(
                    price=bid.get("price", 0.0),
                    bid_size=bid.get("size", 0),
                )
            )
        for ask in asks:
            # Find matching level or create new
            price = ask.get("price", 0.0)
            matched = False
            for level in levels:
                if level.price == price:
                    level.ask_size = ask.get("size", 0)
                    matched = True
                    break
            if not matched:
                levels.append(
                    DOMLevel(
                        price=price,
                        ask_size=ask.get("size", 0),
                    )
                )

        levels.sort(key=lambda x: x.price, reverse=True)

        snapshot = DOMSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol=str(contract_id),
            levels=levels,
        )

        # Dispatch to callbacks
        for cb_symbol, callbacks in self._dom_callbacks.items():
            for callback in callbacks:
                try:
                    await callback(snapshot)
                except Exception as e:
                    logger.error(
                        "tradovate.md.dom_callback_error",
                        error=str(e),
                    )
