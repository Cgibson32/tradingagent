"""Tradovate Trading WebSocket client.

Implements the Tradovate custom text-based WebSocket protocol for
real-time account, order, and position updates.

Protocol:
- Outgoing: "<endpoint>\\n<requestId>\\n\\n<jsonBody>"
- Incoming frames: 'o' (open), 'a[...]' (data), 'h' (heartbeat), 'c[...]' (close)
- Response: {"s": 200, "i": <requestId>, "d": <data>}
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import structlog
import websockets
from websockets.asyncio.client import ClientConnection

logger = structlog.get_logger(__name__)

RECONNECT_BACKOFF_BASE = 2.0
MAX_RECONNECT_DELAY = 60.0


class TradovateWebSocket:
    """Base WebSocket client for Tradovate's custom text protocol.

    Handles connection lifecycle, frame parsing, request/response correlation,
    heartbeat management, and automatic reconnection.
    """

    def __init__(self, url: str, token: str):
        self._url = url
        self._token = token
        self._ws: ClientConnection | None = None
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._event_handlers: dict[str, list[Callable]] = {}
        self._connected = False
        self._running = False
        self._receive_task: asyncio.Task | None = None
        self._reconnect_count = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def update_token(self, token: str) -> None:
        """Update the token for re-authorization after renewal."""
        self._token = token

    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler.

        Events: 'position', 'order', 'account', 'fill', 'quote', 'chart', 'dom'
        """
        self._event_handlers.setdefault(event, []).append(handler)

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def connect(self) -> None:
        """Connect to the WebSocket and start receiving messages."""
        self._running = True
        self._reconnect_count = 0
        await self._connect()

    async def _connect(self) -> None:
        """Internal connection with reconnection logic."""
        while self._running:
            try:
                logger.info("tradovate.ws.connecting", url=self._url)
                self._ws = await websockets.connect(self._url)
                self._connected = True
                self._reconnect_count = 0
                logger.info("tradovate.ws.connected")

                # Start receiving messages
                self._receive_task = asyncio.create_task(self._receive_loop())
                await self._receive_task

            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                self._connected = False
                if not self._running:
                    break
                delay = min(
                    RECONNECT_BACKOFF_BASE * (2**self._reconnect_count),
                    MAX_RECONNECT_DELAY,
                )
                self._reconnect_count += 1
                logger.warning(
                    "tradovate.ws.disconnected",
                    error=str(e),
                    reconnect_in=delay,
                )
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break

    async def _receive_loop(self) -> None:
        """Main loop that receives and parses WebSocket frames."""
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                await self._handle_frame(message)
        except websockets.ConnectionClosed:
            self._connected = False
            raise
        except asyncio.CancelledError:
            pass

    async def _handle_frame(self, frame: str) -> None:
        """Parse and dispatch a Tradovate WebSocket frame.

        Frame types:
        - 'o': Connection opened → authorize
        - 'a[...]': Array of response/event objects
        - 'h': Heartbeat → respond with '[]'
        - 'c[code,"reason"]': Connection closed
        """
        if not frame:
            return

        first_char = frame[0]

        if first_char == "o":
            # Connection opened — send authorization
            logger.info("tradovate.ws.open_frame")
            await self._authorize()

        elif first_char == "a":
            # Data frame — parse JSON array
            try:
                data = json.loads(frame[1:])
                for item in data:
                    await self._handle_response(item)
            except json.JSONDecodeError:
                logger.error("tradovate.ws.invalid_json", frame=frame[:200])

        elif first_char == "h":
            # Heartbeat — respond with empty array
            if self._ws:
                await self._ws.send("[]")

        elif first_char == "c":
            # Connection closed by server
            try:
                close_data = json.loads(frame[1:])
                logger.warning("tradovate.ws.server_close", data=close_data)
            except json.JSONDecodeError:
                logger.warning("tradovate.ws.server_close", raw=frame)

    async def _authorize(self) -> None:
        """Send authorization frame after connection opens."""
        req_id = self._next_request_id()
        frame = f"authorize\n{req_id}\n\n{self._token}"
        logger.info("tradovate.ws.authorizing", request_id=req_id)
        await self._ws.send(frame)

    async def _handle_response(self, item: dict) -> None:
        """Handle a single response object from an 'a[...]' frame."""
        status = item.get("s")
        req_id = item.get("i")
        data = item.get("d")
        event_name = item.get("e")

        # Resolve pending request future
        if req_id and req_id in self._pending_requests:
            future = self._pending_requests.pop(req_id)
            if not future.done():
                if status and status >= 400:
                    future.set_exception(
                        RuntimeError(f"Tradovate WS error {status}: {data}")
                    )
                else:
                    future.set_result(data)

        # Dispatch events (real-time updates from syncrequest)
        if event_name and event_name in self._event_handlers:
            for handler in self._event_handlers[event_name]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(
                        "tradovate.ws.handler_error",
                        event=event_name,
                        error=str(e),
                    )

        # Also check for entity type in data for sync updates
        if isinstance(data, dict):
            entity_type = data.get("entityType")
            if entity_type and entity_type.lower() in self._event_handlers:
                for handler in self._event_handlers[entity_type.lower()]:
                    try:
                        await handler(data)
                    except Exception as e:
                        logger.error(
                            "tradovate.ws.handler_error",
                            entity=entity_type,
                            error=str(e),
                        )
        elif isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    entity_type = entry.get("entityType")
                    if entity_type and entity_type.lower() in self._event_handlers:
                        for handler in self._event_handlers[entity_type.lower()]:
                            try:
                                await handler(entry)
                            except Exception as e:
                                logger.error(
                                    "tradovate.ws.handler_error",
                                    entity=entity_type,
                                    error=str(e),
                                )

    async def send_request(
        self, endpoint: str, body: dict | None = None, timeout: float = 10.0
    ) -> Any:
        """Send a request and wait for the correlated response.

        Args:
            endpoint: The Tradovate WS endpoint (e.g., 'user/syncrequest')
            body: Optional JSON body
            timeout: Seconds to wait for response

        Returns:
            Response data from the server.
        """
        if not self._ws or not self._connected:
            raise ConnectionError("WebSocket not connected")

        req_id = self._next_request_id()
        body_str = json.dumps(body) if body else ""
        frame = f"{endpoint}\n{req_id}\n\n{body_str}"

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[req_id] = future

        await self._ws.send(frame)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(req_id, None)
            raise TimeoutError(f"Timeout waiting for response to {endpoint}")

    async def sync_request(self) -> Any:
        """Subscribe to real-time account/order/position updates.

        This is the primary method for getting live updates on the trading WS.
        """
        return await self.send_request(
            "user/syncrequest",
            body={"users": [self._token]},
            timeout=15.0,
        )

    async def close(self) -> None:
        """Disconnect and clean up."""
        self._running = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._connected = False

        # Cancel any pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()


class TradingWebSocket(TradovateWebSocket):
    """Trading-specific WebSocket client.

    Extends the base with convenience methods for order and position monitoring.
    """

    async def start_sync(self) -> None:
        """Start real-time sync of account data.

        After calling this, registered event handlers will receive updates for:
        - 'order': Order status changes
        - 'position': Position changes
        - 'account': Account balance updates
        - 'fill': Trade fills
        """
        try:
            result = await self.sync_request()
            logger.info("tradovate.trading_ws.sync_started", data_keys=type(result).__name__)
        except Exception as e:
            logger.error("tradovate.trading_ws.sync_failed", error=str(e))
            raise
