"""Tests for Tradovate WebSocket protocol handling."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.websocket import TradovateWebSocket, TradingWebSocket


class TestFrameParsing:
    """Test the Tradovate custom text protocol frame parsing."""

    @pytest.fixture
    def ws(self):
        return TradovateWebSocket(
            url="wss://demo.tradovateapi.com/v1/websocket",
            token="test_token",
        )

    @pytest.mark.asyncio
    async def test_heartbeat_response(self, ws):
        """Heartbeat 'h' frame should trigger '[]' response."""
        ws._ws = AsyncMock()
        ws._connected = True

        await ws._handle_frame("h")

        ws._ws.send.assert_called_once_with("[]")

    @pytest.mark.asyncio
    async def test_open_frame_triggers_auth(self, ws):
        """'o' frame should trigger authorization."""
        ws._ws = AsyncMock()
        ws._connected = True

        await ws._handle_frame("o")

        # Should have sent an authorize frame
        ws._ws.send.assert_called_once()
        sent = ws._ws.send.call_args[0][0]
        assert sent.startswith("authorize\n")
        assert "test_token" in sent

    @pytest.mark.asyncio
    async def test_data_frame_parsing(self, ws):
        """'a[...]' frames should be parsed as JSON arrays."""
        ws._ws = AsyncMock()
        ws._connected = True

        # Set up a pending request
        future = asyncio.get_event_loop().create_future()
        ws._pending_requests[1] = future

        frame = 'a[{"s":200,"i":1,"d":{"test":"data"}}]'
        await ws._handle_frame(frame)

        result = future.result()
        assert result == {"test": "data"}

    @pytest.mark.asyncio
    async def test_error_response(self, ws):
        """Error responses should set exception on the future."""
        ws._ws = AsyncMock()
        ws._connected = True

        future = asyncio.get_event_loop().create_future()
        ws._pending_requests[2] = future

        frame = 'a[{"s":404,"i":2,"d":"Not found"}]'
        await ws._handle_frame(frame)

        with pytest.raises(RuntimeError, match="404"):
            future.result()

    @pytest.mark.asyncio
    async def test_close_frame(self, ws):
        """'c[...]' frame should be handled without error."""
        ws._ws = AsyncMock()
        ws._connected = True

        # Should not raise
        await ws._handle_frame('c[1000,"Normal closure"]')

    @pytest.mark.asyncio
    async def test_invalid_json_frame(self, ws):
        """Invalid JSON in 'a' frame should be handled gracefully."""
        ws._ws = AsyncMock()
        ws._connected = True

        # Should not raise, just log error
        await ws._handle_frame("a[invalid json")

    def test_request_id_increment(self, ws):
        """Request IDs should auto-increment."""
        id1 = ws._next_request_id()
        id2 = ws._next_request_id()
        assert id2 == id1 + 1

    def test_update_token(self, ws):
        """Token update should be reflected."""
        ws.update_token("new_token")
        assert ws._token == "new_token"

    @pytest.mark.asyncio
    async def test_event_handler_dispatch(self, ws):
        """Event handlers should be called for matching events."""
        handler = AsyncMock()
        ws.on("position", handler)

        ws._ws = AsyncMock()
        ws._connected = True

        frame = 'a[{"e":"position","d":{"netPos":2}}]'
        await ws._handle_frame(frame)

        handler.assert_called_once_with({"netPos": 2})

    @pytest.mark.asyncio
    async def test_send_request_format(self, ws):
        """send_request should format the Tradovate protocol correctly."""
        ws._ws = AsyncMock()
        ws._connected = True

        # Don't await the future — just check the sent frame
        task = asyncio.create_task(
            ws.send_request("user/syncrequest", body={"users": ["token"]}, timeout=1.0)
        )

        await asyncio.sleep(0.1)

        ws._ws.send.assert_called_once()
        sent = ws._ws.send.call_args[0][0]
        parts = sent.split("\n")
        assert parts[0] == "user/syncrequest"
        assert json.loads(parts[3]) == {"users": ["token"]}

        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, TimeoutError):
            pass
