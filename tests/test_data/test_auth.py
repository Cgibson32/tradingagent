"""Tests for Tradovate authentication."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.auth import TokenManager


@pytest.fixture
def token_manager():
    return TokenManager(
        base_url="https://demo.tradovateapi.com/v1",
        username="test_user",
        password="test_pass",
        app_id="test_app",
        client_id="test_client",
        client_secret="test_secret",
        renew_minutes=45,
    )


class TestTokenManager:
    def test_initial_state(self, token_manager):
        assert token_manager.access_token is None
        assert token_manager.md_access_token is None
        assert token_manager.user_id is None
        assert token_manager.is_authenticated is False
        assert token_manager.is_expired is True

    @pytest.mark.asyncio
    async def test_authenticate_success(self, token_manager):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "accessToken": "test_token_123",
            "mdAccessToken": "test_md_token_456",
            "userId": 12345,
            "expirationTime": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            await token_manager.authenticate()

        assert token_manager.access_token == "test_token_123"
        assert token_manager.md_access_token == "test_md_token_456"
        assert token_manager.user_id == 12345
        assert token_manager.is_authenticated is True
        assert token_manager.is_expired is False

        await token_manager.close()

    @pytest.mark.asyncio
    async def test_authenticate_failure(self, token_manager):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "errorText": "Invalid credentials",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(RuntimeError, match="Invalid credentials"):
                await token_manager.authenticate()

        await token_manager.close()

    def test_on_refresh_callback(self, token_manager):
        callback = AsyncMock()
        token_manager.on_refresh(callback)
        assert len(token_manager._on_refresh_callbacks) == 1

    @pytest.mark.asyncio
    async def test_ensure_valid_when_expired(self, token_manager):
        # When no token exists, ensure_valid should call renew (which calls authenticate)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "accessToken": "refreshed_token",
            "mdAccessToken": "refreshed_md",
            "userId": 12345,
            "expirationTime": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            token = await token_manager.ensure_valid()

        assert token == "refreshed_token"
        await token_manager.close()
