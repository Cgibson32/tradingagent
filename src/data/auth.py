"""Tradovate OAuth2 token lifecycle management.

Handles initial authentication, automatic token renewal, and provides
tokens for both REST and WebSocket connections.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import httpx
import structlog

logger = structlog.get_logger(__name__)


class TokenManager:
    """Manages Tradovate API tokens with automatic renewal.

    Tradovate tokens expire after ~60 minutes. This manager:
    - Obtains initial tokens via REST API
    - Schedules automatic renewal at a configurable interval (default 45min)
    - Provides separate tokens for REST (accessToken) and market data (mdAccessToken)
    - Emits callbacks when tokens refresh so WebSocket clients can re-authorize
    """

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        app_id: str,
        client_id: str,
        client_secret: str,
        renew_minutes: int = 45,
    ):
        self._base_url = base_url
        self._username = username
        self._password = password
        self._app_id = app_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._renew_minutes = renew_minutes

        self._access_token: str | None = None
        self._md_access_token: str | None = None
        self._user_id: int | None = None
        self._expiration_time: datetime | None = None
        self._renewal_task: asyncio.Task | None = None
        self._on_refresh_callbacks: list = []
        self._http_client: httpx.AsyncClient | None = None

    @property
    def access_token(self) -> str | None:
        return self._access_token

    @property
    def md_access_token(self) -> str | None:
        return self._md_access_token

    @property
    def user_id(self) -> int | None:
        return self._user_id

    @property
    def is_authenticated(self) -> bool:
        return self._access_token is not None

    @property
    def is_expired(self) -> bool:
        if self._expiration_time is None:
            return True
        return datetime.now(timezone.utc) >= self._expiration_time

    def on_refresh(self, callback) -> None:
        """Register a callback to be called when tokens are refreshed.

        Callback signature: async def callback(access_token: str, md_access_token: str)
        """
        self._on_refresh_callbacks.append(callback)

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def authenticate(self) -> None:
        """Obtain initial access tokens from Tradovate."""
        client = await self._get_http_client()
        url = f"{self._base_url}/auth/accesstokenrequest"
        payload = {
            "name": self._username,
            "password": self._password,
            "appId": self._app_id,
            "appVersion": "1.0",
            "cid": self._client_id,
            "sec": self._client_secret,
        }

        logger.info("tradovate.auth.requesting_token", url=url)

        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                "tradovate.auth.failed",
                status_code=e.response.status_code,
                body=e.response.text,
            )
            raise
        except httpx.RequestError as e:
            logger.error("tradovate.auth.request_error", error=str(e))
            raise

        self._access_token = data.get("accessToken")
        self._md_access_token = data.get("mdAccessToken")
        self._user_id = data.get("userId")
        expiration_str = data.get("expirationTime")

        if not self._access_token:
            error_text = data.get("errorText", "Unknown error")
            raise RuntimeError(f"Tradovate authentication failed: {error_text}")

        if expiration_str:
            # Tradovate returns ISO format expiration
            self._expiration_time = datetime.fromisoformat(
                expiration_str.replace("Z", "+00:00")
            )

        logger.info(
            "tradovate.auth.success",
            user_id=self._user_id,
            expires_at=str(self._expiration_time),
        )

        # Start automatic renewal
        self._start_renewal_loop()

    async def renew(self) -> None:
        """Renew the access token before it expires."""
        if not self._access_token:
            logger.warning("tradovate.auth.renew_no_token")
            await self.authenticate()
            return

        client = await self._get_http_client()
        url = f"{self._base_url}/auth/renewaccesstoken"

        try:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {self._access_token}"},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.warning(
                "tradovate.auth.renew_failed",
                status_code=e.response.status_code,
                body=e.response.text,
            )
            # If renewal fails, try full re-authentication
            await self.authenticate()
            return

        self._access_token = data.get("accessToken", self._access_token)
        self._md_access_token = data.get("mdAccessToken", self._md_access_token)
        expiration_str = data.get("expirationTime")

        if expiration_str:
            self._expiration_time = datetime.fromisoformat(
                expiration_str.replace("Z", "+00:00")
            )

        logger.info("tradovate.auth.renewed", expires_at=str(self._expiration_time))

        # Notify WebSocket clients to re-authorize
        for callback in self._on_refresh_callbacks:
            try:
                await callback(self._access_token, self._md_access_token)
            except Exception as e:
                logger.error("tradovate.auth.refresh_callback_error", error=str(e))

    async def ensure_valid(self) -> str:
        """Ensure we have a valid token, renewing if necessary.

        Returns:
            Valid access token string.
        """
        if not self._access_token or self.is_expired:
            await self.renew()
        return self._access_token

    def _start_renewal_loop(self) -> None:
        """Start background task to auto-renew tokens."""
        if self._renewal_task and not self._renewal_task.done():
            self._renewal_task.cancel()
        self._renewal_task = asyncio.create_task(self._renewal_loop())

    async def _renewal_loop(self) -> None:
        """Background loop that renews tokens before expiry."""
        while True:
            await asyncio.sleep(self._renew_minutes * 60)
            try:
                await self.renew()
            except Exception as e:
                logger.error("tradovate.auth.renewal_loop_error", error=str(e))
                # Retry in 60 seconds
                await asyncio.sleep(60)

    async def close(self) -> None:
        """Clean up resources."""
        if self._renewal_task and not self._renewal_task.done():
            self._renewal_task.cancel()
            try:
                await self._renewal_task
            except asyncio.CancelledError:
                pass
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
