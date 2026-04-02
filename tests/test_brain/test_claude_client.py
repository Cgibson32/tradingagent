"""Tests for Claude client wrapper."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.brain.claude_client import ClaudeClient


@pytest.fixture
def client():
    return ClaudeClient(
        api_key="test-key",
        daily_budget_usd=5.0,
        cache_ttl_seconds=60,
    )


class TestClaudeClient:
    def test_initial_state(self, client):
        assert client.daily_cost == 0.0
        assert client.budget_remaining == 5.0
        assert client.is_over_budget is False

    def test_budget_tracking(self, client):
        from datetime import datetime, timezone
        client._budget_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        client._daily_cost = 4.5
        assert client.budget_remaining == pytest.approx(0.5)
        assert client.is_over_budget is False

        client._daily_cost = 5.0
        assert client.is_over_budget is True

    @pytest.mark.asyncio
    async def test_over_budget_returns_fallback(self, client):
        from datetime import datetime, timezone
        client._budget_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        client._daily_cost = 10.0
        result = await client.ask("test prompt")
        assert result["fallback"] is True
        assert "budget" in result["error"].lower()

    def test_json_parsing_clean(self, client):
        text = '{"regime": "trending_bullish", "confidence": 0.8}'
        result = client._parse_json_response(text)
        assert result["regime"] == "trending_bullish"
        assert result["confidence"] == 0.8

    def test_json_parsing_with_markdown(self, client):
        text = '```json\n{"regime": "ranging"}\n```'
        result = client._parse_json_response(text)
        assert result["regime"] == "ranging"

    def test_json_parsing_embedded(self, client):
        text = 'Here is my analysis: {"decision": "approve"} as requested.'
        result = client._parse_json_response(text)
        assert result["decision"] == "approve"

    def test_json_parsing_failure(self, client):
        text = "This is not JSON at all"
        result = client._parse_json_response(text)
        assert result.get("fallback") is True

    def test_cache_key_deterministic(self, client):
        key1 = client._cache_key("model", "sys", [{"role": "user", "content": "hi"}])
        key2 = client._cache_key("model", "sys", [{"role": "user", "content": "hi"}])
        assert key1 == key2

    def test_cache_key_differs(self, client):
        key1 = client._cache_key("model", "sys", [{"role": "user", "content": "hi"}])
        key2 = client._cache_key("model", "sys", [{"role": "user", "content": "bye"}])
        assert key1 != key2

    def test_usage_summary(self, client):
        summary = client.get_usage_summary()
        assert "daily_cost_usd" in summary
        assert "budget_remaining_usd" in summary
        assert "total_calls" in summary

    def test_cost_estimation(self, client):
        cost = client._estimate_cost("claude-sonnet-4-20250514", 1000, 500)
        assert cost > 0
        # Sonnet: 1000 * 3.0/1M + 500 * 15.0/1M = 0.003 + 0.0075 = 0.0105
        assert abs(cost - 0.0105) < 0.001
