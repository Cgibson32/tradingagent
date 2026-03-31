"""Claude API client wrapper with budget tracking and caching.

Wraps the Anthropic SDK with:
- Token counting and cost tracking ($5/day budget cap)
- Prompt caching with configurable TTL
- Structured JSON response parsing
- Graceful degradation when API is unavailable
- Separate models for real-time vs analysis tasks
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Approximate pricing (per 1M tokens)
MODEL_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
}


class BudgetExceededError(Exception):
    pass


class ClaudeClient:
    """Async wrapper around the Anthropic Claude API.

    Features:
    - Automatic model selection (Sonnet for real-time, Opus for analysis)
    - Token usage tracking and daily budget enforcement
    - Response caching to avoid duplicate API calls
    - Structured JSON output parsing with Pydantic validation
    - Graceful fallback when API is unavailable
    """

    def __init__(
        self,
        api_key: str,
        realtime_model: str = "claude-sonnet-4-20250514",
        analysis_model: str = "claude-opus-4-6",
        daily_budget_usd: float = 5.0,
        cache_ttl_seconds: int = 300,
        realtime_timeout: int = 10,
        analysis_timeout: int = 60,
    ):
        self._api_key = api_key
        self._realtime_model = realtime_model
        self._analysis_model = analysis_model
        self._daily_budget = daily_budget_usd
        self._cache_ttl = cache_ttl_seconds
        self._realtime_timeout = realtime_timeout
        self._analysis_timeout = analysis_timeout

        # Usage tracking
        self._daily_cost: float = 0.0
        self._daily_tokens_in: int = 0
        self._daily_tokens_out: int = 0
        self._total_calls: int = 0
        self._budget_reset_date: str = ""

        # Response cache: hash → (response, timestamp)
        self._cache: dict[str, tuple[dict, float]] = {}

        # Anthropic client (lazy init)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    @property
    def daily_cost(self) -> float:
        self._check_budget_reset()
        return self._daily_cost

    @property
    def budget_remaining(self) -> float:
        self._check_budget_reset()
        return self._daily_budget - self._daily_cost

    @property
    def is_over_budget(self) -> bool:
        return self.budget_remaining <= 0

    def _check_budget_reset(self) -> None:
        """Reset daily budget at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._budget_reset_date != today:
            self._daily_cost = 0.0
            self._daily_tokens_in = 0
            self._daily_tokens_out = 0
            self._budget_reset_date = today

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for a request."""
        pricing = MODEL_PRICING.get(model, {"input": 3.0, "output": 15.0})
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    def _cache_key(self, model: str, system: str, messages: list) -> str:
        """Generate a cache key from the request parameters."""
        content = json.dumps({"model": model, "system": system, "messages": messages}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached(self, key: str) -> dict | None:
        """Get a cached response if it exists and hasn't expired."""
        if key in self._cache:
            response, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return response
            else:
                del self._cache[key]
        return None

    async def ask(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        mode: str = "realtime",
        parse_json: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Send a prompt to Claude and get a structured response.

        Args:
            prompt: The user message
            system: System prompt
            model: Override model (defaults based on mode)
            mode: 'realtime' (Sonnet, fast) or 'analysis' (Opus, thorough)
            parse_json: If True, parse response as JSON
            use_cache: If True, check cache before calling API

        Returns:
            Dict with parsed response. On failure: {"error": "...", "fallback": True}
        """
        self._check_budget_reset()

        if self.is_over_budget:
            logger.warning("claude.over_budget", daily_cost=self._daily_cost, budget=self._daily_budget)
            return {"error": "Daily budget exceeded", "fallback": True}

        if model is None:
            model = self._realtime_model if mode == "realtime" else self._analysis_model

        timeout = self._realtime_timeout if mode == "realtime" else self._analysis_timeout

        messages = [{"role": "user", "content": prompt}]

        # Check cache
        if use_cache:
            cache_key = self._cache_key(model, system, messages)
            cached = self._get_cached(cache_key)
            if cached:
                logger.debug("claude.cache_hit", model=model)
                return cached

        # Call API
        start_time = time.time()
        try:
            client = self._get_client()
            response = await asyncio.wait_for(
                client.messages.create(
                    model=model,
                    max_tokens=2048,
                    system=system if system else "You are a trading analysis assistant. Always respond in valid JSON.",
                    messages=messages,
                ),
                timeout=timeout,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Track usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._estimate_cost(model, input_tokens, output_tokens)

            self._daily_cost += cost
            self._daily_tokens_in += input_tokens
            self._daily_tokens_out += output_tokens
            self._total_calls += 1

            logger.info(
                "claude.response",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=round(cost, 4),
                latency_ms=latency_ms,
                daily_cost=round(self._daily_cost, 4),
            )

            # Extract text content
            text = response.content[0].text if response.content else ""

            # Parse JSON
            if parse_json:
                result = self._parse_json_response(text)
            else:
                result = {"text": text}

            result["_meta"] = {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "latency_ms": latency_ms,
            }

            # Cache the result
            if use_cache:
                self._cache[cache_key] = (result, time.time())

            return result

        except asyncio.TimeoutError:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.warning("claude.timeout", model=model, timeout=timeout, latency_ms=latency_ms)
            return {"error": "Timeout", "fallback": True}

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("claude.error", model=model, error=str(e), latency_ms=latency_ms)
            return {"error": str(e), "fallback": True}

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON from Claude's response, handling markdown code blocks."""
        text = text.strip()

        # Strip markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            logger.warning("claude.json_parse_failed", text=text[:200])
            return {"error": "JSON parse failed", "raw_text": text, "fallback": True}

    def get_usage_summary(self) -> dict:
        """Get current usage statistics."""
        self._check_budget_reset()
        return {
            "daily_cost_usd": round(self._daily_cost, 4),
            "budget_remaining_usd": round(self.budget_remaining, 4),
            "daily_tokens_in": self._daily_tokens_in,
            "daily_tokens_out": self._daily_tokens_out,
            "total_calls": self._total_calls,
            "cache_size": len(self._cache),
        }
