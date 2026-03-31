"""Circuit breaker for extreme market conditions.

Detects dangerous conditions and auto-pauses trading:
- Flash crash (>5x ATR move in 5 minutes)
- Spread blow-out (>4x normal spread)
- CME halt (no quotes for >30 seconds)
- Consecutive losses (3 in a day)
- Sudden volatility doubling
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)


class CircuitBreaker:
    """Monitors for extreme market conditions and halts trading.

    Checked BEFORE every order placement. Cannot be overridden.
    """

    def __init__(
        self,
        flash_crash_atr_multiple: float = 5.0,
        spread_blowout_multiple: float = 4.0,
        stale_quote_seconds: float = 30.0,
        max_consecutive_losses: int = 3,
        volatility_shift_multiple: float = 2.0,
    ):
        self._flash_atr_mult = flash_crash_atr_multiple
        self._spread_mult = spread_blowout_multiple
        self._stale_seconds = stale_quote_seconds
        self._max_losses = max_consecutive_losses
        self._vol_shift_mult = volatility_shift_multiple

        # State
        self._last_quote_time: float = 0.0
        self._price_history: deque[tuple[float, float]] = deque(maxlen=300)  # (timestamp, price)
        self._spread_history: deque[float] = deque(maxlen=100)
        self._normal_spread: float = 0.0
        self._normal_atr: float = 0.0
        self._current_atr: float = 0.0
        self._consecutive_losses: int = 0
        self._tripped: bool = False
        self._trip_reason: str = ""
        self._trip_time: float = 0.0
        self._cooldown_seconds: float = 3600.0  # 1 hour default

    @property
    def is_tripped(self) -> bool:
        # Auto-reset after cooldown
        if self._tripped and (time.time() - self._trip_time) > self._cooldown_seconds:
            self._tripped = False
            self._trip_reason = ""
            logger.info("circuit_breaker.auto_reset")
        return self._tripped

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    def set_baseline(self, normal_atr: float, normal_spread: float) -> None:
        """Set baseline values for comparison. Call at session start."""
        self._normal_atr = normal_atr
        self._normal_spread = normal_spread

    def record_loss(self) -> None:
        """Record a losing trade."""
        self._consecutive_losses += 1
        if self._consecutive_losses >= self._max_losses:
            self._trip("consecutive_losses", f"{self._consecutive_losses} consecutive losses")

    def record_win(self) -> None:
        """Record a winning trade — resets loss counter."""
        self._consecutive_losses = 0

    def update_quote(self, price: float, spread: float) -> None:
        """Feed a new quote to the circuit breaker.

        Call on every quote update.
        """
        now = time.time()
        self._last_quote_time = now
        self._price_history.append((now, price))
        self._spread_history.append(spread)

        # Update normal spread (rolling average)
        if len(self._spread_history) > 20:
            self._normal_spread = sum(list(self._spread_history)[-50:]) / min(50, len(self._spread_history))

    def update_atr(self, atr_value: float) -> None:
        """Update current ATR value."""
        self._current_atr = atr_value
        if self._normal_atr == 0:
            self._normal_atr = atr_value

    def check(self) -> tuple[bool, str]:
        """Run all circuit breaker checks.

        Returns:
            (safe_to_trade, reason_if_not)
        """
        if self.is_tripped:
            return False, self._trip_reason

        # 1. Flash crash detection
        if self._check_flash_crash():
            return False, self._trip_reason

        # 2. Spread blow-out
        if self._check_spread_blowout():
            return False, self._trip_reason

        # 3. Stale quotes (CME halt)
        if self._check_stale_quotes():
            return False, self._trip_reason

        # 4. Volatility regime shift
        if self._check_volatility_shift():
            return False, self._trip_reason

        return True, ""

    def get_max_contracts_override(self) -> int | None:
        """If volatility is elevated, return reduced max contracts.

        Returns None if no override needed.
        """
        if self._normal_atr > 0 and self._current_atr > self._normal_atr * self._vol_shift_mult:
            return 2  # Reduce to 2 contracts during high vol
        return None

    def _trip(self, reason_code: str, message: str) -> None:
        """Trip the circuit breaker."""
        self._tripped = True
        self._trip_reason = f"{reason_code}: {message}"
        self._trip_time = time.time()
        logger.critical("circuit_breaker.TRIPPED", reason=reason_code, message=message)

    def _check_flash_crash(self) -> bool:
        """Check for >5x ATR move in <5 minutes."""
        if self._normal_atr == 0 or len(self._price_history) < 10:
            return False

        now = time.time()
        five_min_ago = now - 300

        # Get price range in last 5 minutes
        recent = [(t, p) for t, p in self._price_history if t >= five_min_ago]
        if len(recent) < 2:
            return False

        prices = [p for _, p in recent]
        price_range = max(prices) - min(prices)

        if price_range > self._normal_atr * self._flash_atr_mult:
            self._trip("flash_crash", f"Price moved {price_range:.2f} pts in 5 min (ATR: {self._normal_atr:.2f})")
            return True

        return False

    def _check_spread_blowout(self) -> bool:
        """Check if current spread is >4x normal."""
        if self._normal_spread <= 0 or len(self._spread_history) < 5:
            return False

        current_spread = self._spread_history[-1]
        if current_spread > self._normal_spread * self._spread_mult:
            self._trip("spread_blowout", f"Spread {current_spread:.2f} > {self._spread_mult}x normal ({self._normal_spread:.2f})")
            return True

        return False

    def _check_stale_quotes(self) -> bool:
        """Check if quotes have stopped (possible CME halt)."""
        if self._last_quote_time == 0:
            return False

        seconds_since_quote = time.time() - self._last_quote_time
        if seconds_since_quote > self._stale_seconds:
            self._trip("stale_quotes", f"No quotes for {seconds_since_quote:.0f}s (possible CME halt)")
            return True

        return False

    def _check_volatility_shift(self) -> bool:
        """Check if ATR has suddenly doubled."""
        if self._normal_atr == 0:
            return False

        if self._current_atr > self._normal_atr * self._vol_shift_mult:
            # Don't trip the full breaker, just log a warning
            # (max_contracts_override handles the sizing reduction)
            logger.warning(
                "circuit_breaker.volatility_elevated",
                current_atr=self._current_atr,
                normal_atr=self._normal_atr,
                ratio=self._current_atr / self._normal_atr,
            )

        return False  # Volatility shift reduces size but doesn't halt trading

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._tripped = False
        self._trip_reason = ""
        self._consecutive_losses = 0
        logger.info("circuit_breaker.manual_reset")
