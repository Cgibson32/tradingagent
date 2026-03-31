"""IMMUTABLE risk guardrails — Tradeify prop firm rules.

This is the HARD safety layer. It CANNOT be overridden by config,
LLM decisions, or any other component. It can BLOCK orders but
never PLACE them.

These rules are the final gatekeeper before any order reaches the broker.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import structlog

logger = structlog.get_logger(__name__)

ET = ZoneInfo("America/New_York")

# ── IMMUTABLE CONSTANTS (never change these) ──────────
# These are hardcoded intentionally. They must not be configurable.

_MAX_RISK_PER_TRADE_PCT = 0.015     # 1.5% hard ceiling
_MAX_DAILY_LOSS = 1750.0            # Tradeify 150K Select Daily
_DAILY_LOSS_BUFFER = 250.0          # Stop at $1,500 effective
_MAX_TRAILING_DRAWDOWN = 4500.0     # Tradeify EOD trailing
_DRAWDOWN_BUFFER = 500.0            # Stop at $4,000 effective
_MAX_CONTRACTS = 4                   # Self-imposed conservative limit
_MAX_CONCURRENT_POSITIONS = 2
_MAX_TRADES_PER_DAY = 6
_COOLDOWN_AFTER_LOSS_MINUTES = 15
_NEWS_BLACKOUT_MINUTES = 15
_TRADING_START = time(9, 30)
_TRADING_END = time(16, 0)
_NO_NEW_ENTRIES_AFTER = time(15, 0)
_CLOSE_ALL_TIME = time(15, 45)
_DRAWDOWN_LOCK_THRESHOLD = 100.0    # Lock when equity reaches start + $100
_ACCOUNT_SIZE = 150000
_EMERGENCY_EXIT_BUFFER = 100.0      # Close everything if within $100 of floor


class RiskViolation:
    """Describes a risk rule violation."""

    def __init__(self, rule: str, message: str, severity: str = "block"):
        self.rule = rule
        self.message = message
        self.severity = severity  # "block" or "warn"

    def __repr__(self):
        return f"RiskViolation({self.rule}: {self.message})"


class RiskGuard:
    """Enforces Tradeify prop firm rules. Immutable safety layer.

    This module wraps the execution engine. Every order must pass
    through can_trade() and validate_order() before reaching the broker.
    """

    def __init__(self):
        # Tracked state (updated by the orchestrator)
        self._daily_pnl: float = 0.0
        self._equity: float = _ACCOUNT_SIZE
        self._high_water_mark: float = _ACCOUNT_SIZE
        self._drawdown_floor: float = _ACCOUNT_SIZE - _MAX_TRAILING_DRAWDOWN + _DRAWDOWN_BUFFER
        self._drawdown_locked: bool = False
        self._trades_today: int = 0
        self._consecutive_losses: int = 0
        self._last_loss_time: datetime | None = None
        self._open_position_count: int = 0
        self._upcoming_news_times: list[datetime] = []

    # ── State updates (called by orchestrator) ──

    def update_pnl(self, daily_pnl: float) -> None:
        self._daily_pnl = daily_pnl

    def update_equity(self, equity: float) -> None:
        self._equity = equity
        # Check drawdown lock
        if not self._drawdown_locked and equity >= _ACCOUNT_SIZE + _DRAWDOWN_LOCK_THRESHOLD:
            self._drawdown_locked = True
            self._drawdown_floor = equity - _MAX_TRAILING_DRAWDOWN + _DRAWDOWN_BUFFER
            logger.info("risk_guard.drawdown_locked", floor=self._drawdown_floor)

    def update_high_water_mark(self, hwm: float) -> None:
        self._high_water_mark = hwm
        if not self._drawdown_locked:
            self._drawdown_floor = hwm - _MAX_TRAILING_DRAWDOWN + _DRAWDOWN_BUFFER

    def record_trade(self, is_loss: bool) -> None:
        self._trades_today += 1
        if is_loss:
            self._consecutive_losses += 1
            self._last_loss_time = datetime.now(ET)
        else:
            self._consecutive_losses = 0

    def set_open_positions(self, count: int) -> None:
        self._open_position_count = count

    def set_news_times(self, times: list[datetime]) -> None:
        self._upcoming_news_times = times

    def reset_daily(self) -> None:
        """Reset daily counters. Called at 6:00 PM ET."""
        self._daily_pnl = 0.0
        self._trades_today = 0
        self._consecutive_losses = 0
        self._last_loss_time = None

    @property
    def effective_daily_limit(self) -> float:
        return _MAX_DAILY_LOSS - _DAILY_LOSS_BUFFER

    @property
    def effective_drawdown_limit(self) -> float:
        return _MAX_TRAILING_DRAWDOWN - _DRAWDOWN_BUFFER

    @property
    def drawdown_floor(self) -> float:
        return self._drawdown_floor

    # ── Core checks ──

    def can_trade(self, now: datetime | None = None) -> tuple[bool, list[RiskViolation]]:
        """Check if trading is currently allowed.

        Returns:
            (allowed, list of violations)
        """
        if now is None:
            now = datetime.now(ET)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        elif now.tzinfo != ET:
            now = now.astimezone(ET)

        violations: list[RiskViolation] = []

        # 1. Trading hours
        now_time = now.time()
        if now_time < _TRADING_START or now_time >= _TRADING_END:
            violations.append(RiskViolation("trading_hours", f"Outside trading hours ({_TRADING_START}-{_TRADING_END} ET)"))

        # 2. No new entries after 3:00 PM
        if now_time >= _NO_NEW_ENTRIES_AFTER:
            violations.append(RiskViolation("no_new_entries", "No new entries after 3:00 PM ET"))

        # 3. Daily loss limit
        if self._daily_pnl <= -self.effective_daily_limit:
            violations.append(RiskViolation("daily_loss", f"Daily loss limit reached: ${self._daily_pnl:.2f} (limit: -${self.effective_daily_limit:.2f})"))

        # 4. Trailing drawdown
        if self._equity <= self._drawdown_floor:
            violations.append(RiskViolation("drawdown", f"Equity ${self._equity:.2f} at/below drawdown floor ${self._drawdown_floor:.2f}"))

        # 5. Max trades per day
        if self._trades_today >= _MAX_TRADES_PER_DAY:
            violations.append(RiskViolation("max_trades", f"Max trades per day reached: {self._trades_today}"))

        # 6. Max concurrent positions
        if self._open_position_count >= _MAX_CONCURRENT_POSITIONS:
            violations.append(RiskViolation("max_positions", f"Max concurrent positions: {self._open_position_count}"))

        # 7. Cooldown after loss
        if self._last_loss_time:
            minutes_since_loss = (now - self._last_loss_time).total_seconds() / 60
            if minutes_since_loss < _COOLDOWN_AFTER_LOSS_MINUTES:
                remaining = _COOLDOWN_AFTER_LOSS_MINUTES - minutes_since_loss
                violations.append(RiskViolation("cooldown", f"Cooldown: {remaining:.0f} min remaining after loss"))

        # 8. Consecutive losses
        if self._consecutive_losses >= 3:
            violations.append(RiskViolation("consecutive_losses", f"{self._consecutive_losses} consecutive losses — pausing"))

        # 9. News blackout
        for news_time in self._upcoming_news_times:
            if news_time.tzinfo is None:
                news_time = news_time.replace(tzinfo=ET)
            minutes_to_news = (news_time - now).total_seconds() / 60
            if abs(minutes_to_news) < _NEWS_BLACKOUT_MINUTES:
                violations.append(RiskViolation("news_blackout", f"High-impact news within {abs(minutes_to_news):.0f} minutes"))
                break

        allowed = len(violations) == 0
        if not allowed:
            logger.warning("risk_guard.blocked", violations=[str(v) for v in violations])

        return allowed, violations

    def validate_order(
        self, qty: int, entry_price: float, stop_loss: float, account_balance: float
    ) -> tuple[bool, list[RiskViolation]]:
        """Validate a specific order against all rules.

        Args:
            qty: Number of contracts
            entry_price: Planned entry price
            stop_loss: Stop loss price
            account_balance: Current account equity

        Returns:
            (valid, list of violations)
        """
        violations: list[RiskViolation] = []

        # 1. Position size
        if qty > _MAX_CONTRACTS:
            violations.append(RiskViolation("max_contracts", f"Size {qty} exceeds max {_MAX_CONTRACTS}"))

        if qty <= 0:
            violations.append(RiskViolation("invalid_size", "Order size must be positive"))

        # 2. Must have a stop loss
        if stop_loss == 0 or stop_loss == entry_price:
            violations.append(RiskViolation("no_stop_loss", "Order must have a valid stop loss"))

        # 3. Risk per trade check
        if account_balance > 0 and entry_price > 0 and stop_loss > 0:
            risk_dollars = abs(entry_price - stop_loss) * 20.0 * qty  # NQ point value = $20
            risk_pct = risk_dollars / account_balance
            if risk_pct > _MAX_RISK_PER_TRADE_PCT:
                violations.append(RiskViolation(
                    "max_risk",
                    f"Risk {risk_pct:.1%} exceeds max {_MAX_RISK_PER_TRADE_PCT:.1%} (${risk_dollars:.2f})",
                ))

            # Would this breach daily loss limit if stopped out?
            potential_loss = self._daily_pnl - risk_dollars
            if potential_loss < -self.effective_daily_limit:
                violations.append(RiskViolation(
                    "daily_loss_projected",
                    f"Would breach daily limit if stopped: ${potential_loss:.2f}",
                ))

            # Would this breach drawdown?
            potential_equity = self._equity - risk_dollars
            if potential_equity < self._drawdown_floor:
                violations.append(RiskViolation(
                    "drawdown_projected",
                    f"Would breach drawdown floor if stopped: ${potential_equity:.2f} < ${self._drawdown_floor:.2f}",
                ))

        valid = len(violations) == 0
        if not valid:
            logger.warning("risk_guard.order_rejected", violations=[str(v) for v in violations])

        return valid, violations

    def check_emergency_exit(self) -> bool:
        """Check if we need to emergency-close ALL positions.

        Returns True if equity is dangerously close to drawdown floor.
        """
        if self._equity <= self._drawdown_floor + _EMERGENCY_EXIT_BUFFER:
            logger.critical(
                "risk_guard.EMERGENCY_EXIT",
                equity=self._equity,
                floor=self._drawdown_floor,
                buffer=_EMERGENCY_EXIT_BUFFER,
            )
            return True

        # Also check daily loss
        if self._daily_pnl <= -(self.effective_daily_limit - 50):
            logger.critical(
                "risk_guard.EMERGENCY_EXIT_DAILY",
                daily_pnl=self._daily_pnl,
                limit=self.effective_daily_limit,
            )
            return True

        return False

    def must_close_all(self, now: datetime | None = None) -> bool:
        """Check if it's time to close all positions (3:45 PM ET)."""
        if now is None:
            now = datetime.now(ET)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        elif now.tzinfo != ET:
            now = now.astimezone(ET)
        return now.time() >= _CLOSE_ALL_TIME
