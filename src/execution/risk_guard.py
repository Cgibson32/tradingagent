"""Risk guardrails for personal Tradovate account.

This is the HARD safety layer. It can BLOCK orders but never PLACE them.
Values are configurable at init time but enforced immutably at runtime.

The LLM cannot override these rules. They are the final gatekeeper
before any order reaches the broker.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import structlog

logger = structlog.get_logger(__name__)

ET = ZoneInfo("America/New_York")

# CME Globex hours
_DAILY_HALT_START = time(16, 0)   # 4:00 PM ET
_DAILY_HALT_END = time(17, 0)     # 5:00 PM ET


class RiskViolation:
    """Describes a risk rule violation."""

    def __init__(self, rule: str, message: str, severity: str = "block"):
        self.rule = rule
        self.message = message
        self.severity = severity  # "block" or "warn"

    def __repr__(self):
        return f"RiskViolation({self.rule}: {self.message})"


class RiskGuard:
    """Enforces risk rules for a personal trading account.

    All limits are configurable at init time. Once constructed,
    the rules cannot be changed at runtime.
    """

    def __init__(
        self,
        max_risk_per_trade_pct: float = 0.02,
        max_daily_loss_pct: float = 0.03,
        max_weekly_loss_pct: float = 0.05,
        max_contracts: int = 4,
        max_concurrent_positions: int = 2,
        max_trades_per_day: int = 6,
        cooldown_after_loss_minutes: int = 15,
        news_blackout_minutes: int = 15,
        max_consecutive_losses: int = 3,
        account_size: float = 10000.0,
    ):
        # Config (set once at init, not changed)
        self._max_risk_pct = max_risk_per_trade_pct
        self._max_daily_loss_pct = max_daily_loss_pct
        self._max_weekly_loss_pct = max_weekly_loss_pct
        self._max_contracts = max_contracts
        self._max_concurrent = max_concurrent_positions
        self._max_trades_day = max_trades_per_day
        self._cooldown_minutes = cooldown_after_loss_minutes
        self._news_blackout = news_blackout_minutes
        self._max_consec_losses = max_consecutive_losses

        # Tracked state (updated by the orchestrator)
        self._equity: float = account_size
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._trades_today: int = 0
        self._consecutive_losses: int = 0
        self._last_loss_time: datetime | None = None
        self._open_position_count: int = 0
        self._upcoming_news_times: list[datetime] = []

    # ── Properties ──

    @property
    def max_contracts(self) -> int:
        return self._max_contracts

    @property
    def daily_loss_limit(self) -> float:
        """Dollar amount of daily loss limit based on current equity."""
        return self._equity * self._max_daily_loss_pct

    @property
    def weekly_loss_limit(self) -> float:
        """Dollar amount of weekly loss limit based on current equity."""
        return self._equity * self._max_weekly_loss_pct

    @property
    def equity(self) -> float:
        return self._equity

    # ── State updates (called by orchestrator) ──

    def update_equity(self, equity: float) -> None:
        self._equity = equity

    def update_pnl(self, daily_pnl: float, weekly_pnl: float | None = None) -> None:
        self._daily_pnl = daily_pnl
        if weekly_pnl is not None:
            self._weekly_pnl = weekly_pnl

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
        """Reset daily counters. Called at daily maintenance halt."""
        self._daily_pnl = 0.0
        self._trades_today = 0
        self._consecutive_losses = 0
        self._last_loss_time = None

    def reset_weekly(self) -> None:
        """Reset weekly P&L. Called on Sunday open."""
        self._weekly_pnl = 0.0

    # ── Core checks ──

    def can_trade(self, now: datetime | None = None) -> tuple[bool, list[RiskViolation]]:
        """Check if trading is currently allowed."""
        if now is None:
            now = datetime.now(ET)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        elif now.tzinfo != ET:
            now = now.astimezone(ET)

        violations: list[RiskViolation] = []

        # 1. CME daily maintenance halt (4:00-5:00 PM ET)
        now_time = now.time()
        if _DAILY_HALT_START <= now_time < _DAILY_HALT_END:
            violations.append(RiskViolation("daily_halt", "CME daily maintenance halt (4:00-5:00 PM ET)"))

        # 2. Daily loss limit (percentage-based)
        if self._daily_pnl <= -self.daily_loss_limit:
            violations.append(RiskViolation(
                "daily_loss",
                f"Daily loss limit reached: ${self._daily_pnl:.2f} (limit: -${self.daily_loss_limit:.2f}, {self._max_daily_loss_pct:.0%} of equity)",
            ))

        # 3. Weekly loss limit
        if self._weekly_pnl <= -self.weekly_loss_limit:
            violations.append(RiskViolation(
                "weekly_loss",
                f"Weekly loss limit reached: ${self._weekly_pnl:.2f} (limit: -${self.weekly_loss_limit:.2f})",
            ))

        # 4. Max trades per day
        if self._trades_today >= self._max_trades_day:
            violations.append(RiskViolation("max_trades", f"Max trades per day reached: {self._trades_today}"))

        # 5. Max concurrent positions
        if self._open_position_count >= self._max_concurrent:
            violations.append(RiskViolation("max_positions", f"Max concurrent positions: {self._open_position_count}"))

        # 6. Cooldown after loss
        if self._last_loss_time:
            minutes_since = (now - self._last_loss_time).total_seconds() / 60
            if minutes_since < self._cooldown_minutes:
                remaining = self._cooldown_minutes - minutes_since
                violations.append(RiskViolation("cooldown", f"Cooldown: {remaining:.0f} min remaining after loss"))

        # 7. Consecutive losses
        if self._consecutive_losses >= self._max_consec_losses:
            violations.append(RiskViolation("consecutive_losses", f"{self._consecutive_losses} consecutive losses — pausing"))

        # 8. News blackout
        for news_time in self._upcoming_news_times:
            if news_time.tzinfo is None:
                news_time = news_time.replace(tzinfo=ET)
            minutes_to_news = (news_time - now).total_seconds() / 60
            if abs(minutes_to_news) < self._news_blackout:
                violations.append(RiskViolation("news_blackout", f"High-impact news within {abs(minutes_to_news):.0f} minutes"))
                break

        allowed = len(violations) == 0
        if not allowed:
            logger.warning("risk_guard.blocked", violations=[str(v) for v in violations])

        return allowed, violations

    def validate_order(
        self, qty: int, entry_price: float, stop_loss: float,
        point_value: float = 2.0, margin_required: float = 0.0,
    ) -> tuple[bool, list[RiskViolation]]:
        """Validate a specific order against all rules.

        Args:
            qty: Number of contracts
            entry_price: Planned entry price
            stop_loss: Stop loss price
            point_value: Dollar value per point (MNQ=2, NQ=20)
            margin_required: Total margin needed for this position
        """
        violations: list[RiskViolation] = []

        # 1. Position size
        if qty > self._max_contracts:
            violations.append(RiskViolation("max_contracts", f"Size {qty} exceeds max {self._max_contracts}"))

        if qty <= 0:
            violations.append(RiskViolation("invalid_size", "Order size must be positive"))

        # 2. Must have a stop loss
        if stop_loss == 0 or stop_loss == entry_price:
            violations.append(RiskViolation("no_stop_loss", "Order must have a valid stop loss"))

        # 3. Risk per trade check
        if self._equity > 0 and entry_price > 0 and stop_loss > 0:
            risk_dollars = abs(entry_price - stop_loss) * point_value * qty
            risk_pct = risk_dollars / self._equity
            if risk_pct > self._max_risk_pct:
                violations.append(RiskViolation(
                    "max_risk",
                    f"Risk {risk_pct:.1%} exceeds max {self._max_risk_pct:.1%} (${risk_dollars:.2f})",
                ))

            # Would this breach daily loss limit if stopped out?
            potential_daily = self._daily_pnl - risk_dollars
            if potential_daily < -self.daily_loss_limit:
                violations.append(RiskViolation(
                    "daily_loss_projected",
                    f"Would breach daily limit if stopped: ${potential_daily:.2f}",
                ))

        # 4. Margin check
        if margin_required > 0 and margin_required > self._equity * 0.8:
            violations.append(RiskViolation(
                "insufficient_margin",
                f"Margin ${margin_required:.2f} exceeds 80% of equity ${self._equity:.2f}",
            ))

        valid = len(violations) == 0
        if not valid:
            logger.warning("risk_guard.order_rejected", violations=[str(v) for v in violations])

        return valid, violations

    def check_emergency_exit(self) -> bool:
        """Check if we need to emergency-close ALL positions.

        Returns True if losses are dangerously close to limits.
        """
        # Within 90% of daily loss limit
        if self._daily_pnl <= -(self.daily_loss_limit * 0.9):
            logger.critical(
                "risk_guard.EMERGENCY_EXIT",
                daily_pnl=self._daily_pnl,
                limit=self.daily_loss_limit,
            )
            return True

        # Within 90% of weekly loss limit
        if self._weekly_pnl <= -(self.weekly_loss_limit * 0.9):
            logger.critical(
                "risk_guard.EMERGENCY_EXIT_WEEKLY",
                weekly_pnl=self._weekly_pnl,
                limit=self.weekly_loss_limit,
            )
            return True

        return False

    def is_in_maintenance_halt(self, now: datetime | None = None) -> bool:
        """Check if we're in the CME daily maintenance halt."""
        if now is None:
            now = datetime.now(ET)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=ET)
        elif now.tzinfo != ET:
            now = now.astimezone(ET)
        return _DAILY_HALT_START <= now.time() < _DAILY_HALT_END
