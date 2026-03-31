"""Position sizing with Tradeify prop firm limits.

Calculates position size based on:
- Fixed fractional risk (1% of account per trade)
- Distance to stop loss
- NQ contract specifications
- Tradeify account limits (max 4 contracts self-imposed)
- Commission and slippage
"""

from __future__ import annotations

import math

import structlog

logger = structlog.get_logger(__name__)


class PositionSizer:
    """Calculates safe position sizes for NQ futures trades.

    size = (balance * risk_pct) / (entry - stop) / point_value

    Constrained by:
    - Max contracts (4, self-imposed)
    - Max risk per trade (1.5% hard ceiling)
    - Daily loss limit remaining
    - Drawdown limit remaining
    """

    def __init__(
        self,
        point_value: float = 20.0,
        tick_size: float = 0.25,
        tick_value: float = 5.0,
        commission_per_contract: float = 0.82,
        slippage_ticks: int = 2,
        max_contracts: int = 4,
        risk_per_trade_pct: float = 0.01,
        max_risk_per_trade_pct: float = 0.015,
    ):
        self._point_value = point_value
        self._tick_size = tick_size
        self._tick_value = tick_value
        self._commission = commission_per_contract
        self._slippage_ticks = slippage_ticks
        self._max_contracts = max_contracts
        self._risk_pct = risk_per_trade_pct
        self._max_risk_pct = max_risk_per_trade_pct

    def calculate(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        daily_loss_remaining: float | None = None,
        drawdown_remaining: float | None = None,
    ) -> int:
        """Calculate position size in contracts.

        Args:
            account_balance: Current account equity
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            daily_loss_remaining: Dollars remaining before daily loss limit
            drawdown_remaining: Dollars remaining before drawdown limit

        Returns:
            Number of contracts (0 if trade should be skipped)
        """
        # Calculate risk per contract in dollars
        stop_distance_points = abs(entry_price - stop_loss_price)
        if stop_distance_points == 0:
            logger.warning("position_sizer.zero_stop_distance")
            return 0

        risk_per_contract = stop_distance_points * self._point_value
        slippage_cost = self._slippage_ticks * self._tick_value
        total_risk_per_contract = risk_per_contract + slippage_cost + self._commission

        # Target risk in dollars
        target_risk = account_balance * self._risk_pct

        # Hard ceiling on risk
        max_risk = account_balance * self._max_risk_pct
        target_risk = min(target_risk, max_risk)

        # Calculate size from risk
        size_from_risk = math.floor(target_risk / total_risk_per_contract)

        # Constrain by daily loss remaining
        size_from_daily = self._max_contracts
        if daily_loss_remaining is not None and daily_loss_remaining > 0:
            size_from_daily = math.floor(daily_loss_remaining / total_risk_per_contract)

        # Constrain by drawdown remaining
        size_from_dd = self._max_contracts
        if drawdown_remaining is not None and drawdown_remaining > 0:
            size_from_dd = math.floor(drawdown_remaining / total_risk_per_contract)

        # Take the minimum of all constraints
        size = min(
            size_from_risk,
            size_from_daily,
            size_from_dd,
            self._max_contracts,
        )

        # Must be at least 1 if we're trading, 0 if impossible
        size = max(size, 0)

        if size == 0:
            logger.warning(
                "position_sizer.zero_size",
                risk_per_contract=total_risk_per_contract,
                target_risk=target_risk,
                daily_remaining=daily_loss_remaining,
                dd_remaining=drawdown_remaining,
            )

        logger.info(
            "position_sizer.calculated",
            size=size,
            risk_per_contract=round(total_risk_per_contract, 2),
            target_risk=round(target_risk, 2),
            stop_distance=round(stop_distance_points, 2),
        )

        return size

    def max_risk_dollars(self, contracts: int, entry_price: float, stop_loss: float) -> float:
        """Calculate total risk in dollars for a given position."""
        stop_distance = abs(entry_price - stop_loss)
        risk_per_contract = stop_distance * self._point_value
        slippage = self._slippage_ticks * self._tick_value
        commission = self._commission
        return contracts * (risk_per_contract + slippage + commission)
