"""Position sizing with MNQ/NQ auto-scaling and margin checking.

Calculates position size based on:
- Fixed fractional risk (1% of account per trade)
- Distance to stop loss
- Contract specifications (MNQ or NQ)
- Margin requirements (day and overnight)
- Auto-scaling: MNQ below $25K equity, NQ at $25K+
"""

from __future__ import annotations

import math

import structlog

logger = structlog.get_logger(__name__)


class PositionSizer:
    """Calculates safe position sizes for MNQ/NQ futures.

    Auto-selects instrument based on equity thresholds:
    - Below $25K → MNQ micros (max 4 contracts)
    - $25K-$50K → MNQ or NQ (1 NQ contract)
    - $50K+ → NQ minis (max 2 contracts)
    """

    def __init__(
        self,
        # MNQ specs
        mnq_point_value: float = 2.0,
        mnq_tick_value: float = 0.50,
        mnq_commission: float = 0.62,
        mnq_day_margin: float = 1700.0,
        mnq_overnight_margin: float = 2100.0,
        # NQ specs
        nq_point_value: float = 20.0,
        nq_tick_value: float = 5.0,
        nq_commission: float = 0.82,
        nq_day_margin: float = 17000.0,
        nq_overnight_margin: float = 21000.0,
        # Scaling thresholds
        mnq_to_nq_threshold: float = 25000.0,
        nq_primary_threshold: float = 50000.0,
        max_mnq_contracts: int = 4,
        max_nq_contracts: int = 2,
        # Risk
        risk_per_trade_pct: float = 0.01,
        max_risk_per_trade_pct: float = 0.02,
        slippage_ticks: int = 2,
    ):
        self._mnq_pv = mnq_point_value
        self._mnq_tv = mnq_tick_value
        self._mnq_comm = mnq_commission
        self._mnq_day_margin = mnq_day_margin
        self._mnq_overnight_margin = mnq_overnight_margin

        self._nq_pv = nq_point_value
        self._nq_tv = nq_tick_value
        self._nq_comm = nq_commission
        self._nq_day_margin = nq_day_margin
        self._nq_overnight_margin = nq_overnight_margin

        self._mnq_threshold = mnq_to_nq_threshold
        self._nq_threshold = nq_primary_threshold
        self._max_mnq = max_mnq_contracts
        self._max_nq = max_nq_contracts

        self._risk_pct = risk_per_trade_pct
        self._max_risk_pct = max_risk_per_trade_pct
        self._slippage_ticks = slippage_ticks

    def get_instrument(self, equity: float) -> str:
        """Determine which instrument to trade based on equity.

        Returns 'mnq' or 'nq'.
        """
        if equity >= self._nq_threshold:
            return "nq"
        return "mnq"

    def get_contract_specs(self, instrument: str) -> dict:
        """Get contract specifications for an instrument."""
        if instrument == "nq":
            return {
                "point_value": self._nq_pv,
                "tick_value": self._nq_tv,
                "commission": self._nq_comm,
                "day_margin": self._nq_day_margin,
                "overnight_margin": self._nq_overnight_margin,
                "max_contracts": self._max_nq,
            }
        return {
            "point_value": self._mnq_pv,
            "tick_value": self._mnq_tv,
            "commission": self._mnq_comm,
            "day_margin": self._mnq_day_margin,
            "overnight_margin": self._mnq_overnight_margin,
            "max_contracts": self._max_mnq,
        }

    def calculate(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss_price: float,
        instrument: str | None = None,
        check_overnight_margin: bool = True,
    ) -> tuple[int, str]:
        """Calculate position size in contracts.

        Args:
            account_equity: Current account equity
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            instrument: 'mnq' or 'nq' (auto-selected if None)
            check_overnight_margin: If True, ensure margin for overnight hold

        Returns:
            Tuple of (contracts, instrument) — 0 contracts if trade should be skipped
        """
        if instrument is None:
            instrument = self.get_instrument(account_equity)

        specs = self.get_contract_specs(instrument)
        point_value = specs["point_value"]
        tick_value = specs["tick_value"]
        commission = specs["commission"]
        margin = specs["overnight_margin"] if check_overnight_margin else specs["day_margin"]
        max_contracts = specs["max_contracts"]

        # Calculate risk per contract in dollars
        stop_distance_points = abs(entry_price - stop_loss_price)
        if stop_distance_points == 0:
            logger.warning("position_sizer.zero_stop_distance")
            return 0, instrument

        risk_per_contract = stop_distance_points * point_value
        slippage_cost = self._slippage_ticks * tick_value
        total_risk_per_contract = risk_per_contract + slippage_cost + commission

        # Target risk in dollars (percentage of equity)
        target_risk = account_equity * self._risk_pct
        max_risk = account_equity * self._max_risk_pct
        target_risk = min(target_risk, max_risk)

        # Size from risk
        size_from_risk = math.floor(target_risk / total_risk_per_contract) if total_risk_per_contract > 0 else 0

        # Size from margin
        available_for_margin = account_equity * 0.8  # Use max 80% of equity for margin
        size_from_margin = math.floor(available_for_margin / margin) if margin > 0 else max_contracts

        # Take the minimum of all constraints
        size = min(size_from_risk, size_from_margin, max_contracts)
        size = max(size, 0)

        if size == 0:
            logger.warning(
                "position_sizer.zero_size",
                instrument=instrument,
                risk_per_contract=round(total_risk_per_contract, 2),
                target_risk=round(target_risk, 2),
                margin_per_contract=margin,
                equity=round(account_equity, 2),
            )

        logger.info(
            "position_sizer.calculated",
            size=size,
            instrument=instrument,
            risk_per_contract=round(total_risk_per_contract, 2),
            target_risk=round(target_risk, 2),
            stop_distance=round(stop_distance_points, 2),
            margin_per_contract=margin,
        )

        return size, instrument

    def max_risk_dollars(
        self, contracts: int, entry_price: float, stop_loss: float,
        instrument: str = "mnq",
    ) -> float:
        """Calculate total risk in dollars for a given position."""
        specs = self.get_contract_specs(instrument)
        stop_distance = abs(entry_price - stop_loss)
        risk_per = stop_distance * specs["point_value"]
        slippage = self._slippage_ticks * specs["tick_value"]
        commission = specs["commission"]
        return contracts * (risk_per + slippage + commission)

    def margin_required(self, contracts: int, instrument: str = "mnq", overnight: bool = True) -> float:
        """Calculate total margin required for a position."""
        specs = self.get_contract_specs(instrument)
        margin = specs["overnight_margin"] if overnight else specs["day_margin"]
        return contracts * margin
