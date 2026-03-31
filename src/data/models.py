"""Data models for the trading agent.

All data structures that cross module boundaries are defined here as Pydantic models.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


class OrderAction(str, Enum):
    BUY = "Buy"
    SELL = "Sell"


class OrderType(str, Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_LIMIT = "StopLimit"


class TimeInForce(str, Enum):
    DAY = "Day"
    GTC = "GTC"
    IOC = "IOC"


class OrderStatus(str, Enum):
    PENDING = "pending"
    WORKING = "working"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TradeStatus(str, Enum):
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    OPEN = "open"


class MarketRegime(str, Enum):
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING = "ranging"
    VOLATILE_CHOPPY = "volatile_choppy"


# ── Market Data Models ─────────────────────────────────


class Candle(BaseModel):
    """OHLCV candle data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    symbol: str = ""
    timeframe: str = ""

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2


class Quote(BaseModel):
    """Real-time quote data from Tradovate WebSocket."""

    timestamp: datetime
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int = 0
    ask_size: int = 0
    last_price: float = 0.0
    last_size: int = 0
    total_volume: int = 0

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2


class DOMLevel(BaseModel):
    """Single price level in the depth of market."""

    price: float
    bid_size: int = 0
    ask_size: int = 0

    @property
    def imbalance_ratio(self) -> float:
        total = self.bid_size + self.ask_size
        if total == 0:
            return 0.0
        return (self.bid_size - self.ask_size) / total


class DOMSnapshot(BaseModel):
    """Depth of market snapshot."""

    timestamp: datetime
    symbol: str
    levels: list[DOMLevel] = Field(default_factory=list)

    @property
    def total_bid_volume(self) -> int:
        return sum(level.bid_size for level in self.levels)

    @property
    def total_ask_volume(self) -> int:
        return sum(level.ask_size for level in self.levels)


# ── Order Models (Tradovate API) ───────────────────────


class PlaceOrderRequest(BaseModel):
    """Request to place an order on Tradovate."""

    account_spec: str = Field(alias="accountSpec", default="")
    account_id: int = Field(alias="accountId", default=0)
    action: OrderAction
    symbol: str
    order_qty: int = Field(alias="orderQty")
    order_type: OrderType = Field(alias="orderType")
    price: float | None = None
    stop_price: float | None = Field(None, alias="stopPrice")
    time_in_force: TimeInForce = Field(TimeInForce.DAY, alias="timeInForce")
    is_automated: bool = Field(True, alias="isAutomated")

    model_config = {"populate_by_name": True}


class OrderResult(BaseModel):
    """Result from Tradovate order placement."""

    order_id: int = Field(alias="orderId", default=0)
    account_id: int = Field(alias="accountId", default=0)
    action: str = ""
    symbol: str = ""
    order_qty: int = Field(alias="orderQty", default=0)
    order_type: str = Field(alias="orderType", default="")
    price: float | None = None
    stop_price: float | None = Field(None, alias="stopPrice")
    status: str = ""
    fill_price: float | None = Field(None, alias="fillPrice")
    filled_qty: int = Field(alias="filledQty", default=0)
    timestamp: datetime | None = None

    model_config = {"populate_by_name": True}


# ── Position Models ────────────────────────────────────


class Position(BaseModel):
    """Current open position from Tradovate."""

    account_id: int = Field(alias="accountId", default=0)
    contract_id: int = Field(alias="contractId", default=0)
    symbol: str = ""
    net_pos: int = Field(alias="netPos", default=0)
    net_price: float = Field(alias="netPrice", default=0.0)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: datetime | None = None

    model_config = {"populate_by_name": True}

    @property
    def direction(self) -> Direction | None:
        if self.net_pos > 0:
            return Direction.LONG
        elif self.net_pos < 0:
            return Direction.SHORT
        return None

    @property
    def size(self) -> int:
        return abs(self.net_pos)

    @property
    def is_flat(self) -> bool:
        return self.net_pos == 0


# ── Account Models ─────────────────────────────────────


class AccountBalance(BaseModel):
    """Account balance and equity information."""

    account_id: int = 0
    cash_balance: float = 0.0
    open_trade_equity: float = 0.0
    total_equity: float = 0.0
    realized_pnl: float = 0.0
    margin_used: float = 0.0
    timestamp: datetime | None = None


# ── Signal Models ──────────────────────────────────────


class ICTSignal(BaseModel):
    """Signal from ICT/SMC pattern detection."""

    pattern: str  # "liquidity_sweep", "fvg", "bos", "choch", "order_block", "displacement", "smt"
    direction: Direction
    price_level: float
    confidence: float = Field(ge=0.0, le=1.0)
    timeframe: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class TradeSignal(BaseModel):
    """Aggregated trade signal ready for evaluation."""

    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    time_adjusted_tp: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    minutes_until_close: float = 0.0
    estimated_duration_minutes: float = 0.0
    strategy_name: str = ""
    contributing_signals: list[ICTSignal] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def risk_points(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_points(self) -> float:
        tp = self.time_adjusted_tp if self.time_adjusted_tp else self.take_profit
        return abs(tp - self.entry_price)

    @property
    def risk_reward_ratio(self) -> float:
        if self.risk_points == 0:
            return 0.0
        return self.reward_points / self.risk_points


# ── Key Levels ─────────────────────────────────────────


class KeyLevels(BaseModel):
    """Critical ICT reference levels."""

    # Previous day
    prev_day_high: float = 0.0
    prev_day_low: float = 0.0
    prev_day_close: float = 0.0
    prev_day_midpoint: float = 0.0

    # Previous week
    prev_week_high: float = 0.0
    prev_week_low: float = 0.0
    prev_week_open: float = 0.0

    # Previous month
    prev_month_high: float = 0.0
    prev_month_low: float = 0.0
    prev_month_open: float = 0.0

    # Quarterly open
    quarterly_open: float = 0.0

    # Timestamp of last update
    last_updated: datetime | None = None


# ── Order Flow Models ──────────────────────────────────


class OrderFlowState(BaseModel):
    """Current state of order flow analysis."""

    cumulative_delta: float = 0.0
    delta_trend: str = "neutral"  # "rising", "falling", "neutral"
    absorption_detected: bool = False
    absorption_direction: Direction | None = None
    stacked_imbalances: list[float] = Field(default_factory=list)  # Price levels
    imbalance_direction: Direction | None = None
    delta_divergence: bool = False
    timestamp: datetime | None = None


# ── Intermarket Context ────────────────────────────────


class IntermarketSnapshot(BaseModel):
    """Snapshot of correlated instruments for context."""

    dxy_price: float | None = None
    dxy_change_pct: float | None = None
    vix_price: float | None = None
    vix_level: str = "normal"  # "low", "normal", "elevated", "high"
    zn_price: float | None = None
    zn_change_pct: float | None = None
    timestamp: datetime | None = None


# ── Market State (combined snapshot for AI) ────────────


class MarketState(BaseModel):
    """Complete market state snapshot fed to the AI decision engine."""

    symbol: str
    current_price: float
    quote: Quote | None = None
    regime: MarketRegime = MarketRegime.RANGING
    regime_confidence: float = 0.0
    key_levels: KeyLevels = Field(default_factory=KeyLevels)
    order_flow: OrderFlowState = Field(default_factory=OrderFlowState)
    intermarket: IntermarketSnapshot = Field(default_factory=IntermarketSnapshot)
    active_kill_zone: str | None = None
    minutes_until_close: float = 0.0
    daily_pnl: float = 0.0
    equity: float = 0.0
    drawdown_floor: float = 0.0
    trades_today: int = 0
    consecutive_losses: int = 0
    timestamp: datetime | None = None
