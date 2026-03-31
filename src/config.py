"""Configuration loader that merges YAML files with environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dict. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_config(mode: str = "paper") -> dict[str, Any]:
    """Load and merge YAML config files: default.yaml + {mode}.yaml + secrets.yaml."""
    config_dir = Path(__file__).parent.parent / "config"

    # Load default
    default_path = config_dir / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")
    with open(default_path) as f:
        config = yaml.safe_load(f) or {}

    # Merge mode-specific overrides
    mode_path = config_dir / f"{mode}.yaml"
    if mode_path.exists():
        with open(mode_path) as f:
            mode_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, mode_config)

    # Merge secrets (gitignored)
    secrets_path = config_dir / "secrets.yaml"
    if secrets_path.exists():
        with open(secrets_path) as f:
            secrets_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, secrets_config)

    return config


# ── Pydantic sub-models for typed config sections ──


class TradovateUrls(BaseModel):
    rest: str
    ws_trading: str
    ws_market_data: str


class TradovateAuthConfig(BaseModel):
    token_renew_minutes: int = 45


class TradovateConfig(BaseModel):
    environment: str = "demo"
    urls: dict[str, TradovateUrls] = Field(default_factory=dict)
    auth: TradovateAuthConfig = Field(default_factory=TradovateAuthConfig)

    @property
    def active_urls(self) -> TradovateUrls:
        return self.urls[self.environment]


class SessionConfig(BaseModel):
    trading_start: str = "09:30"
    trading_end: str = "16:00"
    close_all_time: str = "15:45"
    no_new_entries_after: str = "15:00"
    tighten_stops_time: str = "15:30"
    daily_reset_time: str = "18:00"
    eod_drawdown_update: str = "17:00"


class KillZoneWindow(BaseModel):
    start: str
    end: str
    label: str


class KillZonesConfig(BaseModel):
    london_open: KillZoneWindow = Field(
        default_factory=lambda: KillZoneWindow(start="02:00", end="05:00", label="London Open")
    )
    ny_open: KillZoneWindow = Field(
        default_factory=lambda: KillZoneWindow(start="09:30", end="11:00", label="NY Open")
    )
    ny_lunch: KillZoneWindow = Field(
        default_factory=lambda: KillZoneWindow(start="12:00", end="13:30", label="NY Lunch")
    )
    ny_pm: KillZoneWindow = Field(
        default_factory=lambda: KillZoneWindow(start="13:30", end="15:00", label="NY PM Session")
    )
    confidence_penalty_outside_kz: float = 0.2
    time_decay_start: str = "13:30"
    no_entry_cutoff: str = "14:30"


class RiskConfig(BaseModel):
    risk_per_trade_pct: float = 0.01
    min_risk_reward: float = 2.5
    max_risk_per_trade_pct: float = 0.015
    max_contracts: int = 4
    max_concurrent_positions: int = 2
    max_trades_per_day: int = 6
    max_daily_loss: float = 1750.0
    daily_loss_buffer: float = 250.0
    max_trailing_drawdown: float = 4500.0
    drawdown_safety_buffer: float = 500.0
    drawdown_lock_threshold: float = 100.0
    account_size: int = 150000
    cooldown_after_loss_minutes: int = 15
    news_blackout_minutes: int = 15
    flash_crash_atr_multiple: float = 5.0
    spread_blowout_multiple: float = 4.0
    max_consecutive_losses: int = 3
    volatility_shift_atr_multiple: float = 2.0

    @property
    def effective_daily_loss_limit(self) -> float:
        return self.max_daily_loss - self.daily_loss_buffer

    @property
    def effective_drawdown_limit(self) -> float:
        return self.max_trailing_drawdown - self.drawdown_safety_buffer


class NQConfig(BaseModel):
    tick_size: float = 0.25
    tick_value: float = 5.00
    point_value: float = 20.00
    commission_per_contract: float = 0.82
    slippage_ticks: int = 2


class LLMConfig(BaseModel):
    realtime_model: str = "claude-sonnet-4-20250514"
    analysis_model: str = "claude-opus-4-6"
    realtime_timeout_seconds: int = 10
    analysis_timeout_seconds: int = 60
    daily_budget_usd: float = 5.00
    cache_ttl_seconds: int = 300
    regime_update_interval_minutes: int = 30


class SignalWeights(BaseModel):
    ict_pattern: float = 0.35
    htf_bias: float = 0.20
    indicators: float = 0.15
    order_flow: float = 0.15
    key_levels: float = 0.10
    intermarket: float = 0.05


class SignalConfig(BaseModel):
    ema_periods: list[int] = Field(default_factory=lambda: [9, 21, 50, 200])
    adx_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    atr_period: int = 14
    swing_lookback: int = 5
    fvg_min_ticks: int = 4
    displacement_atr_multiple: float = 2.0
    displacement_lookback: int = 5
    liquidity_pool_tolerance: int = 2
    fvg_max_age_bars: int = 50
    max_tracked_fvgs: int = 20
    min_confidence_threshold: float = 0.65
    signal_weights: SignalWeights = Field(default_factory=SignalWeights)


class TradeConfig(BaseModel):
    partial_tp_r: float = 1.0
    partial_tp_pct: float = 0.50
    full_tp_r: float = 2.5
    trailing_stop_atr_multiple: float = 1.5
    limit_order_timeout_minutes: int = 30
    break_even_trigger_pct: float = 0.40


class StorageConfig(BaseModel):
    db_path: str = "data/trading.db"
    parquet_dir: str = "data/candles"
    historical_months: int = 6
    checkpoint_interval_seconds: int = 60


class NotificationsConfig(BaseModel):
    enabled: bool = False
    discord_webhook_url: str = ""


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    rotate_days: int = 30
    log_dir: str = "logs"


class FeaturesConfig(BaseModel):
    order_flow_enabled: bool = True
    intermarket_enabled: bool = True
    shadow_runner_enabled: bool = False
    llm_enabled: bool = True
    paper_mode: bool = True


class SymbolsConfig(BaseModel):
    primary: str = "NQU5"
    smt_compare: str = "ESU5"
    intermarket: list[str] = Field(default_factory=lambda: ["DXU5", "VXQ5", "ZNU5"])


class TimeframesConfig(BaseModel):
    entry: str = "5m"
    htf_bias: list[str] = Field(default_factory=lambda: ["30m", "1h"])
    context: list[str] = Field(default_factory=lambda: ["15m", "4h", "1d"])


# ── Environment variables for secrets ──


class SecretSettings(BaseSettings):
    """Secrets loaded from environment variables or .env file."""

    tradovate_username: str = ""
    tradovate_password: SecretStr = SecretStr("")
    tradovate_app_id: str = ""
    tradovate_client_id: str = ""
    tradovate_client_secret: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")
    discord_webhook_url: str = ""
    tradovate_environment: str = "demo"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# ── Main AppConfig ──


class AppConfig(BaseModel):
    """Complete application configuration assembled from YAML + env vars."""

    tradovate: TradovateConfig = Field(default_factory=TradovateConfig)
    symbols: SymbolsConfig = Field(default_factory=SymbolsConfig)
    timeframes: TimeframesConfig = Field(default_factory=TimeframesConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    kill_zones: KillZonesConfig = Field(default_factory=KillZonesConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    nq: NQConfig = Field(default_factory=NQConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    signals: SignalConfig = Field(default_factory=SignalConfig)
    trade: TradeConfig = Field(default_factory=TradeConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    secrets: SecretSettings = Field(default_factory=SecretSettings)

    def validate_required_secrets(self) -> list[str]:
        """Return list of missing required secrets."""
        missing = []
        if not self.secrets.tradovate_username:
            missing.append("TRADOVATE_USERNAME")
        if not self.secrets.tradovate_password.get_secret_value():
            missing.append("TRADOVATE_PASSWORD")
        if not self.secrets.tradovate_app_id:
            missing.append("TRADOVATE_APP_ID")
        if not self.secrets.tradovate_client_id:
            missing.append("TRADOVATE_CLIENT_ID")
        if not self.secrets.tradovate_client_secret.get_secret_value():
            missing.append("TRADOVATE_CLIENT_SECRET")
        if self.features.llm_enabled and not self.secrets.anthropic_api_key.get_secret_value():
            missing.append("ANTHROPIC_API_KEY")
        return missing


def load_config(mode: str = "paper") -> AppConfig:
    """Load full application config from YAML files + environment variables.

    Args:
        mode: One of 'backtest', 'paper', 'live'

    Returns:
        Fully validated AppConfig instance.

    Raises:
        FileNotFoundError: If default.yaml is missing.
        ValueError: If required secrets are missing.
    """
    yaml_data = load_yaml_config(mode)
    secrets = SecretSettings()

    # Override tradovate environment from env var if set
    if secrets.tradovate_environment:
        yaml_data.setdefault("tradovate", {})["environment"] = secrets.tradovate_environment

    config = AppConfig(**yaml_data, secrets=secrets)

    # Validate secrets
    missing = config.validate_required_secrets()
    if missing:
        raise ValueError(
            f"Missing required configuration: {', '.join(missing)}. "
            f"Set these as environment variables or in config/secrets.yaml"
        )

    return config
