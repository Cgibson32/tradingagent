"""Tests for the configuration system."""

from __future__ import annotations

import os
import pytest

from src.config import (
    AppConfig,
    RiskConfig,
    NQConfig,
    load_yaml_config,
    _deep_merge,
)


class TestDeepMerge:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_override_replaces_non_dict(self):
        base = {"a": [1, 2, 3]}
        override = {"a": [4, 5]}
        result = _deep_merge(base, override)
        assert result == {"a": [4, 5]}

    def test_base_not_modified(self):
        base = {"a": 1}
        override = {"a": 2}
        _deep_merge(base, override)
        assert base == {"a": 1}


class TestYamlConfig:
    def test_load_default(self):
        config = load_yaml_config("paper")
        assert "tradovate" in config
        assert "risk" in config
        assert "signals" in config

    def test_paper_override(self):
        config = load_yaml_config("paper")
        assert config["features"]["paper_mode"] is True


class TestRiskConfig:
    def test_daily_loss_percentage(self):
        risk = RiskConfig()
        # 3% of $10K = $300
        assert risk.daily_loss_limit(10000.0) == 300.0

    def test_weekly_loss_percentage(self):
        risk = RiskConfig()
        # 5% of $10K = $500
        assert risk.weekly_loss_limit(10000.0) == 500.0

    def test_max_contracts_default(self):
        risk = RiskConfig()
        assert risk.max_contracts == 4

    def test_account_size_default(self):
        risk = RiskConfig()
        assert risk.account_size == 10000


class TestNQConfig:
    def test_tick_values(self):
        nq = NQConfig()
        assert nq.tick_size == 0.25
        assert nq.tick_value == 5.00
        assert nq.point_value == 20.00
        assert nq.commission_per_contract == 0.82
        assert nq.slippage_ticks == 2


class TestAppConfig:
    def test_validate_missing_secrets(self):
        from pydantic import SecretStr
        from src.config import SecretSettings

        config = AppConfig(
            secrets=SecretSettings(
                tradovate_username="",
                tradovate_password=SecretStr(""),
                tradovate_app_id="",
                tradovate_client_id="",
                tradovate_client_secret=SecretStr(""),
                anthropic_api_key=SecretStr(""),
            )
        )
        missing = config.validate_required_secrets()
        assert "TRADOVATE_USERNAME" in missing

    def test_default_symbols(self):
        config = AppConfig()
        assert config.symbols.primary == "MNQU5"
        assert config.symbols.smt_compare == "MESU5"

    def test_kill_zones(self):
        config = AppConfig()
        assert config.kill_zones.ny_open.start == "09:30"
        assert config.kill_zones.ny_open.end == "11:00"
        assert config.kill_zones.overnight.label == "Overnight/Asia"

    def test_signal_weights_sum(self):
        config = AppConfig()
        weights = config.signals.signal_weights
        total = (
            weights.ict_pattern + weights.htf_bias + weights.indicators
            + weights.order_flow + weights.key_levels + weights.intermarket
        )
        assert abs(total - 1.0) < 0.01
