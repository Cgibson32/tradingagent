"""Tests for performance analytics."""

from __future__ import annotations

import pytest

from src.feedback.analytics import PerformanceAnalytics


@pytest.fixture
def analytics():
    return PerformanceAnalytics()


@pytest.fixture
def sample_trades():
    return [
        {"pnl_r": 2.5, "pnl_dollars": 50, "fees": 1.24, "status": "win",
         "entry_time": "2025-06-15T10:00:00", "exit_time": "2025-06-15T10:45:00",
         "regime": "trending_bullish", "strategy_name": "ict"},
        {"pnl_r": -1.0, "pnl_dollars": -20, "fees": 1.24, "status": "loss",
         "entry_time": "2025-06-15T11:00:00", "exit_time": "2025-06-15T11:30:00",
         "regime": "trending_bullish", "strategy_name": "ict"},
        {"pnl_r": 1.8, "pnl_dollars": 36, "fees": 1.24, "status": "win",
         "entry_time": "2025-06-16T10:30:00", "exit_time": "2025-06-16T11:15:00",
         "regime": "ranging", "strategy_name": "ict"},
        {"pnl_r": -1.0, "pnl_dollars": -20, "fees": 1.24, "status": "loss",
         "entry_time": "2025-06-16T14:00:00", "exit_time": "2025-06-16T14:20:00",
         "regime": "ranging", "strategy_name": "ict"},
        {"pnl_r": 2.0, "pnl_dollars": 40, "fees": 1.24, "status": "win",
         "entry_time": "2025-06-17T09:45:00", "exit_time": "2025-06-17T10:30:00",
         "regime": "trending_bullish", "strategy_name": "ict"},
    ]


class TestPerformanceAnalytics:
    def test_calculate_metrics(self, analytics, sample_trades):
        metrics = analytics.calculate_metrics(sample_trades)
        assert metrics["total_trades"] == 5
        assert metrics["wins"] == 3
        assert metrics["losses"] == 2
        assert metrics["win_rate"] == 60.0

    def test_profit_factor(self, analytics, sample_trades):
        metrics = analytics.calculate_metrics(sample_trades)
        # Gross profit: 50 + 36 + 40 = 126
        # Gross loss: 20 + 20 = 40
        assert metrics["profit_factor"] == pytest.approx(3.15, abs=0.1)

    def test_empty_trades(self, analytics):
        metrics = analytics.calculate_metrics([])
        assert metrics["total_trades"] == 0

    def test_all_wins(self, analytics):
        trades = [
            {"pnl_r": 2.0, "pnl_dollars": 40, "fees": 1.0, "status": "win"},
            {"pnl_r": 1.5, "pnl_dollars": 30, "fees": 1.0, "status": "win"},
        ]
        metrics = analytics.calculate_metrics(trades)
        assert metrics["win_rate"] == 100.0

    def test_streaks(self, analytics, sample_trades):
        metrics = analytics.calculate_metrics(sample_trades)
        assert metrics["max_win_streak"] >= 1
        assert metrics["max_loss_streak"] >= 1

    def test_segment_by_regime(self, analytics, sample_trades):
        segments = analytics.segment_by(sample_trades, "regime")
        assert "trending_bullish" in segments
        assert "ranging" in segments
        assert segments["trending_bullish"]["total_trades"] == 3

    def test_segment_by_strategy(self, analytics, sample_trades):
        segments = analytics.segment_by(sample_trades, "strategy_name")
        assert "ict" in segments
        assert segments["ict"]["total_trades"] == 5

    def test_format_summary(self, analytics, sample_trades):
        metrics = analytics.calculate_metrics(sample_trades)
        summary = analytics.format_summary(metrics)
        assert "Win Rate: 60.0%" in summary
        assert "Profit Factor: 3.15" in summary

    def test_rolling_metrics(self, analytics, sample_trades):
        rolling = analytics.rolling_metrics(sample_trades, window=3)
        assert len(rolling) == 3  # 5 trades, window 3 → snapshots at indices 3,4,5
        assert all("total_trades" in s for s in rolling)

    def test_duration_calculation(self, analytics, sample_trades):
        metrics = analytics.calculate_metrics(sample_trades)
        assert metrics["avg_duration_minutes"] > 0

    def test_net_pnl_includes_fees(self, analytics, sample_trades):
        metrics = analytics.calculate_metrics(sample_trades)
        assert metrics["net_pnl"] < metrics["total_pnl"]  # Fees reduce net
