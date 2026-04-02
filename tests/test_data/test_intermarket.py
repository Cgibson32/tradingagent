"""Tests for intermarket tracking."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.data.intermarket import IntermarketTracker, VIX_ELEVATED, VIX_HIGH
from src.data.models import Quote


@pytest.fixture
def tracker():
    return IntermarketTracker()


class TestIntermarketTracker:
    def test_initial_snapshot(self, tracker):
        snap = tracker.snapshot
        assert snap.dxy_price is None
        assert snap.vix_price is None
        assert snap.zn_price is None
        assert snap.vix_level == "normal"

    def test_process_dxy_quote(self, tracker):
        quote = Quote(
            timestamp=datetime.now(timezone.utc),
            symbol="DXU5",
            bid_price=104.0,
            ask_price=104.1,
            last_price=104.05,
        )
        tracker.process_quote("DXU5", quote)
        assert tracker.snapshot.dxy_price == 104.05

    def test_process_vix_quote(self, tracker):
        quote = Quote(
            timestamp=datetime.now(timezone.utc),
            symbol="VXQ5",
            bid_price=26.0,
            ask_price=26.5,
            last_price=26.25,
        )
        tracker.process_quote("VXQ5", quote)
        assert tracker.snapshot.vix_price == 26.25
        assert tracker.snapshot.vix_level == "elevated"

    def test_vix_high(self, tracker):
        quote = Quote(
            timestamp=datetime.now(timezone.utc),
            symbol="VXQ5",
            bid_price=35.0, ask_price=35.5, last_price=35.25,
        )
        tracker.process_quote("VXQ5", quote)
        assert tracker.snapshot.vix_level == "high"

    def test_vix_low(self, tracker):
        quote = Quote(
            timestamp=datetime.now(timezone.utc),
            symbol="VXQ5",
            bid_price=12.0, ask_price=12.5, last_price=12.25,
        )
        tracker.process_quote("VXQ5", quote)
        assert tracker.snapshot.vix_level == "low"

    def test_should_reduce_size_elevated_vix(self, tracker):
        quote = Quote(
            timestamp=datetime.now(timezone.utc),
            symbol="VXQ5",
            bid_price=28.0, ask_price=28.5, last_price=28.25,
        )
        tracker.process_quote("VXQ5", quote)
        assert tracker.should_reduce_size() is True

    def test_should_not_reduce_size_normal_vix(self, tracker):
        quote = Quote(
            timestamp=datetime.now(timezone.utc),
            symbol="VXQ5",
            bid_price=18.0, ask_price=18.5, last_price=18.25,
        )
        tracker.process_quote("VXQ5", quote)
        assert tracker.should_reduce_size() is False

    def test_dxy_change_pct(self, tracker):
        tracker.set_previous_closes(dxy_close=103.0)
        quote = Quote(
            timestamp=datetime.now(timezone.utc),
            symbol="DXU5",
            bid_price=104.0, ask_price=104.1, last_price=104.03,
        )
        tracker.process_quote("DXU5", quote)
        snap = tracker.snapshot
        assert snap.dxy_change_pct is not None
        assert snap.dxy_change_pct == pytest.approx(1.0, abs=0.1)  # ~1% gain

    def test_nq_bias_bearish_on_strong_dollar(self, tracker):
        tracker.set_previous_closes(dxy_close=100.0)
        # Dollar up 1%
        tracker.process_quote("DXU5", Quote(
            timestamp=datetime.now(timezone.utc),
            symbol="DXU5",
            bid_price=101.0, ask_price=101.1, last_price=101.0,
        ))
        assert tracker.get_nq_bias() == "bearish"

    def test_nq_bias_bullish_on_weak_dollar(self, tracker):
        tracker.set_previous_closes(dxy_close=100.0)
        # Dollar down 1%
        tracker.process_quote("DXU5", Quote(
            timestamp=datetime.now(timezone.utc),
            symbol="DXU5",
            bid_price=99.0, ask_price=99.1, last_price=99.0,
        ))
        assert tracker.get_nq_bias() == "bullish"

    def test_nq_bias_neutral_no_data(self, tracker):
        assert tracker.get_nq_bias() == "neutral"
