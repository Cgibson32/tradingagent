"""Streamlit dashboard for the MNQ/NQ trading agent.

Run: streamlit run src/dashboard/app.py

Pages:
- Live View: positions, P&L, regime, signals, account status
- Trade History: filterable table, equity curve, drawdown chart
- Analytics: win rate over time, by regime, R-distribution
- LLM Log: AI decisions, trade journals, weekly reviews
- Settings: current config, system status
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import streamlit as st
except ImportError:
    print("Install streamlit: pip install streamlit")
    sys.exit(1)

import aiosqlite

DB_PATH = "data/trading.db"

# ── Professional CSS (taste-skill inspired minimalism) ──
CUSTOM_CSS = """<style>
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
[data-testid="stAppViewContainer"] > .main { animation: fadeInUp 0.4s ease-out; }
h1 {
    font-weight: 700 !important; letter-spacing: -0.02em !important;
    color: #111111 !important; font-size: 1.75rem !important;
    padding-bottom: 0.5rem !important; border-bottom: 1px solid #EAEAEA;
    margin-bottom: 1.5rem !important;
}
h2, [data-testid="stSubheader"] {
    font-weight: 600 !important; color: #2F3437 !important;
    font-size: 1.15rem !important; margin-top: 1.5rem !important;
}
[data-testid="stSidebar"] {
    background-color: #FAFAF9 !important; border-right: 1px solid #EAEAEA !important;
}
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #EAEAEA; border-radius: 8px;
    padding: 1rem 1.25rem; transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: #D0D0D0; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="stMetric"] label {
    font-size: 0.75rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.05em !important;
    color: #787774 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.5rem !important; font-weight: 700 !important; color: #111111 !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #EAEAEA; border-radius: 8px; overflow: hidden;
}
[data-testid="stExpander"] {
    border: 1px solid #EAEAEA !important; border-radius: 8px !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important; color: #2F3437 !important;
    transition: background-color 0.15s ease;
}
[data-testid="stExpander"] summary:hover { background-color: #FAFAF9 !important; }
.stButton > button {
    border-radius: 6px !important; font-weight: 600 !important;
    font-size: 0.85rem !important; padding: 0.5rem 1.25rem !important;
    transition: all 0.15s ease !important; box-shadow: none !important;
    border: 1px solid #EAEAEA !important;
}
.stButton > button[data-testid="stBaseButton-primary"] {
    background-color: #111111 !important; color: #FFFFFF !important;
    border: 1px solid #111111 !important;
}
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background-color: #2F3437 !important;
}
hr { border: none !important; border-top: 1px solid #EAEAEA !important; margin: 1.5rem 0 !important; }
[data-testid="stJson"] {
    background: #FAFAF9 !important; border: 1px solid #EAEAEA !important; border-radius: 6px !important;
}
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {
    border: 1px solid #EAEAEA; border-radius: 8px; padding: 0.5rem;
}
.block-container { padding-top: 2rem !important; max-width: 1100px !important; }
.metric-card {
    background: #FFFFFF; border: 1px solid #EAEAEA; border-radius: 8px;
    padding: 1.25rem 1.5rem; transition: border-color 0.2s ease, box-shadow 0.2s ease;
    animation: fadeInUp 0.5s ease-out both;
}
.metric-card:hover { border-color: #D0D0D0; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
.metric-card .metric-label {
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: #787774; margin-bottom: 0.35rem;
}
.metric-card .metric-value { font-size: 1.6rem; font-weight: 700; color: #111111; line-height: 1.2; }
.metric-card .metric-delta { font-size: 0.8rem; font-weight: 500; margin-top: 0.25rem; }
.metric-card .metric-delta.positive { color: #2D7A4F; }
.metric-card .metric-delta.negative { color: #C4473A; }
.metric-card .metric-delta.neutral  { color: #787774; }
.status-badge {
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 4px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em; margin-right: 0.5rem;
}
.status-badge.win       { background: #E8F5E9; color: #2D7A4F; }
.status-badge.loss      { background: #FFEBEE; color: #C4473A; }
.status-badge.breakeven { background: #FFF8E1; color: #9A8C3B; }
.status-badge.open      { background: #E3F2FD; color: #1565C0; }
</style>"""


def mc(label: str, value: str, delta: str = "", delta_class: str = "neutral") -> str:
    """Return HTML for a premium metric card."""
    d = f'<div class="metric-delta {delta_class}">{delta}</div>' if delta else ""
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div>{d}</div>'


def run_async(coro):
    """Run async function in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


async def query_db(sql: str, params: tuple = ()) -> list[dict]:
    """Query the trading database."""
    if not Path(DB_PATH).exists():
        return []
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


# ── Page Config ──
st.set_page_config(page_title="AI Trading Agent", page_icon=":chart_with_upwards_trend:", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Sidebar Navigation ──
page = st.sidebar.radio("Navigation", ["Live View", "Trade History", "Analytics", "LLM Log", "Settings"])


# ── Live View ──
if page == "Live View":
    st.title("Live View")

    col1, col2, col3, col4 = st.columns(4)

    daily_stats = run_async(query_db(
        "SELECT * FROM daily_stats ORDER BY date DESC LIMIT 1"
    ))

    if daily_stats:
        latest = daily_stats[0]
        equity = latest.get("ending_equity", 10000)
        daily_pnl = latest.get("total_pnl", 0)
        trades_today = latest.get("trades_taken", 0)
        win_rate = latest.get("win_rate", 0)
        dc = "positive" if daily_pnl >= 0 else "negative"

        with col1:
            st.markdown(mc("Equity", f"${equity:,.2f}"), unsafe_allow_html=True)
        with col2:
            st.markdown(mc("Daily P&L", f"${daily_pnl:+,.2f}", f"{daily_pnl:+.2f}", dc), unsafe_allow_html=True)
        with col3:
            st.markdown(mc("Trades Today", str(trades_today)), unsafe_allow_html=True)
        with col4:
            wr = f"{win_rate:.0%}" if isinstance(win_rate, float) else "N/A"
            st.markdown(mc("Win Rate", wr), unsafe_allow_html=True)
    else:
        with col1:
            st.markdown(mc("Equity", "$10,000.00"), unsafe_allow_html=True)
        with col2:
            st.markdown(mc("Daily P&L", "$0.00"), unsafe_allow_html=True)
        with col3:
            st.markdown(mc("Trades Today", "0"), unsafe_allow_html=True)
        with col4:
            st.markdown(mc("Win Rate", "N/A"), unsafe_allow_html=True)

    # Recent trades
    st.subheader("Recent Trades")
    # Status badge summary
    trades = run_async(query_db(
        """SELECT t.*, tc.regime, tc.strategy_name, tc.signal_confidence
           FROM trades t LEFT JOIN trade_context tc ON t.id = tc.trade_id
           ORDER BY t.entry_time DESC LIMIT 10"""
    ))

    if trades:
        df = pd.DataFrame(trades)
        display_cols = ["id", "symbol", "direction", "status", "entry_price",
                        "exit_price", "pnl_dollars", "pnl_r", "regime", "strategy_name"]
        available = [c for c in display_cols if c in df.columns]
        # Status badges
        if "status" in df.columns:
            counts = df["status"].value_counts().to_dict()
            badges = " ".join(f'<span class="status-badge {k}">{k}: {v}</span>' for k, v in counts.items())
            st.markdown(f'<div style="margin-bottom:0.75rem">{badges}</div>', unsafe_allow_html=True)
        st.dataframe(df[available], use_container_width=True, hide_index=True)
    else:
        st.info("No trades recorded yet. Start the agent to begin trading.")

    # AI decisions
    st.subheader("Recent AI Decisions")
    decisions = run_async(query_db(
        "SELECT * FROM ai_decisions ORDER BY timestamp DESC LIMIT 5"
    ))
    if decisions:
        for d in decisions:
            with st.expander(f"{d.get('decision_type', '?')} — {d.get('timestamp', '')[:19]}"):
                st.json(json.loads(d.get("output_decision", "{}")))
                st.caption(f"Model: {d.get('model_used')} | Tokens: {d.get('tokens_used')} | Cost: ${d.get('cost_usd', 0):.4f} | Latency: {d.get('latency_ms')}ms")


# ── Trade History ──
elif page == "Trade History":
    st.title("Trade History")

    trades = run_async(query_db(
        """SELECT t.*, tc.regime, tc.strategy_name, tc.signal_confidence
           FROM trades t LEFT JOIN trade_context tc ON t.id = tc.trade_id
           ORDER BY t.entry_time DESC LIMIT 200"""
    ))

    if trades:
        df = pd.DataFrame(trades)

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.multiselect("Status", ["win", "loss", "breakeven", "open"],
                                           default=["win", "loss", "breakeven"])
        with col2:
            direction_filter = st.multiselect("Direction", ["long", "short"],
                                              default=["long", "short"])

        mask = df["status"].isin(status_filter) & df["direction"].isin(direction_filter)
        filtered = df[mask]

        st.dataframe(filtered, use_container_width=True, hide_index=True)

        # Equity curve
        closed = filtered[filtered["status"].isin(["win", "loss", "breakeven"])].copy()
        if len(closed) > 0 and "pnl_dollars" in closed.columns:
            closed["pnl_dollars"] = pd.to_numeric(closed["pnl_dollars"], errors="coerce").fillna(0)
            closed = closed.sort_values("entry_time")
            closed["cumulative_pnl"] = closed["pnl_dollars"].cumsum()

            st.subheader("Equity Curve")
            st.line_chart(closed.set_index("entry_time")["cumulative_pnl"])
    else:
        st.info("No trade history available.")


# ── Analytics ──
elif page == "Analytics":
    st.title("Analytics")

    trades = run_async(query_db(
        """SELECT t.*, tc.regime, tc.strategy_name, tc.signal_confidence
           FROM trades t LEFT JOIN trade_context tc ON t.id = tc.trade_id
           WHERE t.status IN ('win', 'loss', 'breakeven')
           ORDER BY t.entry_time ASC"""
    ))

    if trades and len(trades) > 0:
        from src.feedback.analytics import PerformanceAnalytics
        analytics = PerformanceAnalytics()
        metrics = analytics.calculate_metrics(trades)

        # Key metrics - Row 1
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(mc("Total Trades", str(metrics["total_trades"])), unsafe_allow_html=True)
        with col2:
            st.markdown(mc("Win Rate", f"{metrics['win_rate']}%"), unsafe_allow_html=True)
        with col3:
            st.markdown(mc("Profit Factor", str(metrics["profit_factor"])), unsafe_allow_html=True)
        with col4:
            st.markdown(mc("Sharpe Ratio", str(metrics["sharpe_ratio"])), unsafe_allow_html=True)

        # Row 2
        col5, col6, col7, col8 = st.columns(4)
        pnl_c = "positive" if metrics["total_pnl"] >= 0 else "negative"
        with col5:
            st.markdown(mc("Total P&L", f"${metrics['total_pnl']:.2f}", delta_class=pnl_c), unsafe_allow_html=True)
        with col6:
            st.markdown(mc("Max Drawdown", f"${metrics['max_drawdown']:.2f}"), unsafe_allow_html=True)
        with col7:
            st.markdown(mc("Avg Win", f"{metrics['avg_win_r']:.2f}R"), unsafe_allow_html=True)
        with col8:
            st.markdown(mc("Avg Loss", f"{metrics['avg_loss_r']:.2f}R"), unsafe_allow_html=True)

        # R-multiple distribution
        st.subheader("R-Multiple Distribution")
        df = pd.DataFrame(trades)
        df["pnl_r"] = pd.to_numeric(df["pnl_r"], errors="coerce")
        st.bar_chart(df["pnl_r"].dropna())

        # Performance by regime
        st.subheader("Performance by Regime")
        regime_segments = analytics.segment_by(trades, "regime")
        if regime_segments:
            regime_df = pd.DataFrame([
                {"Regime": r, "Trades": m["total_trades"], "Win Rate": m["win_rate"],
                 "PF": m["profit_factor"], "P&L": m["total_pnl"]}
                for r, m in regime_segments.items()
            ])
            st.dataframe(regime_df, use_container_width=True, hide_index=True)
    else:
        st.info("No completed trades for analysis.")


# ── LLM Log ──
elif page == "LLM Log":
    st.title("LLM Decision Log")

    decisions = run_async(query_db(
        "SELECT * FROM ai_decisions ORDER BY timestamp DESC LIMIT 50"
    ))

    if decisions:
        # Usage summary
        total_cost = sum(d.get("cost_usd", 0) or 0 for d in decisions)
        total_tokens = sum(d.get("tokens_used", 0) or 0 for d in decisions)
        lc1, lc2 = st.columns(2)
        with lc1:
            st.markdown(mc("Total API Cost", f"${total_cost:.4f}"), unsafe_allow_html=True)
        with lc2:
            st.markdown(mc("Total Tokens", f"{total_tokens:,}"), unsafe_allow_html=True)

        for d in decisions:
            ts = d.get("timestamp", "")[:19]
            dtype = d.get("decision_type", "unknown")
            with st.expander(f"[{ts}] {dtype}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Input:**")
                    try:
                        st.json(json.loads(d.get("input_context", "{}")))
                    except (json.JSONDecodeError, TypeError):
                        st.text(str(d.get("input_context", "")))
                with col2:
                    st.write("**Output:**")
                    try:
                        st.json(json.loads(d.get("output_decision", "{}")))
                    except (json.JSONDecodeError, TypeError):
                        st.text(str(d.get("output_decision", "")))
    else:
        st.info("No AI decisions logged yet.")


# ── Settings ──
elif page == "Settings":
    st.title("Settings")

    CONFIG_PATH = Path("config/default.yaml")

    # Load config into session state
    if "_cfg" not in st.session_state:
        if CONFIG_PATH.exists():
            st.session_state["_cfg"] = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        else:
            st.session_state["_cfg"] = {}

    cfg = st.session_state["_cfg"]

    # ── Top controls ──
    top_col1, top_col2, top_col3 = st.columns([1, 1, 2])

    with top_col1:
        save_clicked = st.button("Save Changes", type="primary", use_container_width=True)
    with top_col2:
        reset_clicked = st.button("Reset to Defaults", use_container_width=True)
    with top_col3:
        auto_tune = cfg.get("features", {}).get("auto_tune_enabled", False)
        new_auto_tune = st.toggle("AI Auto-Tune", value=auto_tune,
                                   help="Allow AI to automatically adjust parameters based on performance analysis")
        cfg.setdefault("features", {})["auto_tune_enabled"] = new_auto_tune

    validation_errors = []

    # ── 1. Risk Management ──
    with st.expander("Risk Management", expanded=True):
        risk = cfg.setdefault("risk", {})

        col1, col2 = st.columns(2)
        with col1:
            risk["risk_per_trade_pct"] = st.slider(
                "Risk per Trade (%)", 0.1, 5.0,
                value=float(risk.get("risk_per_trade_pct", 0.01)) * 100,
                step=0.1, format="%.1f%%",
                help="Percentage of account risked on each trade"
            ) / 100

            risk["max_risk_per_trade_pct"] = st.slider(
                "Max Risk per Trade (%) — Hard Ceiling", 0.5, 5.0,
                value=float(risk.get("max_risk_per_trade_pct", 0.02)) * 100,
                step=0.1, format="%.1f%%",
                help="Absolute maximum risk, even if AI suggests more"
            ) / 100

            risk["max_daily_loss_pct"] = st.slider(
                "Max Daily Loss (%)", 1.0, 10.0,
                value=float(risk.get("max_daily_loss_pct", 0.03)) * 100,
                step=0.5, format="%.1f%%",
                help="Stop trading for the day at this loss level"
            ) / 100

            risk["max_weekly_loss_pct"] = st.slider(
                "Max Weekly Loss (%)", 2.0, 15.0,
                value=float(risk.get("max_weekly_loss_pct", 0.05)) * 100,
                step=0.5, format="%.1f%%",
                help="Stop trading for the week at this loss level"
            ) / 100

        with col2:
            risk["min_risk_reward"] = st.number_input(
                "Minimum R:R", min_value=1.0, max_value=5.0,
                value=float(risk.get("min_risk_reward", 2.5)), step=0.5,
                help="Minimum risk-reward ratio for entry"
            )
            risk["max_contracts"] = st.number_input(
                "Max Contracts", min_value=1, max_value=20,
                value=int(risk.get("max_contracts", 4)),
                help="Maximum contracts per trade"
            )
            risk["max_concurrent_positions"] = st.number_input(
                "Max Concurrent Positions", min_value=1, max_value=10,
                value=int(risk.get("max_concurrent_positions", 2))
            )
            risk["max_trades_per_day"] = st.number_input(
                "Max Trades per Day", min_value=1, max_value=20,
                value=int(risk.get("max_trades_per_day", 6)),
                help="Prevent overtrading"
            )

        col3, col4 = st.columns(2)
        with col3:
            risk["cooldown_after_loss_minutes"] = st.number_input(
                "Cooldown After Loss (min)", min_value=0, max_value=120,
                value=int(risk.get("cooldown_after_loss_minutes", 15))
            )
        with col4:
            risk["news_blackout_minutes"] = st.number_input(
                "News Blackout (min)", min_value=0, max_value=60,
                value=int(risk.get("news_blackout_minutes", 15)),
                help="No trading N minutes before/after high-impact news"
            )

        st.markdown("**Circuit Breaker**")
        cb1, cb2 = st.columns(2)
        with cb1:
            risk["flash_crash_atr_multiple"] = st.number_input(
                "Flash Crash ATR Multiple", min_value=2.0, max_value=10.0,
                value=float(risk.get("flash_crash_atr_multiple", 5.0)), step=0.5
            )
            risk["max_consecutive_losses"] = st.number_input(
                "Max Consecutive Losses", min_value=1, max_value=10,
                value=int(risk.get("max_consecutive_losses", 3))
            )
        with cb2:
            risk["spread_blowout_multiple"] = st.number_input(
                "Spread Blowout Multiple", min_value=2.0, max_value=10.0,
                value=float(risk.get("spread_blowout_multiple", 4.0)), step=0.5
            )
            risk["volatility_shift_atr_multiple"] = st.number_input(
                "Volatility Shift ATR Multiple", min_value=1.0, max_value=5.0,
                value=float(risk.get("volatility_shift_atr_multiple", 2.0)), step=0.5
            )

        # Validation
        if risk["max_risk_per_trade_pct"] < risk["risk_per_trade_pct"]:
            validation_errors.append("Max risk per trade must be >= risk per trade")
            st.error("Max risk per trade must be >= risk per trade")

    # ── 2. Signal Engine ──
    with st.expander("Signal Engine"):
        signals = cfg.setdefault("signals", {})

        signals["min_confidence_threshold"] = st.slider(
            "Min Confidence Threshold", 0.0, 1.0,
            value=float(signals.get("min_confidence_threshold", 0.65)),
            step=0.05, help="Minimum confidence score to trigger a trade"
        )

        st.markdown("**Signal Weights** (must sum to 1.0)")
        weights = signals.setdefault("signal_weights", {})
        wc1, wc2 = st.columns(2)
        with wc1:
            weights["ict_pattern"] = st.slider("ICT Patterns", 0.0, 1.0, float(weights.get("ict_pattern", 0.35)), 0.05)
            weights["htf_bias"] = st.slider("HTF Bias", 0.0, 1.0, float(weights.get("htf_bias", 0.20)), 0.05)
            weights["indicators"] = st.slider("Indicators", 0.0, 1.0, float(weights.get("indicators", 0.15)), 0.05)
        with wc2:
            weights["order_flow"] = st.slider("Order Flow", 0.0, 1.0, float(weights.get("order_flow", 0.15)), 0.05)
            weights["key_levels"] = st.slider("Key Levels", 0.0, 1.0, float(weights.get("key_levels", 0.10)), 0.05)
            weights["intermarket"] = st.slider("Intermarket", 0.0, 1.0, float(weights.get("intermarket", 0.05)), 0.05)

        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) < 0.011:
            st.success(f"Total weight: {total_weight:.2f} ✓")
        else:
            st.error(f"Total weight: {total_weight:.2f} — must equal 1.0")
            validation_errors.append("Signal weights must sum to 1.0")

        st.markdown("**Indicator Periods**")
        ip1, ip2, ip3 = st.columns(3)
        with ip1:
            signals["adx_period"] = st.number_input("ADX Period", 5, 30, int(signals.get("adx_period", 14)))
            signals["rsi_period"] = st.number_input("RSI Period", 5, 30, int(signals.get("rsi_period", 14)))
        with ip2:
            signals["atr_period"] = st.number_input("ATR Period", 5, 30, int(signals.get("atr_period", 14)))
            signals["swing_lookback"] = st.number_input("Swing Lookback", 2, 20, int(signals.get("swing_lookback", 5)))
        with ip3:
            signals["fvg_min_ticks"] = st.number_input("FVG Min Ticks", 1, 20, int(signals.get("fvg_min_ticks", 4)))
            signals["displacement_atr_multiple"] = st.number_input("Displacement ATR Mult", 1.0, 5.0, float(signals.get("displacement_atr_multiple", 2.0)), 0.5)

    # ── 3. Trade Management ──
    with st.expander("Trade Management"):
        trade = cfg.setdefault("trade", {})
        tc1, tc2 = st.columns(2)
        with tc1:
            trade["partial_tp_r"] = st.number_input("Partial TP (R)", 0.5, 3.0, float(trade.get("partial_tp_r", 1.0)), 0.5)
            trade["full_tp_r"] = st.number_input("Full TP (R)", 1.0, 5.0, float(trade.get("full_tp_r", 2.5)), 0.5)
            trade["trailing_stop_atr_multiple"] = st.number_input("Trailing Stop ATR Mult", 0.5, 4.0, float(trade.get("trailing_stop_atr_multiple", 1.5)), 0.25)
        with tc2:
            trade["partial_tp_pct"] = st.slider("Partial TP Close %", 10, 90, int(float(trade.get("partial_tp_pct", 0.5)) * 100), 5, format="%d%%") / 100
            trade["break_even_trigger_pct"] = st.slider("Break-Even Trigger %", 10, 80, int(float(trade.get("break_even_trigger_pct", 0.4)) * 100), 5, format="%d%%") / 100
            trade["limit_order_timeout_minutes"] = st.number_input("Limit Order Timeout (min)", 5, 120, int(trade.get("limit_order_timeout_minutes", 30)))

        if trade["partial_tp_r"] >= trade["full_tp_r"]:
            st.error("Partial TP must be less than Full TP")
            validation_errors.append("Partial TP must be less than Full TP")

    # ── 4. Auto-Scaling ──
    with st.expander("Auto-Scaling (MNQ to NQ)"):
        scaling = cfg.setdefault("scaling", {})
        sc1, sc2 = st.columns(2)
        with sc1:
            scaling["mnq_to_nq_threshold"] = st.number_input(
                "MNQ → NQ Threshold ($)", 10000, 100000,
                int(scaling.get("mnq_to_nq_threshold", 25000)), 5000,
                help="Allow NQ trades above this equity"
            )
            scaling["max_mnq_contracts"] = st.number_input("Max MNQ Contracts", 1, 20, int(scaling.get("max_mnq_contracts", 4)))
        with sc2:
            scaling["nq_primary_threshold"] = st.number_input(
                "NQ Primary Threshold ($)", 20000, 200000,
                int(scaling.get("nq_primary_threshold", 50000)), 5000,
                help="Default to NQ above this equity"
            )
            scaling["max_nq_contracts"] = st.number_input("Max NQ Contracts", 1, 10, int(scaling.get("max_nq_contracts", 2)))

        if scaling["nq_primary_threshold"] <= scaling["mnq_to_nq_threshold"]:
            st.error("NQ Primary threshold must be > MNQ→NQ threshold")
            validation_errors.append("NQ Primary threshold must be > MNQ→NQ threshold")

    # ── 5. AI / Claude ──
    with st.expander("AI / Claude API"):
        llm = cfg.setdefault("llm", {})
        lc1, lc2 = st.columns(2)
        with lc1:
            llm["daily_budget_usd"] = st.number_input("Daily Budget ($)", 0.0, 50.0, float(llm.get("daily_budget_usd", 5.0)), 1.0)
            llm["realtime_timeout_seconds"] = st.number_input("Realtime Timeout (s)", 1, 60, int(llm.get("realtime_timeout_seconds", 10)))
            llm["cache_ttl_seconds"] = st.number_input("Cache TTL (s)", 0, 3600, int(llm.get("cache_ttl_seconds", 300)), 60)
        with lc2:
            llm["analysis_timeout_seconds"] = st.number_input("Analysis Timeout (s)", 10, 300, int(llm.get("analysis_timeout_seconds", 60)))
            llm["regime_update_interval_minutes"] = st.number_input("Regime Update Interval (min)", 5, 120, int(llm.get("regime_update_interval_minutes", 30)))
        st.caption(f"Models: {llm.get('realtime_model', 'N/A')} (realtime) / {llm.get('analysis_model', 'N/A')} (analysis) — :orange[restart required to change]")

    # ── 6. Kill Zones ──
    with st.expander("Kill Zones (Advisory)"):
        kz = cfg.setdefault("kill_zones", {})
        kz["confidence_penalty_outside_kz"] = st.slider(
            "Confidence Penalty Outside Kill Zones", 0.0, 0.5,
            float(kz.get("confidence_penalty_outside_kz", 0.15)), 0.05
        )
        st.caption("Kill zones affect confidence scoring but do NOT block entries. Overnight holds allowed.")
        for zone_key in ["london_open", "ny_open", "ny_lunch", "ny_pm", "overnight"]:
            zone = kz.setdefault(zone_key, {})
            zc1, zc2, zc3 = st.columns([2, 1, 1])
            with zc1:
                zone["label"] = st.text_input(f"Label", zone.get("label", zone_key), key=f"kz_{zone_key}_label")
            with zc2:
                zone["start"] = st.text_input(f"Start (HH:MM)", zone.get("start", "00:00"), key=f"kz_{zone_key}_start")
            with zc3:
                zone["end"] = st.text_input(f"End (HH:MM)", zone.get("end", "00:00"), key=f"kz_{zone_key}_end")

    # ── 7. Notifications ──
    with st.expander("Notifications"):
        notif = cfg.setdefault("notifications", {})
        notif["enabled"] = st.toggle("Enable Discord Notifications", value=bool(notif.get("enabled", False)))
        if notif["enabled"]:
            notif["discord_webhook_url"] = st.text_input(
                "Discord Webhook URL",
                value=notif.get("discord_webhook_url", ""),
                type="password"
            )

    # ── 8. Feature Flags ──
    with st.expander("Feature Flags"):
        features = cfg.setdefault("features", {})
        fc1, fc2 = st.columns(2)
        with fc1:
            features["paper_mode"] = st.toggle("Paper Mode", value=bool(features.get("paper_mode", True)),
                                                help="Must be ON for simulator. :orange[Restart required]")
            features["llm_enabled"] = st.toggle("LLM Enabled", value=bool(features.get("llm_enabled", True)),
                                                 help="Enable Claude AI for decisions. :orange[Restart required]")
            features["order_flow_enabled"] = st.toggle("Order Flow", value=bool(features.get("order_flow_enabled", True)))
        with fc2:
            features["intermarket_enabled"] = st.toggle("Intermarket", value=bool(features.get("intermarket_enabled", True)))
            features["shadow_runner_enabled"] = st.toggle("Shadow Runner (A/B)", value=bool(features.get("shadow_runner_enabled", False)),
                                                           help=":orange[Restart required]")
            features["overnight_holds"] = st.toggle("Overnight Holds", value=bool(features.get("overnight_holds", True)))

        st.markdown("**Auto-Tune Settings**")
        features["auto_tune_max_changes_per_day"] = st.number_input(
            "Max Auto-Tune Changes/Day", 1, 10, int(features.get("auto_tune_max_changes_per_day", 3))
        )
        features["auto_tune_min_trades_required"] = st.number_input(
            "Min Trades Before Tuning", 5, 100, int(features.get("auto_tune_min_trades_required", 20))
        )

    # ── 9. System Info + AI Tuning History ──
    with st.expander("System Info"):
        si1, si2 = st.columns(2)
        with si1:
            st.markdown("**Environment**")
            st.code(cfg.get("tradovate", {}).get("environment", "demo"))
            st.markdown("**Primary Symbol**")
            st.code(cfg.get("symbols", {}).get("primary", "MNQU5"))
            st.markdown("**Database**")
            st.code("Connected" if Path(DB_PATH).exists() else "Not found")
        with si2:
            st.markdown("**MNQ Specs**")
            mnq = cfg.get("mnq", {})
            st.json({"tick_value": mnq.get("tick_value", 0.5), "point_value": mnq.get("point_value", 2.0),
                      "commission": mnq.get("commission_per_contract", 0.62), "overnight_margin": mnq.get("overnight_margin", 2100)})
            st.markdown("**NQ Specs**")
            nq = cfg.get("nq", {})
            st.json({"tick_value": nq.get("tick_value", 5.0), "point_value": nq.get("point_value", 20.0),
                      "commission": nq.get("commission_per_contract", 0.82), "overnight_margin": nq.get("overnight_margin", 21000)})

    with st.expander("AI Tuning History"):
        adjustments = run_async(query_db(
            "SELECT * FROM strategy_adjustments ORDER BY id DESC LIMIT 20"
        ))
        if adjustments:
            for adj in adjustments:
                applied = "✅ Applied" if adj.get("applied") else "📋 Suggested"
                st.markdown(f"**[{adj.get('created_at', '')[:10]}] {applied}** — {adj.get('review_type', '')}")
                st.caption(adj.get("adjustment_text", ""))
                if adj.get("parameters_json"):
                    try:
                        st.json(json.loads(adj["parameters_json"]))
                    except (json.JSONDecodeError, TypeError):
                        pass
                st.divider()
        else:
            st.info("No tuning history yet. Enable Auto-Tune and trade for the AI to start optimizing.")

    # ── Save / Reset logic ──
    if save_clicked:
        if validation_errors:
            st.error(f"Cannot save — {len(validation_errors)} error(s): {', '.join(validation_errors)}")
        else:
            CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
            st.success("Settings saved to config/default.yaml")

    if reset_clicked:
        if CONFIG_PATH.exists():
            st.session_state["_cfg"] = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            st.rerun()
