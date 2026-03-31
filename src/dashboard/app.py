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
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import streamlit as st
except ImportError:
    print("Install streamlit: pip install streamlit")
    sys.exit(1)

import aiosqlite

DB_PATH = "data/trading.db"


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
st.set_page_config(page_title="AI Trading Agent", page_icon="📈", layout="wide")

# ── Sidebar Navigation ──
page = st.sidebar.radio("Navigation", ["Live View", "Trade History", "Analytics", "LLM Log", "Settings"])


# ── Live View ──
if page == "Live View":
    st.title("📈 Live View")

    # Account summary
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

        col1.metric("Equity", f"${equity:,.2f}")
        col2.metric("Daily P&L", f"${daily_pnl:+,.2f}",
                     delta=f"{daily_pnl:+.2f}", delta_color="normal")
        col3.metric("Trades Today", trades_today)
        col4.metric("Win Rate", f"{win_rate:.0%}" if isinstance(win_rate, float) else "N/A")
    else:
        col1.metric("Equity", "$10,000.00")
        col2.metric("Daily P&L", "$0.00")
        col3.metric("Trades Today", 0)
        col4.metric("Win Rate", "N/A")

    # Recent trades
    st.subheader("Recent Trades")
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
        st.dataframe(df[available], use_container_width=True)
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
    st.title("📋 Trade History")

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

        st.dataframe(filtered, use_container_width=True)

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
    st.title("📊 Analytics")

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

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", metrics["total_trades"])
        col2.metric("Win Rate", f"{metrics['win_rate']}%")
        col3.metric("Profit Factor", metrics["profit_factor"])
        col4.metric("Sharpe Ratio", metrics["sharpe_ratio"])

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Total P&L", f"${metrics['total_pnl']:.2f}")
        col6.metric("Max Drawdown", f"${metrics['max_drawdown']:.2f}")
        col7.metric("Avg Win", f"{metrics['avg_win_r']:.2f}R")
        col8.metric("Avg Loss", f"{metrics['avg_loss_r']:.2f}R")

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
            st.dataframe(regime_df, use_container_width=True)
    else:
        st.info("No completed trades for analysis.")


# ── LLM Log ──
elif page == "LLM Log":
    st.title("🤖 LLM Decision Log")

    decisions = run_async(query_db(
        "SELECT * FROM ai_decisions ORDER BY timestamp DESC LIMIT 50"
    ))

    if decisions:
        # Usage summary
        total_cost = sum(d.get("cost_usd", 0) or 0 for d in decisions)
        total_tokens = sum(d.get("tokens_used", 0) or 0 for d in decisions)
        st.metric("Total API Cost (shown)", f"${total_cost:.4f}")
        st.metric("Total Tokens (shown)", f"{total_tokens:,}")

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
    st.title("⚙️ Settings")

    st.subheader("Current Configuration")

    config_path = Path("config/default.yaml")
    if config_path.exists():
        st.code(config_path.read_text(), language="yaml")
    else:
        st.warning("Config file not found.")

    st.subheader("System Status")
    st.json({
        "database": "Connected" if Path(DB_PATH).exists() else "Not found",
        "config": "Loaded" if config_path.exists() else "Missing",
    })

    # Checkpoint info
    checkpoints = run_async(query_db(
        "SELECT * FROM agent_checkpoints ORDER BY id DESC LIMIT 1"
    ))
    if checkpoints:
        st.subheader("Last Checkpoint")
        st.json(checkpoints[0])
