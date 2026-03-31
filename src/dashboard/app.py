"""Streamlit dashboard for the NQ trading agent.

Pages:
- Live View: positions, P&L, regime, signals, Tradeify rule status
- Trade History: filterable table, equity curve, drawdown chart
- Analytics: win rate over time, by regime, R-distribution
- LLM Log: AI decisions, journal entries, weekly reviews
- Settings: config viewer, system status

Run: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DB_PATH = "data/trading.db"


def get_db():
    """Get SQLite connection."""
    db_path = Path(DB_PATH)
    if not db_path.exists():
        return None
    return sqlite3.connect(str(db_path))


def main():
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Run: pip install streamlit")
        return

    st.set_page_config(
        page_title="NQ Trading Agent",
        page_icon="📈",
        layout="wide",
    )

    st.sidebar.title("NQ Trading Agent")
    page = st.sidebar.radio(
        "Navigation",
        ["Live View", "Trade History", "Analytics", "LLM Log", "Settings"],
    )

    conn = get_db()
    if conn is None:
        st.warning("No database found. Run the trading agent first to generate data.")
        return

    if page == "Live View":
        render_live_view(conn)
    elif page == "Trade History":
        render_trade_history(conn)
    elif page == "Analytics":
        render_analytics(conn)
    elif page == "LLM Log":
        render_llm_log(conn)
    elif page == "Settings":
        render_settings()

    conn.close()


def render_live_view(conn):
    import streamlit as st

    st.header("Live View")

    col1, col2, col3, col4 = st.columns(4)

    # Daily stats
    try:
        daily = pd.read_sql(
            "SELECT * FROM daily_stats ORDER BY date DESC LIMIT 1", conn
        )
        if len(daily) > 0:
            row = daily.iloc[0]
            col1.metric("Daily P&L", f"${row.get('total_pnl', 0):,.2f}")
            col2.metric("Trades Today", int(row.get("trades_taken", 0)))
            col3.metric("Win Rate", f"{row.get('win_rate', 0):.0%}")
            col4.metric("Equity", f"${row.get('ending_equity', 150000):,.2f}")
        else:
            col1.metric("Daily P&L", "$0.00")
            col2.metric("Trades Today", 0)
            col3.metric("Win Rate", "N/A")
            col4.metric("Equity", "$150,000")
    except Exception:
        st.info("No daily stats available yet.")

    # Recent trades
    st.subheader("Recent Trades")
    try:
        trades = pd.read_sql(
            """SELECT t.id, t.direction, t.entry_price, t.exit_price,
                      t.pnl_dollars, t.pnl_r, t.status, t.entry_time,
                      tc.strategy_name, tc.regime
               FROM trades t LEFT JOIN trade_context tc ON t.id = tc.trade_id
               WHERE t.is_shadow = 0
               ORDER BY t.entry_time DESC LIMIT 10""",
            conn,
        )
        if len(trades) > 0:
            st.dataframe(trades, use_container_width=True)
        else:
            st.info("No trades yet.")
    except Exception:
        st.info("No trade data available.")


def render_trade_history(conn):
    import streamlit as st
    import matplotlib.pyplot as plt

    st.header("Trade History")

    try:
        trades = pd.read_sql(
            """SELECT t.*, tc.regime, tc.strategy_name, tc.signal_confidence
               FROM trades t LEFT JOIN trade_context tc ON t.id = tc.trade_id
               WHERE t.is_shadow = 0 AND t.status IN ('win', 'loss', 'breakeven')
               ORDER BY t.entry_time""",
            conn,
        )
    except Exception:
        st.info("No trade history available.")
        return

    if len(trades) == 0:
        st.info("No closed trades yet.")
        return

    # Equity curve
    st.subheader("Equity Curve")
    trades["cumulative_pnl"] = trades["pnl_dollars"].cumsum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(trades)), trades["cumulative_pnl"], linewidth=1.5)
    ax.fill_between(range(len(trades)), 0, trades["cumulative_pnl"], alpha=0.1)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # R-multiple distribution
    st.subheader("P&L Distribution (R-multiples)")
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    colors = ["green" if r > 0 else "red" for r in trades["pnl_r"]]
    ax2.bar(range(len(trades)), trades["pnl_r"], color=colors, alpha=0.7)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("R")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close()

    # Full table
    st.subheader("All Trades")
    st.dataframe(trades, use_container_width=True)


def render_analytics(conn):
    import streamlit as st

    st.header("Analytics")

    try:
        trades = pd.read_sql(
            """SELECT t.*, tc.regime, tc.strategy_name
               FROM trades t LEFT JOIN trade_context tc ON t.id = tc.trade_id
               WHERE t.is_shadow = 0 AND t.status IN ('win', 'loss', 'breakeven')""",
            conn,
        )
    except Exception:
        st.info("No data for analytics.")
        return

    if len(trades) == 0:
        st.info("No closed trades for analysis.")
        return

    # Summary metrics
    wins = trades[trades["pnl_r"] > 0]
    losses = trades[trades["pnl_r"] <= 0]
    total = len(trades)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", total)
    col2.metric("Win Rate", f"{len(wins) / total:.0%}" if total > 0 else "N/A")
    col3.metric("Total P&L", f"${trades['pnl_dollars'].sum():,.2f}")

    # By regime
    st.subheader("Performance by Regime")
    regime_stats = trades.groupby("regime").agg(
        trades=("id", "count"),
        win_rate=("pnl_r", lambda x: (x > 0).mean()),
        avg_r=("pnl_r", "mean"),
        total_pnl=("pnl_dollars", "sum"),
    ).round(2)
    st.dataframe(regime_stats, use_container_width=True)

    # By strategy
    st.subheader("Performance by Strategy")
    strat_stats = trades.groupby("strategy_name").agg(
        trades=("id", "count"),
        win_rate=("pnl_r", lambda x: (x > 0).mean()),
        avg_r=("pnl_r", "mean"),
        total_pnl=("pnl_dollars", "sum"),
    ).round(2)
    st.dataframe(strat_stats, use_container_width=True)


def render_llm_log(conn):
    import streamlit as st

    st.header("LLM Decision Log")

    try:
        decisions = pd.read_sql(
            """SELECT timestamp, decision_type, output_decision, model_used,
                      tokens_used, cost_usd, latency_ms
               FROM ai_decisions ORDER BY timestamp DESC LIMIT 50""",
            conn,
        )
        if len(decisions) > 0:
            st.dataframe(decisions, use_container_width=True)

            # Cost summary
            total_cost = decisions["cost_usd"].sum()
            st.metric("Total LLM Cost", f"${total_cost:.4f}")
        else:
            st.info("No AI decisions logged yet.")
    except Exception:
        st.info("No LLM log data available.")

    # Weekly reviews
    st.subheader("Strategy Reviews")
    try:
        reviews = pd.read_sql(
            """SELECT created_at, review_type, adjustment_text
               FROM strategy_adjustments ORDER BY created_at DESC LIMIT 10""",
            conn,
        )
        if len(reviews) > 0:
            for _, row in reviews.iterrows():
                st.markdown(f"**{row['created_at']}** ({row['review_type']})")
                st.text(row["adjustment_text"][:500])
                st.divider()
    except Exception:
        pass


def render_settings():
    import streamlit as st
    import yaml

    st.header("Settings (Read-Only)")

    config_path = Path("config/default.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        st.json(config)
    else:
        st.warning("Config file not found.")


if __name__ == "__main__":
    main()
