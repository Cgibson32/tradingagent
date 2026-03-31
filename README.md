# MNQ/NQ Futures AI Trading Agent

Autonomous AI-powered trading agent for MNQ (Micro Nasdaq-100) futures via Tradovate. Uses ICT/SMC patterns, Claude API for decision-making, and a continuous learning loop.

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Set up credentials
cp .env.example .env
# Edit .env with your Tradovate API keys and Anthropic API key

# 3. Run in paper mode (simulator)
python -m src.main --mode paper

# 4. Launch dashboard (separate terminal)
pip install streamlit
streamlit run src/dashboard/app.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│  DATA    │  SIGNAL  │   LLM    │ EXECUTION│  FEEDBACK   │
│  ENGINE  │  ENGINE  │  BRAIN   │  ENGINE  │  LOOP       │
│          │          │          │          │             │
│ - OHLCV  │ - ICT/SMC│ - Regime │ - Order  │ - Trade log │
│ - DOM    │ - EMA    │ - Filter │ - Risk   │ - Metrics   │
│ - News   │ - ADX    │ - Journal│ - Margin │ - Optimize  │
│ - Inter  │ - MACD   │ - Decide │ - Trail  │ - Shadow    │
│   market │ - RSI    │          │          │             │
└──────────┴──────────┴──────────┴──────────┴─────────────┘
```

## Account Configuration

| Setting | Value |
|---------|-------|
| Starting Balance | $10,000 |
| Primary Instrument | MNQ (Micro Nasdaq-100) |
| Max Contracts | 4 MNQ |
| Risk per Trade | 1% of equity |
| Max Daily Loss | 3% of equity |
| Max Weekly Loss | 5% of equity |
| Overnight Holds | Allowed |
| Auto-Scale to NQ | At $25K equity |

## Signal Engine

- **ICT/SMC Patterns**: Liquidity sweeps, displacement, FVG, BOS/CHoCH, order blocks, SMT divergence
- **Technical Indicators**: EMA (9/21/50/200), ADX, MACD, RSI, ATR, VWAP
- **HTF Bias**: 30m/1H structure filters 5m entries
- **Key Levels**: Previous day/week/month H/L/O, quarterly opens
- **Kill Zones**: Advisory confidence weighting (NY Open = highest probability)
- **Order Flow**: Cumulative delta, absorption, stacked imbalances

## Risk Management

Hard safety rules (cannot be overridden by AI):
- Max 2% risk per trade (hard ceiling)
- Daily loss limit: 3% of equity → stop for the day
- Weekly loss limit: 5% of equity → stop for the week
- Margin check before every trade (MNQ overnight margin ~$2,100)
- Circuit breaker: flash crash detection, spread blowout, CME halt detection
- 15-minute cooldown after losing trade
- News blackout 15 minutes before/after FOMC, CPI, NFP

## LLM Integration (Claude API)

- **Sonnet**: Real-time signal evaluation (<10s)
- **Opus**: Weekly deep analysis (<60s)
- $5/day budget cap with prompt caching
- Graceful degradation: system continues trading on rules alone if API unavailable

## Self-Improvement Loop

1. **Trade Journal**: Every trade logged with full context (indicators, regime, AI reasoning)
2. **Performance Analytics**: Rolling win rate, Sharpe, profit factor, segmented by regime/strategy
3. **Walk-Forward Optimizer**: Monthly parameter suggestions (bounded, never auto-deployed)
4. **Weekly Review**: Claude Opus analyzes all trades and suggests improvements
5. **Shadow Runner**: A/B test strategy variants before deploying

## Configuration

All settings in `config/default.yaml`. Override per-environment:
- `config/paper.yaml` — Paper trading (simulator)
- `config/live.yaml` — Live trading (gitignored)
- `config/secrets.yaml` — API keys (gitignored)

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
src/
├── main.py               # Entry point
├── config.py             # Configuration loader
├── data/                 # Tradovate API, WebSocket, market data
├── signals/              # Indicators, ICT patterns, kill zones
├── brain/                # Claude API, regime detection, signal evaluation
├── execution/            # Order management, risk guard, position sizing
├── feedback/             # Trade journal, analytics, optimizer
├── notifications/        # Discord alerts
└── dashboard/            # Streamlit UI
```
