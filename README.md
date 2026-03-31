# NQ Futures AI Trading Agent

Autonomous AI-powered trading agent for NQ (Nasdaq-100 E-mini) futures via Tradovate, designed for Tradeify prop firm accounts.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│  DATA    │  SIGNAL  │   LLM    │ EXECUTION│  FEEDBACK   │
│  ENGINE  │  ENGINE  │  BRAIN   │  ENGINE  │  LOOP       │
│          │          │          │          │             │
│ Tradovate│ ICT/SMC  │ Claude   │ Orders   │ Trade Log   │
│ REST+WS  │ EMA/ADX  │ Regime   │ Risk     │ Analytics   │
│ Order    │ MACD/RSI │ Evaluate │ Circuit  │ Optimizer   │
│ Flow/DOM │ Kill Zone│ Journal  │ Breaker  │ Shadow A/B  │
└──────────┴──────────┴──────────┴──────────┴─────────────┘
```

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/cgibson32/tradingagent.git
cd tradingagent
pip install -e ".[dev]"
```

### 2. Configure

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
# Edit .env with your Tradovate API credentials and Anthropic API key
```

### 3. Run

```bash
# Paper trading (simulator)
python -m src.main --mode paper

# Backtesting
python -m src.main --mode backtest

# Dashboard
streamlit run src/dashboard/app.py
```

### 4. Docker

```bash
docker compose up -d
# Dashboard at http://localhost:8501
```

## Configuration

All config in `config/default.yaml` with environment-specific overrides:

| File | Purpose |
|------|---------|
| `config/default.yaml` | All default settings |
| `config/paper.yaml` | Paper/simulator overrides |
| `config/live.yaml` | Live trading (gitignored) |
| `config/secrets.yaml` | API keys (gitignored) |

## Risk Rules (Tradeify 150K Select Daily)

| Rule | Limit | Buffer | Effective |
|------|-------|--------|-----------|
| Daily Loss | $1,750 | $250 | $1,500 |
| Trailing Drawdown | $4,500 | $500 | $4,000 |
| Max Contracts | 12 (Tradeify) | - | **4** (self-imposed) |
| Session | 9:30-4:00 ET | Close 3:45 | No entries after 3:00 PM |

These are **immutable hard rules** in `src/execution/risk_guard.py` — they cannot be overridden by config, LLM, or any other component.

## Signal Engine

### ICT/SMC Patterns
- Liquidity sweeps (stop hunts)
- Fair Value Gaps (FVG) with retest detection
- Break of Structure (BOS) / Change of Character (CHoCH)
- Order blocks (institutional supply/demand)
- SMT Divergence (NQ vs ES)
- Displacement candles (institutional momentum)

### Kill Zones (ET)
- London Open: 2:00-5:00 AM
- NY Open: 9:30-11:00 AM (highest probability)
- NY Lunch: 12:00-1:30 PM
- NY PM: 1:30-3:00 PM
- No entries after 2:30 PM, close all by 3:45 PM

### Time-Aware Trading
Every decision factors in time remaining. Confidence decays linearly from 1:30 PM, reaching zero at 2:30 PM. TP targets reduce proportionally.

## LLM Integration (Claude)

- **Sonnet**: Real-time signal evaluation (<10s)
- **Opus**: Weekly deep analysis (<60s)
- $5/day budget cap with automatic fallback to rule-based decisions
- Structured JSON prompts with Pydantic validation

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
src/
├── config.py              # YAML + env var config loader
├── main.py                # Entry point / orchestrator
├── data/                  # Tradovate API, market data, order flow
├── signals/               # Indicators, ICT patterns, kill zones
├── brain/                 # Claude AI client, prompts, regime
├── execution/             # Orders, risk guard, circuit breaker
├── feedback/              # Trade logging, analytics, optimization
├── notifications/         # Discord alerts
└── dashboard/             # Streamlit UI
```

## Runbook

### Starting the agent
```bash
python -m src.main --mode paper
```

### Switching paper → live
1. Set `features.paper_mode: false` in config
2. Set `TRADOVATE_ENVIRONMENT=live` in .env
3. Verify 2+ weeks of paper trading results first
4. Run with `--mode live`

### Adding a new instrument
1. Add symbol to `config/default.yaml` under `symbols`
2. Update contract specs in `nq` config section
3. Add to intermarket list if for context only

### Common errors
- **Auth failed**: Check Tradovate credentials in .env
- **WS disconnect**: Auto-reconnects with exponential backoff
- **Claude timeout**: Falls back to rule-based decisions automatically
- **Risk guard blocked**: Check daily P&L and drawdown status
