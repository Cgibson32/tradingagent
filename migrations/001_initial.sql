-- AI Trading Agent — Initial Database Schema
-- SQLite database for trade journal, performance tracking, and AI decisions

-- ── Candle data cache ─────────────────────────────────
CREATE TABLE IF NOT EXISTS candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER DEFAULT 0,
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf ON candles(symbol, timeframe, timestamp);

-- ── Trade journal ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,                    -- UUID
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,                -- 'long' or 'short'
    entry_time TEXT,
    exit_time TEXT,
    entry_price REAL,
    exit_price REAL,
    position_size INTEGER,
    stop_loss REAL,
    take_profit REAL,
    pnl_dollars REAL,
    pnl_r REAL,                             -- P&L in R-multiples
    fees REAL DEFAULT 0,
    status TEXT DEFAULT 'open',              -- 'win', 'loss', 'breakeven', 'open'
    is_shadow INTEGER DEFAULT 0             -- 1 = shadow/paper comparison trade
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol, entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);

-- ── Trade context (full snapshot at entry) ────────────
CREATE TABLE IF NOT EXISTS trade_context (
    trade_id TEXT PRIMARY KEY REFERENCES trades(id),
    regime TEXT,                             -- Market regime at entry
    htf_bias TEXT,                          -- HTF bias at entry
    signal_confidence REAL,                 -- Aggregated confidence score
    strategy_name TEXT,                     -- Which strategy generated the signal
    indicators_json TEXT,                   -- JSON snapshot of indicator values
    ict_patterns_json TEXT,                 -- JSON of active ICT patterns
    key_levels_json TEXT,                   -- JSON of key levels at entry
    order_flow_json TEXT,                   -- JSON of order flow state
    intermarket_json TEXT,                  -- JSON of intermarket context
    kill_zone TEXT,                         -- Active kill zone at entry
    minutes_until_close REAL,              -- Time remaining at entry
    llm_reasoning TEXT,                     -- Claude's decision reasoning
    llm_journal TEXT,                       -- Post-trade Claude journal entry
    news_events TEXT                        -- Relevant news near the trade
);

-- ── Daily performance summary ─────────────────────────
CREATE TABLE IF NOT EXISTS daily_stats (
    date TEXT PRIMARY KEY,
    starting_equity REAL,
    ending_equity REAL,
    high_water_mark REAL,
    drawdown_floor REAL,
    total_pnl REAL,
    trades_taken INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    win_rate REAL,
    avg_win REAL,
    avg_loss REAL,
    max_win REAL,
    max_loss REAL,
    profit_factor REAL,
    dominant_regime TEXT,
    avg_order_latency_ms REAL,
    avg_llm_latency_ms REAL
);

-- ── AI decision log ───────────────────────────────────
CREATE TABLE IF NOT EXISTS ai_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    decision_type TEXT,                     -- 'regime', 'signal_eval', 'journal', 'weekly_review'
    input_context TEXT,                     -- JSON: what was sent to Claude
    output_decision TEXT,                   -- JSON: what Claude returned
    model_used TEXT,
    tokens_used INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    latency_ms INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_ai_decisions_type ON ai_decisions(decision_type, timestamp);

-- ── Strategy adjustments from learning loop ───────────
CREATE TABLE IF NOT EXISTS strategy_adjustments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    review_type TEXT,                       -- 'per_trade', 'daily', 'weekly', 'monthly'
    adjustment_text TEXT,                   -- What Claude recommends
    parameters_json TEXT,                   -- Suggested parameter changes as JSON
    applied INTEGER DEFAULT 0,              -- Whether it was applied
    impact_notes TEXT                       -- Later: did this help?
);

-- ── Shadow trades (A/B testing) ───────────────────────
CREATE TABLE IF NOT EXISTS shadow_trades (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_time TEXT,
    exit_time TEXT,
    entry_price REAL,
    exit_price REAL,
    position_size INTEGER,
    stop_loss REAL,
    take_profit REAL,
    pnl_dollars REAL,
    pnl_r REAL,
    strategy_variant TEXT,                  -- Name of the shadow strategy variant
    confidence REAL
);

-- ── Economic calendar events ──────────────────────────
CREATE TABLE IF NOT EXISTS economic_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_name TEXT NOT NULL,
    datetime_et TEXT NOT NULL,
    impact_level TEXT NOT NULL,             -- 'high', 'medium', 'low'
    forecast TEXT,
    previous TEXT,
    actual TEXT,
    fetched_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_economic_events_time ON economic_events(datetime_et);

-- ── Agent state checkpoints (for crash recovery) ──────
CREATE TABLE IF NOT EXISTS agent_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    state_json TEXT NOT NULL,               -- Full agent state snapshot
    open_positions_json TEXT,               -- Current positions
    daily_pnl REAL,
    drawdown_floor REAL,
    trades_today INTEGER,
    consecutive_losses INTEGER
);

-- ── News cache ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS news_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    headline TEXT,
    url TEXT,
    published_at TEXT,
    fetched_at TEXT,
    sentiment_score REAL,
    symbols TEXT
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news_cache(published_at);
