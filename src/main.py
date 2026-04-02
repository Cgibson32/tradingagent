"""AI Trading Agent — Main Entry Point.

Usage:
    python -m src.main --mode paper     # Paper trading (default)
    python -m src.main --mode backtest  # Backtesting on historical data
    python -m src.main --mode live      # Live trading (requires explicit confirmation)
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timezone

import structlog

from src.config import load_config, AppConfig
from src.data.auth import TokenManager
from src.data.historical import HistoricalDataManager
from src.data.intermarket import IntermarketTracker
from src.data.md_websocket import MarketDataWebSocket
from src.data.models import Quote
from src.data.order_flow import OrderFlowAnalyzer
from src.data.tradovate_client import TradovateRestClient
from src.data.websocket import TradingWebSocket

logger = structlog.get_logger(__name__)


def setup_logging(config: AppConfig) -> None:
    """Configure structlog for JSON or console output."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if config.logging.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(config.logging.level)
        ),
    )


async def run_backtest(config: AppConfig) -> None:
    """Run backtesting on historical data."""
    logger.info("mode.backtest.starting")

    historical = HistoricalDataManager(config.storage.parquet_dir)

    # Check for existing data
    symbol = config.symbols.primary
    for tf in [config.timeframes.entry] + config.timeframes.htf_bias:
        has_data = historical.has_sufficient_data(symbol, tf)
        logger.info(
            "backtest.data_check",
            symbol=symbol,
            timeframe=tf,
            has_data=has_data,
        )
        if has_data:
            df = historical.load_candles(symbol, tf)
            logger.info(
                "backtest.loaded",
                symbol=symbol,
                timeframe=tf,
                bars=len(df),
                start=str(df["timestamp"].iloc[0]) if len(df) > 0 else "N/A",
                end=str(df["timestamp"].iloc[-1]) if len(df) > 0 else "N/A",
            )

    logger.info("mode.backtest.complete", message="Backtest infrastructure ready. Phase 2 will add strategy logic.")


async def run_paper(config: AppConfig) -> None:
    """Run paper trading on Tradovate simulator."""
    logger.info("mode.paper.starting")

    # Initialize components
    urls = config.tradovate.active_urls
    secrets = config.secrets

    token_manager = TokenManager(
        base_url=urls.rest,
        username=secrets.tradovate_username,
        password=secrets.tradovate_password.get_secret_value(),
        app_id=secrets.tradovate_app_id,
        client_id=secrets.tradovate_client_id,
        client_secret=secrets.tradovate_client_secret.get_secret_value(),
        renew_minutes=config.tradovate.auth.token_renew_minutes,
    )

    rest_client = TradovateRestClient(token_manager, urls.rest)
    order_flow = OrderFlowAnalyzer()
    intermarket = IntermarketTracker()
    historical = HistoricalDataManager(config.storage.parquet_dir)

    # Shutdown handler
    shutdown_event = asyncio.Event()

    def handle_shutdown(sig):
        logger.info("shutdown.signal_received", signal=sig.name)
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_shutdown, sig)

    try:
        # 1. Authenticate
        logger.info("paper.authenticating")
        await token_manager.authenticate()

        # 2. Look up contracts
        logger.info("paper.looking_up_contracts")
        nq_contract = await rest_client.find_contract(config.symbols.primary)
        logger.info("paper.contract_found", symbol=config.symbols.primary, contract=nq_contract)

        # 3. List accounts
        accounts = await rest_client.list_accounts()
        if accounts:
            account = accounts[0]
            logger.info("paper.account", account_id=account.get("id"), name=account.get("name"))

        # 4. Connect Trading WebSocket
        trading_ws = TradingWebSocket(urls.ws_trading, token_manager.access_token)

        # 5. Connect Market Data WebSocket
        md_ws = MarketDataWebSocket(urls.ws_market_data, token_manager.md_access_token)

        # Register token refresh handler
        async def on_token_refresh(access_token, md_token):
            trading_ws.update_token(access_token)
            md_ws.update_token(md_token)
            logger.info("paper.tokens_refreshed")

        token_manager.on_refresh(on_token_refresh)

        # Quote handler
        async def on_nq_quote(quote: Quote):
            order_flow.process_quote(quote)
            logger.debug(
                "paper.quote",
                bid=quote.bid_price,
                ask=quote.ask_price,
                last=quote.last_price,
                spread=quote.spread,
            )

        md_ws.on_quote(config.symbols.primary, on_nq_quote)

        # Start WebSocket connections as background tasks
        trading_task = asyncio.create_task(trading_ws.connect())
        md_task = asyncio.create_task(md_ws.connect())

        # Wait for connections to establish
        await asyncio.sleep(3)

        if md_ws.is_connected:
            # Subscribe to NQ quotes
            await md_ws.subscribe_quotes(config.symbols.primary)
            logger.info("paper.subscribed_quotes", symbol=config.symbols.primary)

            # Subscribe to DOM for order flow
            if config.features.order_flow_enabled:
                await md_ws.subscribe_dom(config.symbols.primary)
                logger.info("paper.subscribed_dom", symbol=config.symbols.primary)

            # Subscribe to intermarket instruments
            if config.features.intermarket_enabled:
                for im_symbol in config.symbols.intermarket:
                    try:
                        await md_ws.subscribe_quotes(im_symbol)
                        logger.info("paper.subscribed_intermarket", symbol=im_symbol)
                    except Exception as e:
                        logger.warning("paper.intermarket_sub_failed", symbol=im_symbol, error=str(e))

        if trading_ws.is_connected:
            await trading_ws.start_sync()
            logger.info("paper.trading_sync_started")

        # Ensure warm data for indicators
        if md_ws.is_connected:
            all_timeframes = [config.timeframes.entry] + config.timeframes.htf_bias + config.timeframes.context
            await historical.ensure_warm_data(md_ws, config.symbols.primary, all_timeframes)

        logger.info("paper.running", message="Agent is live. Press Ctrl+C to stop.")

        # Main loop — wait for shutdown
        await shutdown_event.wait()

    except Exception as e:
        logger.error("paper.error", error=str(e), exc_info=True)
    finally:
        logger.info("paper.shutting_down")
        await token_manager.close()
        await rest_client.close()
        if "trading_ws" in locals():
            await trading_ws.close()
        if "md_ws" in locals():
            await md_ws.close()
        logger.info("paper.shutdown_complete")


async def run_live(config: AppConfig) -> None:
    """Run live trading. Same as paper but with real orders."""
    if config.features.paper_mode:
        logger.error("live.blocked", message="Paper mode is enabled in config. Set features.paper_mode=false to trade live.")
        return
    # Live trading uses the same flow as paper with real Tradovate orders
    logger.info("mode.live.starting")
    await run_paper(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Trading Agent for NQ Futures")
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
        default="paper",
        help="Trading mode (default: paper)",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.mode)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Config file not found: {e}", file=sys.stderr)
        sys.exit(1)

    setup_logging(config)
    logger.info("agent.starting", mode=args.mode, symbol=config.symbols.primary)

    if args.mode == "backtest":
        asyncio.run(run_backtest(config))
    elif args.mode == "paper":
        asyncio.run(run_paper(config))
    elif args.mode == "live":
        asyncio.run(run_live(config))


if __name__ == "__main__":
    main()
