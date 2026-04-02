"""Discord webhook notification system.

Sends alerts for trade entries/exits, daily summaries, risk warnings,
and system errors via Discord webhook.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import structlog

logger = structlog.get_logger(__name__)


class DiscordNotifier:
    """Sends trading notifications via Discord webhook."""

    def __init__(self, webhook_url: str = "", enabled: bool = False):
        self._webhook_url = webhook_url
        self._enabled = enabled and bool(webhook_url)

    async def send(self, title: str, message: str, color: int = 0x3498DB) -> None:
        if not self._enabled:
            return
        embed = {"embeds": [{"title": title, "description": message, "color": color,
                             "timestamp": datetime.now(timezone.utc).isoformat()}]}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self._webhook_url, json=embed, timeout=10.0)
                response.raise_for_status()
        except Exception as e:
            logger.error("notifier.send_failed", error=str(e))

    async def trade_entry(self, symbol: str, direction: str, entry_price: float,
                          stop_loss: float, take_profit: float, size: int,
                          confidence: float, strategy: str) -> None:
        risk_pts = abs(entry_price - stop_loss)
        rr = abs(take_profit - entry_price) / risk_pts if risk_pts > 0 else 0
        await self.send(
            f"{direction.upper()} {symbol}",
            f"**Entry:** {entry_price:.2f} | **SL:** {stop_loss:.2f} | **TP:** {take_profit:.2f}\n"
            f"**R:R:** {rr:.1f} | **Size:** {size} | **Conf:** {confidence:.0%} | **Strategy:** {strategy}",
            color=0x2ECC71 if direction == "long" else 0xE74C3C,
        )

    async def trade_exit(self, symbol: str, direction: str, pnl_dollars: float,
                         pnl_r: float, exit_reason: str) -> None:
        status = "WIN" if pnl_dollars > 0 else "LOSS"
        await self.send(
            f"{status} — {symbol} {direction.upper()}",
            f"**P&L:** ${pnl_dollars:+.2f} ({pnl_r:+.2f}R) | **Reason:** {exit_reason}",
            color=0x2ECC71 if pnl_dollars > 0 else 0xE74C3C,
        )

    async def daily_summary(self, date: str, total_pnl: float, trades: int,
                            wins: int, losses: int, equity: float) -> None:
        wr = (wins / trades * 100) if trades > 0 else 0
        await self.send(
            f"Daily Summary — {date}",
            f"**P&L:** ${total_pnl:+.2f} | **Trades:** {trades} ({wins}W/{losses}L) | "
            f"**WR:** {wr:.0f}% | **Equity:** ${equity:,.2f}",
            color=0x2ECC71 if total_pnl > 0 else 0xE74C3C if total_pnl < 0 else 0x95A5A6,
        )

    async def risk_warning(self, rule: str, message: str) -> None:
        await self.send(f"Risk Warning: {rule}", message, color=0xF1C40F)

    async def system_error(self, error: str, context: str = "") -> None:
        await self.send("System Error",
                        f"**Error:** {error}\n**Context:** {context}" if context else f"**Error:** {error}",
                        color=0xE74C3C)
