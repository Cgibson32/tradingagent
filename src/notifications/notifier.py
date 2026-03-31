"""Discord webhook notification system.

Sends alerts for trade entries/exits, daily summaries, risk warnings,
and system errors via Discord webhooks.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import structlog

logger = structlog.get_logger(__name__)


class DiscordNotifier:
    """Sends trading notifications via Discord webhook."""

    def __init__(self, webhook_url: str, enabled: bool = True):
        self._webhook_url = webhook_url
        self._enabled = enabled and bool(webhook_url)

    async def send(self, title: str, message: str, color: int = 0x3498DB) -> None:
        """Send an embed message to Discord."""
        if not self._enabled:
            return

        payload = {
            "embeds": [{
                "title": title,
                "description": message,
                "color": color,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {"text": "NQ Trading Agent"},
            }]
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self._webhook_url, json=payload)
                if response.status_code != 204:
                    logger.warning("discord.send_failed", status=response.status_code)
        except Exception as e:
            logger.warning("discord.error", error=str(e))

    async def trade_entry(
        self, symbol: str, direction: str, entry: float, sl: float,
        tp: float, size: int, confidence: float, strategy: str,
    ) -> None:
        """Notify on trade entry."""
        color = 0x2ECC71 if direction == "long" else 0xE74C3C
        risk_pts = abs(entry - sl)

        await self.send(
            f"{'🟢' if direction == 'long' else '🔴'} {direction.upper()} {symbol}",
            (
                f"**Entry**: {entry:.2f}\n"
                f"**Stop Loss**: {sl:.2f} ({risk_pts:.2f} pts)\n"
                f"**Take Profit**: {tp:.2f}\n"
                f"**Size**: {size} contracts\n"
                f"**Confidence**: {confidence:.0%}\n"
                f"**Strategy**: {strategy}"
            ),
            color=color,
        )

    async def trade_exit(
        self, symbol: str, direction: str, entry: float, exit_price: float,
        pnl_dollars: float, pnl_r: float, reason: str,
    ) -> None:
        """Notify on trade exit."""
        color = 0x2ECC71 if pnl_dollars >= 0 else 0xE74C3C
        emoji = "✅" if pnl_dollars >= 0 else "❌"

        await self.send(
            f"{emoji} CLOSED {direction.upper()} {symbol}",
            (
                f"**Entry**: {entry:.2f} → **Exit**: {exit_price:.2f}\n"
                f"**P&L**: ${pnl_dollars:+,.2f} ({pnl_r:+.2f}R)\n"
                f"**Reason**: {reason}"
            ),
            color=color,
        )

    async def daily_summary(
        self, date: str, trades: int, wins: int, pnl: float, equity: float, win_rate: float,
    ) -> None:
        """Send end-of-day summary."""
        color = 0x2ECC71 if pnl >= 0 else 0xE74C3C
        await self.send(
            f"📊 Daily Summary — {date}",
            (
                f"**Trades**: {trades} ({wins}W / {trades - wins}L)\n"
                f"**Win Rate**: {win_rate:.0%}\n"
                f"**Daily P&L**: ${pnl:+,.2f}\n"
                f"**Account Equity**: ${equity:,.2f}"
            ),
            color=color,
        )

    async def risk_warning(self, warning: str) -> None:
        """Send a risk warning alert."""
        await self.send("⚠️ Risk Warning", warning, color=0xF39C12)

    async def system_error(self, error: str) -> None:
        """Send a system error alert."""
        await self.send("🚨 System Error", error, color=0xE74C3C)
