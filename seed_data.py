import asyncio
from src.feedback.trade_logger import TradeLogger

async def seed():
    tl = TradeLogger("data/trading.db")
    trades = [
        ("MNQU5","long",20100,20080,20150,2,"ict",0.78,"trending_bullish",20135,70,1.75),
        ("MNQU5","short",20200,20220,20150,1,"ict",0.72,"trending_bearish",20155,90,2.25),
        ("MNQU5","long",20050,20030,20100,2,"ict",0.81,"trending_bullish",20085,70,1.75),
        ("MNQU5","short",20180,20200,20130,1,"ict",0.66,"ranging",20200,-40,-1.0),
        ("MNQU5","long",20120,20100,20170,2,"ict",0.74,"trending_bullish",20155,70,1.75),
        ("MNQU5","long",20090,20070,20140,1,"ict",0.69,"ranging",20070,-40,-1.0),
        ("MNQU5","short",20250,20270,20200,2,"ict",0.82,"trending_bearish",20205,90,2.25),
        ("MNQU5","long",20110,20090,20160,1,"ict",0.77,"trending_bullish",20145,70,1.75),
        ("MNQU5","long",20075,20055,20125,2,"ict",0.85,"trending_bullish",20115,80,2.0),
        ("MNQU5","short",20220,20240,20170,2,"ict",0.79,"trending_bearish",20175,90,2.25),
    ]
    for sym,d,e,sl,tp,sz,st,c,r,ex,pnl,pr in trades:
        tid = await tl.log_trade_open(symbol=sym,direction=d,entry_price=e,stop_loss=sl,take_profit=tp,position_size=sz,strategy_name=st,signal_confidence=c,regime=r)
        await tl.log_trade_close(tid,exit_price=ex,pnl_dollars=pnl,pnl_r=pr,fees=1.24)
    await tl.log_daily_stats("2025-06-15",10000,10230,230,5,4,1,"trending_bullish")
    await tl.log_daily_stats("2025-06-16",10230,10460,230,5,3,2,"trending_bearish")
    print(f"Seeded {len(trades)} trades and 2 daily stats")

asyncio.run(seed())
