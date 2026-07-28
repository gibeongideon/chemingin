"""Systematic M5 breakout SCALPER on XAUUSD (FTMO demo) — 1:RR broker-side TP.

Runs ON the VPS against localhost:18814 (FTMO-Demo 1514025597). Separate magic
(360545) and 0.01 lot so it cannot affect the champion book on the same account.
Honest expectation: intraday gold is net-negative vs the $0.45 spread (proven), so
this is a live experiment, not an edge.

Signal (on the just-CLOSED M5 bar): breakout of the prior N-bar range in the
direction of the bar's body. Enter at market with SL = k*ATR, TP = RR*SL
(broker-managed exit — survives even if monitoring lags). One position at a time.
"""
import os, sys, time, json
from datetime import datetime, timezone, timedelta
from mt5linux import MetaTrader5

# Portable: SCALP_PORT selects the bridge (18812 local / 18814 VPS FTMO), SCALP_LOG
# the log path. Defaults target the LOCAL terminal.
PORT=int(os.environ.get("SCALP_PORT","18812")); SYMBOL="XAUUSD"; MAGIC=360545; LOT=0.01
N=6; ATR_N=14; SL_ATR=1.0; RR=float(sys.argv[1]) if len(sys.argv)>1 else 2.0
MINUTES=int(sys.argv[2]) if len(sys.argv)>2 else 60
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG=os.environ.get("SCALP_LOG", os.path.join(_ROOT,"data","v5_runs","scalp.log"))
os.makedirs(os.path.dirname(LOG), exist_ok=True)

def log(msg):
    line=f"{datetime.now(timezone.utc):%H:%M:%S} {msg}"
    print(line, flush=True)
    open(LOG,"a").write(line+"\n")

m=MetaTrader5(host="localhost",port=PORT); m.initialize()
m.symbol_select(SYMBOL,True)
a=m.account_info()
log(f"=== SCALPER START RR=1:{RR:g} {MINUTES}min  acct {a.login} eq {a.equity:.0f} ===")
TF=m.TIMEFRAME_M5
start_eq=a.equity
def our_pos():
    return [p for p in (m.positions_get(symbol=SYMBOL) or []) if p.magic==MAGIC]
def atr(rates,n):
    trs=[]
    for i in range(1,len(rates)):
        h,l,pc=rates[i][2],rates[i][3],rates[i-1][4]
        trs.append(max(h-l,abs(h-pc),abs(l-pc)))
    return sum(trs[-n:])/n
def send(direction, entry, sl, tp):
    req=dict(action=m.TRADE_ACTION_DEAL, symbol=SYMBOL, volume=LOT,
             type=m.ORDER_TYPE_BUY if direction>0 else m.ORDER_TYPE_SELL,
             price=entry, sl=round(sl,2), tp=round(tp,2), deviation=30, magic=MAGIC,
             comment="scalp", type_time=m.ORDER_TIME_GTC)
    for fill in (m.ORDER_FILLING_IOC, m.ORDER_FILLING_FOK):
        req["type_filling"]=fill
        r=m.order_send(req)
        rc=getattr(r,"retcode",None)
        if rc==10009: return rc
    return rc

t_end=time.time()+MINUTES*60
last_bar=None; ntr=0
while time.time()<t_end:
    try:
        raw=m.copy_rates_from_pos(SYMBOL,TF,0,N+ATR_N+2)
        if raw is None or len(raw)<N+ATR_N: time.sleep(10); continue
        # materialize remote numpy rows -> plain floats (np.* must not cross the bridge)
        r=[(float(x[0]),float(x[1]),float(x[2]),float(x[3]),float(x[4])) for x in raw]
        closed=r[-2]; bt=closed[0]
        pos=our_pos()
        if pos:
            p=pos[0]
            log(f"  holding {('LONG' if p.type==0 else 'SHORT')} {float(p.volume):.2f} @ {float(p.price_open):.2f} pnl {float(p.profit):+.2f}")
        elif bt!=last_bar:
            last_bar=bt
            hi=max(x[2] for x in r[-N-2:-2]); lo=min(x[3] for x in r[-N-2:-2])
            o,c=closed[1],closed[4]; A=atr(r[:-1],ATR_N)
            t=m.symbol_info_tick(SYMBOL)
            ask,bid=float(t.ask),float(t.bid)
            sig=0
            if c>hi and c>o: sig=1
            elif c<lo and c<o: sig=-1
            if sig and A>0:
                entry=ask if sig>0 else bid
                sl=entry-sig*SL_ATR*A; tp=entry+sig*RR*SL_ATR*A
                rc=send(sig,entry,sl,tp); ntr+=1
                log(f"  SIGNAL {'LONG' if sig>0 else 'SHORT'} brk@{c:.2f} ATR{A:.2f} entry~{entry:.2f} sl {sl:.2f} tp {tp:.2f} rc={rc}")
            else:
                log(f"  bar {datetime.fromtimestamp(bt,timezone.utc):%H:%M} no breakout (c{c:.2f} hi{hi:.2f} lo{lo:.2f})")
    except Exception as e:
        log(f"  ERR {type(e).__name__}: {e}")
    time.sleep(20)

# summary
eq=m.account_info().equity
frm=datetime.now(timezone.utc)-timedelta(minutes=MINUTES+5); to=datetime.now(timezone.utc)+timedelta(minutes=5)
deals=[d for d in (m.history_deals_get(frm,to) or []) if d.magic==MAGIC and d.entry==1]
realized=sum(d.profit for d in deals)
op=our_pos(); floating=sum(p.profit for p in op)
log(f"=== SCALPER END: {ntr} entries, {len(deals)} closed, realized {realized:+.2f}, "
    f"{len(op)} still open (floating {floating:+.2f}), equity {start_eq:.0f}->{eq:.0f} ({eq-start_eq:+.2f}) ===")
m.shutdown()
