"""Live NIFTY option-chain collector (bid/ask) using the Upstox Analytics Token.

Saves a full snapshot of every NIFTY expiry (bid, ask, LTP, volume, OI, Upstox IV) during market hours.
These snapshots give clean mid-price surfaces instead of end-of-day closing prices.

Setup (once):
  1. Upstox account -> https://account.upstox.com/developer/apps -> create an app -> "Analytics" tab ->
     Generate Token (free, read-only, valid 1 year).
  2. Save it in  secrets/upstox_token.txt  (this folder is git-ignored)  or set env UPSTOX_ANALYTICS_TOKEN.

Run:
  python collector/collect_chain.py --once            # one snapshot now (test)
  python collector/collect_chain.py --loop --every 15 # snapshot every 15 min until market close

Output: data/live/YYYY-MM-DD.csv (one row per strike per expiry per snapshot; appended)
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
API = "https://api.upstox.com/v2"
UNDERLYING = "NSE_INDEX|Nifty 50"
IST = timezone(timedelta(hours=5, minutes=30))
OPEN, CLOSE = (9, 15), (15, 30)
MAX_DAYS = 400


def token():
    t = os.environ.get("UPSTOX_ANALYTICS_TOKEN")
    if not t:
        f = ROOT / "secrets" / "upstox_token.txt"
        if f.exists():
            t = f.read_text().strip()
    if not t:
        sys.exit("No Upstox token: put it in secrets/upstox_token.txt or set UPSTOX_ANALYTICS_TOKEN")
    return t


def get(path, params, tok, tries=3):
    for i in range(tries):
        r = requests.get(f"{API}{path}", params=params, timeout=20,
                         headers={"Accept": "application/json", "Authorization": f"Bearer {tok}"})
        if r.status_code == 200:
            return r.json().get("data") or []
        if r.status_code == 401:
            sys.exit("Upstox rejected the token (401). Generate a new Analytics Token and save it again.")
        time.sleep(2 * (i + 1))
    r.raise_for_status()


def expiries(tok, today):
    data = get("/option/contract", {"instrument_key": UNDERLYING}, tok)
    exps = sorted({d["expiry"][:10] for d in data if d.get("expiry")})
    return [e for e in exps if 0 <= (pd.Timestamp(e).date() - today).days <= MAX_DAYS]


def flatten(chain, ts):
    """Upstox put/call option-chain JSON -> long table (one row per strike per side)."""
    rows = []
    for item in chain:
        for side, key in (("CE", "call_options"), ("PE", "put_options")):
            o = item.get(key) or {}
            md, gk = o.get("market_data") or {}, o.get("option_greeks") or {}
            rows.append(dict(
                ts=ts, expiry=item.get("expiry"), strike=item.get("strike_price"), option_type=side,
                spot=item.get("underlying_spot_price"), pcr=item.get("pcr"),
                ltp=md.get("ltp"), bid=md.get("bid_price"), ask=md.get("ask_price"),
                bid_qty=md.get("bid_qty"), ask_qty=md.get("ask_qty"), volume=md.get("volume"), oi=md.get("oi"),
                close_price=md.get("close_price"), iv_upstox=gk.get("iv"), delta=gk.get("delta"),
                instrument_key=o.get("instrument_key")))
    return rows


def snapshot(tok):
    now = datetime.now(IST)
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for e in expiries(tok, now.date()):
        rows += flatten(get("/option/chain", {"instrument_key": UNDERLYING, "expiry_date": e}, tok), ts)
        time.sleep(0.25)                                  # stay well under Upstox rate limits
    df = pd.DataFrame(rows)
    LIVE.mkdir(parents=True, exist_ok=True)
    out = LIVE / f"{now:%Y-%m-%d}.csv"
    df.to_csv(out, mode="a", header=not out.exists(), index=False)
    quoted = ((df.bid > 0) & (df.ask > 0)).sum() if len(df) else 0
    print(f"{ts} IST: saved {len(df)} rows ({quoted} with live bid/ask) across {df.expiry.nunique() if len(df) else 0} expiries")
    return df


def market_open(now):
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return OPEN <= t <= CLOSE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="take one snapshot now, even outside market hours")
    ap.add_argument("--loop", action="store_true", help="snapshot every --every minutes until market close")
    ap.add_argument("--every", type=int, default=15)
    a = ap.parse_args()
    tok = token()
    if a.once or not a.loop:
        snapshot(tok)
        return
    while True:
        now = datetime.now(IST)
        if now.weekday() >= 5 or (now.hour, now.minute) > CLOSE:
            print("market closed for the day; exiting")
            return
        if market_open(now):
            try:
                snapshot(tok)
            except Exception as e:                       # keep looping on network hiccups
                print(f"{now:%H:%M} snapshot failed: {e}")
        time.sleep(a.every * 60 - (time.time() % 60))


if __name__ == "__main__":
    main()
