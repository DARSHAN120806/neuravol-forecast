"""
NSE F&O Bhavcopy bulk downloader + cleaner for the Neural Volatility Surface Forecaster.

Usage (run on your own computer, inside the project folder):
    pip install requests pandas pyarrow
    python download_bhavcopy.py --start 2016-01-01 --end 2026-10-30
    python download_bhavcopy.py --build-only        # just rebuild the parquet from downloaded zips

What it does
  1. Downloads one bhavcopy zip per trading day into data/raw/
       - before 2024-07-08 : old format  foDDMONYYYYbhav.csv.zip
       - from   2024-07-08 : UDiFF format BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip
     Weekends are skipped; holidays simply return 404 and are logged.
     Already-downloaded files are skipped, so you can stop and re-run any time.
  2. Parses both formats into ONE common schema, keeps only index options + futures
     for the chosen symbols, and writes data/processed/fo_index.parquet
"""
import argparse, io, random, time, zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

UDIFF_START = date(2024, 7, 8)
SYMBOLS = ["NIFTY", "BANKNIFTY"]
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
LOG = ROOT / "data" / "download_log.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/all-reports-derivatives",
}
HOSTS = ["https://nsearchives.nseindia.com", "https://archives.nseindia.com"]


def file_name(d: date) -> str:
    if d >= UDIFF_START:
        return f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    return f"fo{d:%d}{d:%b}{d:%Y}bhav.csv.zip".replace(d.strftime("%b"), d.strftime("%b").upper())


def url_paths(d: date) -> list[str]:
    if d >= UDIFF_START:
        return [f"/content/fo/{file_name(d)}"]
    mon = d.strftime("%b").upper()
    return [f"/content/historical/DERIVATIVES/{d:%Y}/{mon}/{file_name(d)}"]


def new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    try:  # NSE sets cookies on the homepage; archive requests work better with them
        s.get("https://www.nseindia.com", timeout=15)
    except requests.RequestException:
        pass
    return s


def download(start: date, end: date):
    RAW.mkdir(parents=True, exist_ok=True)
    s = new_session()
    log = []
    d = start
    while d <= end:
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue
        target = RAW / file_name(d)
        if target.exists() and target.stat().st_size > 1000:
            d += timedelta(days=1)
            continue
        status = "missing"
        for attempt in range(3):
            got = False
            for host in HOSTS:
                for p in url_paths(d):
                    try:
                        r = s.get(host + p, timeout=30)
                    except requests.RequestException:
                        continue
                    if r.status_code == 200 and r.content[:2] == b"PK":
                        target.write_bytes(r.content)
                        status, got = "ok", True
                        break
                    if r.status_code == 404:
                        status = "404 (holiday?)"
                    elif r.status_code in (401, 403):
                        status = f"{r.status_code} blocked"
                if got:
                    break
            if got or status.startswith("404"):
                break
            time.sleep(5 * (attempt + 1))
            s = new_session()  # refresh cookies after a block
        print(f"{d}  {status}")
        log.append({"date": d, "status": status})
        time.sleep(random.uniform(0.6, 1.5))  # be polite, avoid getting blocked
        d += timedelta(days=1)
    if log:
        df = pd.DataFrame(log)
        df.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


# ---------- parsing ----------
COMMON = ["date", "symbol", "instrument", "expiry", "strike", "option_type",
          "open", "high", "low", "close", "settle", "volume", "oi", "chg_oi", "underlying"]


def parse_udiff(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    df = df[df["FinInstrmTp"].isin(["IDO", "IDF"]) & df["TckrSymb"].isin(SYMBOLS)]
    return pd.DataFrame({
        "date": pd.to_datetime(df["TradDt"]),
        "symbol": df["TckrSymb"],
        "instrument": df["FinInstrmTp"].map({"IDO": "OPT", "IDF": "FUT"}),
        "expiry": pd.to_datetime(df["XpryDt"]),
        "strike": df["StrkPric"],
        "option_type": df["OptnTp"],
        "open": df["OpnPric"], "high": df["HghPric"], "low": df["LwPric"],
        "close": df["ClsPric"], "settle": df["SttlmPric"],
        "volume": df["TtlTradgVol"], "oi": df["OpnIntrst"], "chg_oi": df["ChngInOpnIntrst"],
        "underlying": df["UndrlygPric"],
    })


def parse_old(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    df = df[df["INSTRUMENT"].isin(["OPTIDX", "FUTIDX"]) & df["SYMBOL"].str.strip().isin(SYMBOLS)]
    opt = df["OPTION_TYP"].str.strip()
    return pd.DataFrame({
        "date": pd.to_datetime(df["TIMESTAMP"], format="%d-%b-%Y"),
        "symbol": df["SYMBOL"].str.strip(),
        "instrument": df["INSTRUMENT"].map({"OPTIDX": "OPT", "FUTIDX": "FUT"}),
        "expiry": pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y"),
        "strike": df["STRIKE_PR"].where(opt != "XX"),
        "option_type": opt.where(opt != "XX"),
        "open": df["OPEN"], "high": df["HIGH"], "low": df["LOW"],
        "close": df["CLOSE"], "settle": df["SETTLE_PR"],
        "volume": df["CONTRACTS"], "oi": df["OPEN_INT"], "chg_oi": df["CHG_IN_OI"],
        "underlying": float("nan"),  # old format has no spot column -> join NIFTY index history later
    })


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for z in sorted(RAW.glob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
                raw = pd.read_csv(io.BytesIO(zf.read(name)), low_memory=False)
            frames.append(parse_udiff(raw) if "FinInstrmTp" in raw.columns.str.strip() else parse_old(raw))
        except Exception as e:
            print(f"!! could not parse {z.name}: {e}")
    if not frames:
        print("No zips found in", RAW)
        return
    df = pd.concat(frames, ignore_index=True)[COMMON].sort_values(["date", "symbol", "instrument", "expiry", "strike"])
    path = OUT / "fo_index.parquet"
    try:
        df.to_parquet(path, index=False)
    except ImportError:  # pyarrow not installed -> fall back to compressed CSV
        path = OUT / "fo_index.csv.gz"
        df.to_csv(path, index=False)
    print(f"\nSaved {len(df):,} rows, {df['date'].nunique()} trading days "
          f"({df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}) to {path}")
    print(df.groupby(["symbol", "instrument"]).size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    if not a.build_only:
        download(datetime.fromisoformat(a.start).date(), datetime.fromisoformat(a.end).date())
    build()
