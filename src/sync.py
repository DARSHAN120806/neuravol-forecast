"""Data sync engine shared by the admin panel and the daily scheduler.

Brings data/surfaces up to a target date:
  * for every missing weekday, download the NSE F&O bhavcopy and build that day's surface
  * if NSE is unreachable (often the case from cloud servers), fall back to an Upstox option-chain snapshot
    for the most recent completed session (mid prices of live bid/ask quotes)
  * refresh the forecast (neural model if PyTorch is installed, otherwise the PCA-VAR factor model)
  * append a row to data/admin/sync_log.csv
"""
from __future__ import annotations

import io
import json
import os
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .surface import build_day, load_rates

ROOT = C.ROOT
SURF = C.SURFACES
ADMIN = C.DATA / "admin"
SYNC_LOG = ADMIN / "sync_log.csv"
IST = timezone(timedelta(hours=5, minutes=30))
SYNC_FILES = [  # what a sync changes; saved to GitHub afterwards
    "data/surfaces/NIFTY_grid.npz", "data/surfaces/NIFTY_slices.csv.gz", "data/surfaces/NIFTY_diagnostics.csv.gz",
    "data/forecasts/latest_forecast.csv", "data/forecasts/forecast_log.csv", "data/latest_chain.csv",
    "data/admin/sync_log.csv",
]


def now_ist():
    return datetime.now(IST)


# ------------------------------------------------------------------ tables
def read_tables():
    z = np.load(SURF / f"{C.SYMBOL}_grid.npz")
    grid = {k: z[k] for k in z.files}

    def rd(stem, cols):
        pq = (SURF / stem).with_suffix(".parquet")
        if pq.exists():
            try:
                return pd.read_parquet(pq)
            except ImportError:
                pass
        return pd.read_csv((SURF / stem).with_suffix(".csv.gz"), parse_dates=cols)

    return grid, rd(f"{C.SYMBOL}_slices", ["date", "expiry"]), rd(f"{C.SYMBOL}_diagnostics", ["date"])


def write_tables(grid, slices, diag):
    np.savez_compressed(SURF / f"{C.SYMBOL}_grid.npz", **grid)
    for name, df in ((f"{C.SYMBOL}_slices", slices), (f"{C.SYMBOL}_diagnostics", diag)):
        df.to_csv((SURF / name).with_suffix(".csv.gz"), index=False)
        pq = (SURF / name).with_suffix(".parquet")
        if pq.exists():
            try:
                df.to_parquet(pq, index=False)
            except ImportError:
                pq.unlink()


def last_date():
    z = np.load(SURF / f"{C.SYMBOL}_grid.npz")
    return pd.Timestamp(z["dates"][-1]).date()


def missing_weekdays(start_after: date, until: date):
    d, out = start_after + timedelta(days=1), []
    while d <= until:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def sessions_behind(today: date | None = None):
    """Weekdays after the last stored session up to the latest completed session (holidays count too)."""
    t = latest_completed_session(now_ist()) if today is None else today
    return len(missing_weekdays(last_date(), t))


def latest_completed_session(now: datetime) -> date:
    d = now.date()
    if d.weekday() < 5 and (now.hour, now.minute) >= (15, 30):
        return d
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


# ------------------------------------------------------------------ sources
def fetch_nse(day: date, session=None):
    """Common-schema DataFrame for one day from the NSE bhavcopy, or None if not available."""
    import download_bhavcopy as dl
    session = session or dl.new_session()
    target = dl.RAW / dl.file_name(day)
    if not (target.exists() and target.stat().st_size > 1000):
        for host in dl.HOSTS:
            for p in dl.url_paths(day):
                try:
                    r = session.get(host + p, timeout=25)
                except Exception:
                    continue
                if r.status_code == 200 and r.content[:2] == b"PK":
                    dl.RAW.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(r.content)
                    break
            if target.exists():
                break
    if not target.exists():
        return None
    with zipfile.ZipFile(target) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        raw = pd.read_csv(io.BytesIO(zf.read(name)), low_memory=False)
    df = dl.parse_udiff(raw) if "FinInstrmTp" in raw.columns.str.strip() else dl.parse_old(raw)
    return df[df.symbol == C.SYMBOL]


def nse_reachable():
    import download_bhavcopy as dl
    try:
        r = dl.new_session().get("https://www.nseindia.com", timeout=15)
        return r.status_code < 400, f"HTTP {r.status_code}"
    except Exception as e:
        return False, type(e).__name__


def upstox_token():
    t = os.environ.get("UPSTOX_ANALYTICS_TOKEN")
    if not t:
        f = ROOT / "secrets" / "upstox_token.txt"
        if f.exists():
            t = f.read_text().strip()
    return t or None


def fetch_upstox_snapshot(token: str):
    """Full NIFTY option chain from Upstox right now (long table), using the collector's code."""
    from collector import collect_chain as cc
    now = now_ist()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    try:
        for e in cc.expiries(token, now.date()):
            rows += cc.flatten(cc.get("/option/chain", {"instrument_key": cc.UNDERLYING, "expiry_date": e}, token), ts)
            time.sleep(0.2)
    except SystemExit as e:              # the collector exits on a bad token; turn that into an error here
        raise RuntimeError(str(e))
    return pd.DataFrame(rows)


def upstox_reachable(token):
    if not token:
        return False, "no token configured"
    from collector import collect_chain as cc
    try:
        ex = cc.expiries(token, now_ist().date())
        return bool(ex), f"{len(ex)} expiries listed"
    except SystemExit as e:
        return False, str(e)
    except Exception as e:
        return False, type(e).__name__


def upstox_day(token, session_day: date):
    from collector.build_mid_surface import snapshot_to_day
    snap = fetch_upstox_snapshot(token)
    if snap.empty:
        return None, snap
    day = snapshot_to_day(snap)
    day["date"] = pd.Timestamp(session_day)
    return day, snap


# ------------------------------------------------------------------ forecast
def refresh_forecast(log):
    """Neural forecast if PyTorch is available, otherwise PCA-VAR. Returns the method used."""
    try:
        import predict
        predict.main()
        return "neural model + PCA-VAR"
    except ImportError:
        pass
    from .evaluation import load_grid
    from .model import forecast_table, pca_var_forecast
    dates, iv, mask, tenors, mon = load_grid(SURF / f"{C.SYMBOL}_grid.npz")
    D = len(dates)
    ivf, mf = iv.reshape(D, -1), mask.reshape(D, -1)
    H = [1, 5, 20]
    pv = pca_var_forecast(ivf, D - 1, H)
    tab = forecast_table(dates[-1], H, tenors, mon, ivf[-1], mf[-1], {"nn": pv * np.nan, "pcavar": pv, "blend": pv})
    out = C.DATA / "forecasts"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "latest_forecast.csv", index=False)
    logf = out / "forecast_log.csv"
    if logf.exists():
        old = pd.read_csv(logf)
        old = old[old.as_of != str(dates[-1].date())]
        tab = pd.concat([old, tab.astype({"as_of": str})], ignore_index=True)
    tab.to_csv(logf, index=False)
    log("PyTorch not installed here: forecast made with the PCA-VAR factor model only.")
    return "PCA-VAR (neural model unavailable)"


# ------------------------------------------------------------------ main entry
@dataclass
class SyncResult:
    added: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    source: dict = field(default_factory=dict)
    forecast: str = ""
    ok: bool = True
    message: str = ""


def run_sync(until: date | None = None, source: str = "auto", update_forecast: bool = True,
             log=print, fetchers=None) -> SyncResult:
    """source: 'auto' (NSE, then Upstox for the latest session), 'nse' or 'upstox'.
    fetchers: optional dict to override 'nse' / 'upstox' functions (used in tests)."""
    t0 = time.time()
    fetchers = fetchers or {}
    f_nse = fetchers.get("nse", fetch_nse)
    f_up = fetchers.get("upstox", upstox_day)
    now = now_ist()
    until = until or latest_completed_session(now)
    res = SyncResult()
    grid, slices, diag = read_tables()
    last = pd.Timestamp(grid["dates"][-1]).date()
    todo = missing_weekdays(last, until)
    log(f"Last stored session: {last:%d %b %Y}. Checking {len(todo)} weekday(s) up to {until:%d %b %Y}.")
    if not todo:
        res.message = "Already up to date."
        log(res.message)
    rates = load_rates()
    token = upstox_token()
    latest_session = latest_completed_session(now)
    session = None
    for d in todo:
        day, src = None, None
        if source in ("auto", "nse"):
            try:
                if session is None:
                    import download_bhavcopy as dl
                    session = dl.new_session()
                day = f_nse(d, session) if f_nse is fetch_nse else f_nse(d)
                src = "NSE bhavcopy" if day is not None else None
            except Exception as e:
                log(f"{d:%d %b}: NSE error ({type(e).__name__}).")
        if day is None and source in ("auto", "upstox") and d == latest_session:
            if token or "upstox" in fetchers:
                try:
                    day, snap = f_up(token, d)
                    src = "Upstox live quotes" if day is not None else None
                    if day is not None and not snap.empty:
                        snap.to_csv(C.DATA / "latest_chain_upstox.csv", index=False)
                except Exception as e:
                    log(f"{d:%d %b}: Upstox error ({e}).")
            else:
                log(f"{d:%d %b}: Upstox fallback unavailable (no token configured).")
        if day is None:
            res.skipped.append(d)
            log(f"{d:%d %b %Y}: no data (market holiday, not yet published, or source blocked).")
            continue
        out = build_day(day, pd.Timestamp(d), rates)
        if out is None:
            res.skipped.append(d)
            log(f"{d:%d %b %Y}: data found but too few clean quotes for a surface.")
            continue
        g, m, sl, dg = out
        grid["dates"] = np.append(grid["dates"], np.datetime64(d, "D"))
        grid["iv"] = np.concatenate([grid["iv"], g[None]])
        grid["mask"] = np.concatenate([grid["mask"], m[None]])
        slices = pd.concat([slices, sl], ignore_index=True)
        diag = pd.concat([diag, pd.DataFrame([dg])], ignore_index=True)
        opt = day[day.instrument == "OPT"]
        if "volume" in opt:
            opt = opt[opt.volume > 0]
        cols = [c for c in ["date", "expiry", "strike", "option_type", "close", "volume", "oi"] if c in opt]
        opt[cols].to_csv(C.DATA / "latest_chain.csv", index=False)
        res.added.append(d)
        res.source[str(d)] = src
        log(f"{d:%d %b %Y}: added from {src} ({dg['n_slices']} expiries, {dg['n_points']} quotes).")
    if res.added:
        write_tables(grid, slices, diag)
        if update_forecast:
            try:
                res.forecast = refresh_forecast(log)
                log(f"Forecast updated ({res.forecast}).")
            except Exception as e:
                res.forecast = f"failed: {e}"
                log(f"Forecast update failed: {e}")
    res.message = res.message or (f"Added {len(res.added)} session(s)." if res.added else "No new sessions added.")
    write_sync_log(res, source, time.time() - t0)
    return res


def write_sync_log(res: SyncResult, source, secs, pushed=""):
    ADMIN.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([dict(time_ist=now_ist().strftime("%Y-%m-%d %H:%M:%S"), source_setting=source,
                             added=";".join(str(x) for x in res.added), skipped=";".join(str(x) for x in res.skipped),
                             sources=json.dumps(res.source), forecast=res.forecast, seconds=round(secs, 1),
                             github=pushed, message=res.message)])
    row.to_csv(SYNC_LOG, mode="a", header=not SYNC_LOG.exists(), index=False)


def mark_last_push(status: str):
    if SYNC_LOG.exists():
        log = pd.read_csv(SYNC_LOG)
        log.loc[log.index[-1], "github"] = status
        log.to_csv(SYNC_LOG, index=False)


def remove_last_session():
    """Undo: drop the most recent session from the stored surfaces."""
    grid, slices, diag = read_tables()
    if len(grid["dates"]) < 2:
        raise RuntimeError("nothing to remove")
    d = pd.Timestamp(grid["dates"][-1])
    for k in ("dates", "iv", "mask"):
        grid[k] = grid[k][:-1]
    slices = slices[pd.to_datetime(slices.date) != d]
    diag = diag[pd.to_datetime(diag.date) != d]
    write_tables(grid, slices, diag)
    return d.date()


def data_health(n=30):
    """Recent sessions with simple quality flags, plus weekdays missing from the history."""
    _, _, diag = read_tables()
    diag = diag.sort_values("date").tail(n).copy()
    diag["fit_error_pts"] = diag.svi_rmse_median * 100
    flags = []
    for r in diag.itertuples():
        f = []
        if r.fit_error_pts > 1.0:
            f.append("high fit error")
        if r.n_points < 100:
            f.append("few quotes")
        if r.coverage < 0.6:
            f.append("low coverage")
        flags.append(", ".join(f) or "OK")
    diag["status"] = flags
    z = np.load(SURF / f"{C.SYMBOL}_grid.npz")
    dts = pd.to_datetime(z["dates"])
    have = set(dts.date)
    start = (dts[-1] - pd.Timedelta(days=90)).date()
    gaps = [d for d in missing_weekdays(start, dts[-1].date()) if d not in have]
    return diag[["date", "n_slices", "n_points", "fit_error_pts", "coverage", "status"]], gaps
