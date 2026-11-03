"""Daily pipeline: fetch new NSE bhavcopies, add their surfaces, refresh the forecast.

    python daily_update.py            # run after ~19:00 IST on trading days
    python daily_update.py --no-forecast

Steps for every trading day after the last surface on file:
  1. download that day's F&O bhavcopy (same code as download_bhavcopy.py)
  2. build the IV surface (same code as build_surfaces.py) and append it to data/surfaces/
  3. save the day's NIFTY option chain to data/latest_chain.csv (strike explorer)
Then run predict.py (needs torch) to write the new forecast and log it for the live track record.
"""
import argparse
import io
import time
import zipfile
from datetime import date, timedelta

import numpy as np
import pandas as pd

import download_bhavcopy as dl
from src import config as C
from src.surface import build_day, load_rates

SURF = C.SURFACES


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
        if pq.exists():                              # keep a parquet copy in sync if one exists
            try:
                df.to_parquet(pq, index=False)
            except ImportError:
                pq.unlink()


def fetch_day(session, d: date):
    target = dl.RAW / dl.file_name(d)
    if not (target.exists() and target.stat().st_size > 1000):
        for host in dl.HOSTS:
            for p in dl.url_paths(d):
                try:
                    r = session.get(host + p, timeout=30)
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
    return dl.parse_udiff(raw) if "FinInstrmTp" in raw.columns.str.strip() else dl.parse_old(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-forecast", action="store_true")
    ap.add_argument("--until", default=date.today().isoformat())
    a = ap.parse_args()

    grid, slices, diag = read_tables()
    last = pd.Timestamp(grid["dates"][-1]).date()
    until = date.fromisoformat(a.until)
    rates = load_rates()
    session = dl.new_session()
    added = 0
    d = last + timedelta(days=1)
    while d <= until:
        if d.weekday() < 5:
            day = fetch_day(session, d)
            if day is None:
                print(f"{d}: no bhavcopy (holiday, or not published yet)")
            else:
                day = day[day.symbol == C.SYMBOL]
                res = build_day(day, pd.Timestamp(d), rates)
                if res is None:
                    print(f"{d}: bhavcopy found but too few clean quotes for a surface")
                else:
                    g, m, sl, dg = res
                    grid["dates"] = np.append(grid["dates"], np.datetime64(d, "D"))
                    grid["iv"] = np.concatenate([grid["iv"], g[None]])
                    grid["mask"] = np.concatenate([grid["mask"], m[None]])
                    slices = pd.concat([slices, sl], ignore_index=True)
                    diag = pd.concat([diag, pd.DataFrame([dg])], ignore_index=True)
                    chain = day[(day.instrument == "OPT") & (day.volume > 0)]
                    chain[["date", "expiry", "strike", "option_type", "close", "volume", "oi"]].to_csv(
                        C.DATA / "latest_chain.csv", index=False)
                    added += 1
                    print(f"{d}: surface added ({dg['n_slices']} expiries, {dg['n_points']} quotes)")
            time.sleep(1.0)
        d += timedelta(days=1)
    if added:
        write_tables(grid, slices, diag)
    print(f"{added} new day(s); surfaces now end {pd.Timestamp(grid['dates'][-1]).date()}")
    if added and not a.no_forecast:
        try:
            import predict
            predict.main()
        except ImportError as e:   # torch missing: surfaces are saved, forecast skipped
            print(f"forecast skipped ({e}); install torch: pip install -r requirements-pipeline.txt")


if __name__ == "__main__":
    main()
