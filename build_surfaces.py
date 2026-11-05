"""Build one implied-volatility surface per trading day.

    python build_surfaces.py                 # all days, NIFTY
    python build_surfaces.py --symbol BANKNIFTY
    python build_surfaces.py --symbol FINNIFTY       # reads data/processed/fo_FINNIFTY.*
    python build_surfaces.py --start 2020-01-01 --end 2020-12-31

Input : data/processed/fo_index.parquet (from download_bhavcopy.py)
Output: data/surfaces/<SYMBOL>_grid.npz      iv[day, tenor, moneyness] + mask
        data/surfaces/<SYMBOL>_slices.*      fitted SVI parameters per expiry per day
        data/surfaces/<SYMBOL>_diagnostics.* quality checks per day
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import config as C
from src.io_utils import load_table, save_table
from src.surface import build_day, load_rates


def _work(args):
    date, day, rates = args
    try:
        return date, build_day(day, date, rates), None
    except Exception as e:  # keep going; log the failure
        return date, None, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=C.SYMBOL)
    ap.add_argument("--input", help="table stem under data/processed (default fo_index)")
    ap.add_argument("--append", action="store_true", help="only build days after the last saved surface and add them")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    t0 = time.time()
    stem = a.input or ("fo_index" if a.symbol in ("NIFTY", "BANKNIFTY") else f"fo_{a.symbol}")
    df = load_table(C.PROCESSED / stem)
    df = df[df.symbol == a.symbol]
    if a.start:
        df = df[df.date >= a.start]
    if a.end:
        df = df[df.date <= a.end]
    out = C.SURFACES
    old = None
    if a.append and (out / f"{a.symbol}_grid.npz").exists():
        old = dict(np.load(out / f"{a.symbol}_grid.npz", allow_pickle=True))
        df = df[df.date > pd.Timestamp(old["dates"][-1])]
        if df.empty:
            print(f"{a.symbol}: already up to date ({old['dates'][-1]})")
            return
    rates = load_rates()
    jobs = [(d, g, rates) for d, g in df.groupby("date")]
    print(f"{a.symbol}: {len(jobs)} days to process with {a.workers} workers")

    grids, masks, dates, slices, diags, fails = [], [], [], [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (date, res, err) in enumerate(ex.map(_work, jobs, chunksize=8)):
            if res is None:
                fails.append(dict(date=date, error=err or "too few clean slices"))
            else:
                g, m, sl, dg = res
                grids.append(g); masks.append(m); dates.append(np.datetime64(date, "D"))
                slices.append(sl); diags.append(dg)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(jobs)} days  ({time.time() - t0:.0f}s)")

    out.mkdir(parents=True, exist_ok=True)
    if not dates:
        print("no new surfaces built")
        return
    dates, grids, masks = np.array(dates), np.array(grids), np.array(masks)
    sl, dg = pd.concat(slices, ignore_index=True), pd.DataFrame(diags)
    if old is not None:
        dates = np.concatenate([old["dates"].astype("datetime64[D]"), dates])
        grids, masks = np.concatenate([old["iv"], grids]), np.concatenate([old["mask"], masks])
        sl = pd.concat([load_table(out / f"{a.symbol}_slices"), sl], ignore_index=True)
        dg = pd.concat([load_table(out / f"{a.symbol}_diagnostics"), dg], ignore_index=True)
    np.savez_compressed(out / f"{a.symbol}_grid.npz", dates=dates, iv=grids,
                        mask=masks, tenors=np.array(C.TENORS_DAYS), moneyness=np.array(C.MONEYNESS))
    save_table(sl, out / f"{a.symbol}_slices")
    save_table(dg, out / f"{a.symbol}_diagnostics")
    if fails:
        pd.DataFrame(fails).to_csv(out / f"{a.symbol}_failed_days.csv", index=False)

    print(f"\nDone in {time.time() - t0:.0f}s: {len(dates)} surfaces, {len(fails)} days skipped")
    print(f"median SVI fit error: {dg.svi_rmse_median.median() * 100:.2f} vol pts | "
          f"grid coverage: {dg.coverage.mean():.0%} | "
          f"days needing calendar fixes: {(dg.calendar_fixes > 0).mean():.0%} | "
          f"slices failing butterfly check: {dg.butterfly_bad_slices.sum()}/{dg.n_slices.sum()}")


if __name__ == "__main__":
    main()
