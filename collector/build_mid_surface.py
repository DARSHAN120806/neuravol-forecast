"""Build a volatility surface from a live snapshot's MID prices (bid+ask)/2 instead of closing prices.

    python collector/build_mid_surface.py                  # every day in data/live/ not yet built
    python collector/build_mid_surface.py --time 15:25     # which snapshot to use (closest to this time)

Output: data/surfaces_mid/NIFTY_grid_mid.npz (+ slices, diagnostics), same layout as data/surfaces/,
so the existing models and dashboard code can use it directly once enough days have been collected.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src.surface import build_day, load_rates  # noqa: E402

LIVE = ROOT / "data" / "live"
OUT = ROOT / "data" / "surfaces_mid"
MAX_REL_SPREAD = 0.25     # ignore quotes whose bid-ask spread is wider than 25% of mid


def snapshot_to_day(snap: pd.DataFrame):
    """Live snapshot -> the common schema build_day() expects, using mid prices of clean two-sided quotes."""
    s = snap.copy()
    s["mid"] = (s.bid + s.ask) / 2
    ok = (s.bid > 0) & (s.ask > 0) & (s.ask >= s.bid) & ((s.ask - s.bid) / s.mid <= MAX_REL_SPREAD)
    s = s[ok]
    day = pd.DataFrame({
        "date": pd.to_datetime(s.ts).dt.normalize(), "symbol": C.SYMBOL, "instrument": "OPT",
        "expiry": pd.to_datetime(s.expiry), "strike": s.strike.astype(float), "option_type": s.option_type,
        "close": s.mid, "settle": s.mid, "volume": 1, "oi": s.oi, "underlying": s.spot,
    })
    return day


def pick_snapshot(df, hhmm):
    ts = pd.to_datetime(df.ts)
    target = ts.dt.normalize() + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
    best = ts.iloc[(ts - target).abs().values.argmin()]
    return df[ts == best], best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", default="15:25")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    gpath = OUT / f"{C.SYMBOL}_grid_mid.npz"
    if gpath.exists():
        z = np.load(gpath)
        grid = {k: z[k] for k in z.files}
        done = set(pd.to_datetime(grid["dates"]).date)
    else:
        grid, done = None, set()
    rates = load_rates()
    slices, diags = [], []
    for f in sorted(LIVE.glob("*.csv")):
        day_date = pd.Timestamp(f.stem).date()
        if day_date in done:
            continue
        snap, when = pick_snapshot(pd.read_csv(f), a.time)
        day = snapshot_to_day(snap)
        res = build_day(day, pd.Timestamp(day_date), rates)
        if res is None:
            print(f"{day_date}: not enough clean quotes")
            continue
        g, m, sl, dg = res
        dg["snapshot"] = str(when)
        if grid is None:
            grid = dict(dates=np.array([np.datetime64(day_date, "D")]), iv=g[None], mask=m[None],
                        tenors=np.array(C.TENORS_DAYS), moneyness=np.array(C.MONEYNESS))
        else:
            grid["dates"] = np.append(grid["dates"], np.datetime64(day_date, "D"))
            grid["iv"] = np.concatenate([grid["iv"], g[None]])
            grid["mask"] = np.concatenate([grid["mask"], m[None]])
        slices.append(sl); diags.append(dg)
        print(f"{day_date}: mid surface from {when:%H:%M} snapshot, {dg['n_points']} quotes, "
              f"SVI fit error {dg['svi_rmse_median'] * 100:.2f} vol pts")
    if not slices:
        print("nothing new to build")
        return
    np.savez_compressed(gpath, **grid)
    for name, new in ((f"{C.SYMBOL}_slices_mid.csv.gz", pd.concat(slices)), (f"{C.SYMBOL}_diagnostics_mid.csv.gz", pd.DataFrame(diags))):
        p = OUT / name
        (pd.concat([pd.read_csv(p), new]) if p.exists() else new).to_csv(p, index=False)


if __name__ == "__main__":
    main()
