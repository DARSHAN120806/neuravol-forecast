"""Extract extra index symbols from the raw NSE bhavcopy zips, resumably.

    python scripts/extract_symbols.py --symbols FINNIFTY --since 2021-01-01 --budget 150

Each run processes zips until the time budget is used, writes one part file to
data/processed/parts_<SYMBOLS>/ and records finished zips in done.txt, so the next run continues where this
one stopped. When it prints "ALL DONE", merge the parts with --merge.
"""
import argparse
import io
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import download_bhavcopy as B  # noqa: E402

MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def zip_date(name: str):
    m = re.search(r"_(\d{8})_F_", name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    m = re.search(r"fo(\d{2})([A-Z]{3})(\d{4})bhav", name)
    if m:
        return datetime(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1))).date()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="FINNIFTY")
    ap.add_argument("--since", default="2021-01-01")
    ap.add_argument("--budget", type=float, default=150, help="seconds per run")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    syms = a.symbols.split(",")
    out = ROOT / "data" / "processed" / f"parts_{'_'.join(syms)}"
    out.mkdir(parents=True, exist_ok=True)
    done_f = out / "done.txt"

    if a.merge:
        parts = sorted(out.glob("part_*.csv.gz"))
        df = pd.concat([pd.read_csv(p, parse_dates=["date", "expiry"]) for p in parts], ignore_index=True)
        df = df.drop_duplicates().sort_values(["date", "symbol", "instrument", "expiry", "strike"])
        dest = ROOT / "data" / "processed" / f"fo_{'_'.join(syms)}.csv.gz"
        df.to_csv(dest, index=False)
        print(f"merged {len(df):,} rows, {df.date.nunique()} days ({df.date.min():%Y-%m-%d} -> {df.date.max():%Y-%m-%d}) -> {dest}")
        print(df.groupby(["symbol", "instrument"]).size().to_string())
        return

    B.SYMBOLS = syms
    since = datetime.fromisoformat(a.since).date()
    done = set(done_f.read_text().split()) if done_f.exists() else set()
    todo = [z for z in sorted(B.RAW.glob("*.zip")) if z.name not in done and (zip_date(z.name) or since) >= since]
    t0, frames, finished = time.time(), [], []
    for z in todo:
        if time.time() - t0 > a.budget:
            break
        try:
            with zipfile.ZipFile(z) as zf:
                name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
                raw = pd.read_csv(io.BytesIO(zf.read(name)), low_memory=False)
            cols = raw.columns.str.strip()
            frames.append(B.parse_udiff(raw) if "FinInstrmTp" in cols else B.parse_old(raw))
        except Exception as e:
            print(f"!! {z.name}: {e}")
        finished.append(z.name)
    if frames:
        part = pd.concat(frames, ignore_index=True)[B.COMMON]
        n = len(list(out.glob("part_*.csv.gz")))
        part.to_csv(out / f"part_{n:04d}.csv.gz", index=False)
    with open(done_f, "a") as f:
        f.write("".join(x + "\n" for x in finished))
    left = len(todo) - len(finished)
    rows = sum(len(f) for f in frames)
    print(f"processed {len(finished)} zips ({rows:,} rows) in {time.time() - t0:.0f}s; {left} left")
    if left == 0:
        print("ALL DONE")


if __name__ == "__main__":
    main()
