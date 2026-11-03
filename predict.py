"""Forecast the surface 1/5/20 trading days ahead from the latest day in data/surfaces (CPU).

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python predict.py

Writes data/forecasts/latest_forecast.csv and appends to data/forecasts/forecast_log.csv
(the log is what the dashboard's live track record is scored from).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.evaluation import load_grid
from src.io_utils import load_table
from src.model import Forecaster, forecast_table

MODEL = C.ROOT / "models" / "neuravol_v2.pt"
OUT = C.DATA / "forecasts"


def main():
    dates, iv, mask, tenors, mon = load_grid(C.SURFACES / f"{C.SYMBOL}_grid.npz")
    D = len(dates)
    ivf, maskf = iv.reshape(D, -1), mask.reshape(D, -1)
    diag = load_table(C.SURFACES / f"{C.SYMBOL}_diagnostics", date_cols=("date",)).set_index("date").reindex(dates)
    fwd = diag["F_near"].ffill().values
    fc = Forecaster(MODEL)
    t = D - 1
    preds = fc.forecast(dates, ivf, fwd, t)
    tab = forecast_table(dates[t], fc.h, tenors, mon, ivf[t], maskf[t], preds)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "latest_forecast.csv", index=False)
    log = OUT / "forecast_log.csv"
    if log.exists():
        old = pd.read_csv(log)
        old = old[old.as_of != str(dates[t].date())]
        tab = pd.concat([old, tab.astype({"as_of": str})], ignore_index=True)
    tab.to_csv(log, index=False)
    atm = tab[(tab.moneyness == 0) & (tab.tenor_days == 30)]
    print(f"forecast as of {dates[t].date()}:")
    print(atm[["horizon_days", "iv_today", "iv_nn", "iv_pcavar", "iv_forecast"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
