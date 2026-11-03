"""Compact out-of-sample track record for the dashboard, from the v2 walk-forward predictions.

    python build_track_record.py      # reads reports/nn_v2/nn_v2_test_predictions.npz
"""
import json

import numpy as np
import pandas as pd

from src import config as C
from src.evaluation import load_grid

MODELS = {"No change": "No_change", "PCA-VAR": "PCA-VAR", "NN v2": "NN_v2_(shrunk)",
          "NeuraVol (NN v2 + PCA-VAR)": "NN_v2_shrunk_+_PCA-VAR_avg"}


def main():
    dates, iv, mask, tenors, mon = load_grid(C.SURFACES / "NIFTY_grid.npz")
    z = np.load(C.REPORTS / "nn_v2" / "nn_v2_test_predictions.npz")
    o = z["origins"]
    nK = len(mon)
    c30 = int(np.where(tenors == 30)[0][0]) * nK + int(np.argmin(np.abs(mon)))
    err_rows, atm_rows = [], []
    for h in (1, 5, 20):
        tgt, msk = z[f"h{h}_target"], z[f"h{h}_mask"].astype(bool)
        for name, key in MODELS.items():
            p = z[f"h{h}_{key}"]
            se = ((p - tgt) * 100) ** 2
            mse = np.array([se[i][msk[i]].mean() if msk[i].any() else np.nan for i in range(len(se))])
            err_rows.append(pd.DataFrame({"origin_date": dates[o], "target_date": dates[o + h], "horizon_days": h,
                                          "model": name, "mse_volpts": mse, "n_cells": msk.sum(1)}))
        d = {"origin_date": dates[o], "target_date": dates[o + h], "horizon_days": h, "actual": tgt[:, c30]}
        for name, key in MODELS.items():
            d[name] = z[f"h{h}_{key}"][:, c30]
        atm_rows.append(pd.DataFrame(d))
    out = C.DATA / "track_record"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(err_rows).to_csv(out / "daily_errors.csv.gz", index=False)
    pd.concat(atm_rows).to_csv(out / "atm30_forecasts.csv.gz", index=False)
    res = json.load(open(C.REPORTS / "nn_v2" / "nn_v2_results.json"))
    dm = {h: r["dm"] for h, r in res["by_horizon"].items()}
    json.dump(dm, open(out / "significance.json", "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
