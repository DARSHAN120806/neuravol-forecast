"""Phase 4: baselines the neural model has to beat.

  RW       random walk: surface at t+h = surface at t
  PCA-VAR  3-factor PCA of the surface, VAR(1) on the factor scores, h-step forecast

Walk-forward, expanding window, refit every 21 trading days. Errors are measured
only on grid cells backed by real quotes on the target day (mask).

    python evaluate_baselines.py                    # NIFTY, horizons 1/5/20, test from 2020
    python evaluate_baselines.py --test-start 2021-01-01
"""
import argparse
import json

import numpy as np
import pandas as pd

from src import config as C
from src.evaluation import diebold_mariano, error_table, load_grid, masked_rmse, walk_forward_splits


def fit_pca_var(X, n_factors=3):
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    V = Vt[:n_factors].T                       # loadings (cells x factors)
    F = (X - mu) @ V                           # factor scores
    Y, Z = F[1:], np.c_[np.ones(len(F) - 1), F[:-1]]
    B = np.linalg.lstsq(Z, Y, rcond=None)[0]   # VAR(1) with intercept
    explained = (S[:n_factors] ** 2).sum() / (S**2).sum()
    return mu, V, B, explained


def forecast_pca_var(x_now, model, h):
    mu, V, B, _ = model
    f = (x_now - mu) @ V
    for _ in range(h):
        f = B[0] + f @ B[1:]
    return mu + f @ V.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=C.SYMBOL)
    ap.add_argument("--test-start", default="2020-01-01")
    ap.add_argument("--horizons", default="1,5,20")
    ap.add_argument("--factors", type=int, default=3)
    a = ap.parse_args()

    dates, iv, mask, tenors, mon = load_grid(C.SURFACES / f"{a.symbol}_grid.npz")
    D, nT, nK = iv.shape
    X = iv.reshape(D, -1)
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    summary = {"symbol": a.symbol, "days": int(D), "first_day": str(dates[0].date()),
               "last_day": str(dates[-1].date()), "test_start": a.test_start, "results": {}}

    for h in map(int, a.horizons.split(",")):
        preds = {"RW": [], "PCA-VAR": []}
        tgt, msk, idx_out = [], [], []
        explained = []
        for train_end, test_idx in walk_forward_splits(dates, a.test_start):
            model = fit_pca_var(X[:train_end], a.factors)
            explained.append(model[3])
            for t in test_idx:
                if t + h >= D:
                    break
                preds["RW"].append(X[t])
                preds["PCA-VAR"].append(forecast_pca_var(X[t], model, h))
                tgt.append(X[t + h]); msk.append(mask[t + h].ravel()); idx_out.append(t)
        tgt, msk = np.array(tgt), np.array(msk)
        res = {}
        daily_loss = {}
        for name, p in preds.items():
            p = np.array(p)
            res[name] = error_table(p.reshape(-1, nT, nK), tgt.reshape(-1, nT, nK),
                                    msk.reshape(-1, nT, nK), tenors, mon)
            se = ((p - tgt) * 100) ** 2
            daily_loss[name] = np.array([se[i][msk[i]].mean() if msk[i].any() else np.nan for i in range(len(se))])
        stat, pval = diebold_mariano(daily_loss["PCA-VAR"], daily_loss["RW"], h)
        res["DM_PCAVAR_vs_RW"] = {"stat": stat, "p_value_pcavar_better": pval}
        res["n_forecasts"] = len(tgt)
        res["pca_variance_explained"] = float(np.mean(explained))
        test_dates = dates[np.array(idx_out)]
        calm = ~((test_dates >= "2020-02-15") & (test_dates <= "2020-06-30"))
        res["excl_covid_crash"] = {name: masked_rmse((np.array(p)[calm] - tgt[calm]) * 100, msk[calm])
                                   for name, p in preds.items()}
        summary["results"][f"h{h}"] = res
        np.savez_compressed(C.REPORTS / f"baseline_preds_h{h}.npz", idx=np.array(idx_out),
                            rw=np.array(preds["RW"]), pcavar=np.array(preds["PCA-VAR"]), target=tgt, mask=msk)

        print(f"\n=== horizon {h} day(s): {len(tgt)} out-of-sample forecasts ===")
        tab = pd.DataFrame({k: v for k, v in res.items() if k in preds}).round(2)
        print(tab.to_string())
        print(f"excluding Feb-Jun 2020: " + ", ".join(f"{k} {v:.2f}" for k, v in res["excl_covid_crash"].items()))
        print(f"Diebold-Mariano PCA-VAR vs RW: stat {stat:.2f}, p = {pval:.3f}  (p<0.05 => PCA-VAR significantly better)")

    with open(C.REPORTS / f"baselines_{a.symbol}.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nsaved reports/baselines_{a.symbol}.json")


if __name__ == "__main__":
    main()
