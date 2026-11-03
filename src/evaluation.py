"""Walk-forward evaluation helpers shared by baselines and neural models."""
import numpy as np
import pandas as pd
from scipy.stats import norm


def load_grid(path):
    z = np.load(path, allow_pickle=False)
    dates = pd.to_datetime(z["dates"])
    iv, mask = z["iv"], z["mask"]
    return dates, iv, mask, z["tenors"], z["moneyness"]


def walk_forward_splits(dates, test_start, min_train=500, refit_every=21):
    """Yield (train_idx_end, test_indices): expanding window, refit every N days."""
    first = int(np.searchsorted(dates.values, np.datetime64(test_start)))
    first = max(first, min_train)
    for s in range(first, len(dates), refit_every):
        yield s, np.arange(s, min(s + refit_every, len(dates)))


def masked_rmse(err, mask):
    e = err[mask]
    return float(np.sqrt(np.mean(e**2))) if e.size else np.nan


def diebold_mariano(loss_a, loss_b, h):
    """DM test that model A has lower loss than B. Newey-West variance with h-1 lags.
    Returns (stat, one-sided p-value); negative stat => A better."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 30:
        return np.nan, np.nan
    dm = d.mean()
    gamma0 = np.var(d, ddof=0)
    lags = max(h - 1, 0)
    var = gamma0 + 2 * sum((1 - l / (lags + 1)) * np.cov(d[l:], d[:-l], ddof=0)[0, 1] for l in range(1, lags + 1))
    stat = dm / np.sqrt(var / n)
    return float(stat), float(norm.cdf(stat))


def error_table(pred, target, mask, tenors, moneyness):
    """RMSE (vol points) overall and by region."""
    err = (pred - target) * 100
    K = np.asarray(moneyness)
    out = {"overall": masked_rmse(err, mask)}
    for i, t in enumerate(tenors):
        out[f"tenor_{t}d"] = masked_rmse(err[:, i, :], mask[:, i, :])
    regions = {"put_wing": K <= -0.05, "near_atm": np.abs(K) <= 0.025, "call_wing": K >= 0.05}
    for name, sel in regions.items():
        out[name] = masked_rmse(err[:, :, sel], mask[:, :, sel])
    return out
