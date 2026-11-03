"""Everything the dashboard computes, kept free of Streamlit so it can be tested on its own."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.pricing import black76

ROOT = Path(__file__).resolve().parent
BLEND = "NeuraVol (NN v2 + PCA-VAR)"
BLEND_KEY = "NN v2 shrunk + PCA-VAR avg"      # name used in nn_v2_results.json
HORIZONS = [1, 5, 20]


def _read(stem, dates=("date",)):
    """Read data/surfaces tables saved either as .parquet (pyarrow installed) or .csv.gz."""
    stem = Path(stem)
    pq = stem.with_suffix(".parquet")
    if pq.exists():
        try:
            df = pd.read_parquet(pq)
            for c in dates:
                df[c] = pd.to_datetime(df[c])
            return df
        except ImportError:
            pass
    return pd.read_csv(stem.with_suffix(".csv.gz"), parse_dates=list(dates))


@dataclass
class Data:
    dates: pd.DatetimeIndex
    iv: np.ndarray          # (days, tenors, moneyness)
    mask: np.ndarray
    tenors: np.ndarray
    mon: np.ndarray
    diag: pd.DataFrame
    slices: pd.DataFrame
    forecast: pd.DataFrame
    forecast_log: pd.DataFrame
    errors: pd.DataFrame
    atm30: pd.DataFrame
    results: dict
    significance: dict
    bt_summary: pd.DataFrame
    bt_trades: pd.DataFrame
    chain: pd.DataFrame


def load(root=ROOT):
    root = Path(root)
    z = np.load(root / "data/surfaces/NIFTY_grid.npz")
    dates = pd.to_datetime(z["dates"])
    diag = _read(root / "data/surfaces/NIFTY_diagnostics")
    slices = _read(root / "data/surfaces/NIFTY_slices", dates=("date", "expiry"))
    fc = pd.read_csv(root / "data/forecasts/latest_forecast.csv", parse_dates=["as_of"])
    flog = pd.read_csv(root / "data/forecasts/forecast_log.csv", parse_dates=["as_of"])
    tr = root / "data/track_record"
    errors = pd.read_csv(tr / "daily_errors.csv.gz", parse_dates=["origin_date", "target_date"])
    atm30 = pd.read_csv(tr / "atm30_forecasts.csv.gz", parse_dates=["origin_date", "target_date"])
    sig = json.load(open(tr / "significance.json"))
    res = json.load(open(root / "reports/nn_v2/nn_v2_results.json"))
    bt = root / "reports/backtest"
    chain_p = root / "data/latest_chain.csv"
    chain = pd.read_csv(chain_p, parse_dates=["date", "expiry"]) if chain_p.exists() else pd.DataFrame()
    return Data(dates, z["iv"], z["mask"].astype(bool), z["tenors"].astype(int), z["moneyness"].astype(float),
                diag, slices, fc, flog, errors, atm30, res, sig,
                pd.read_csv(bt / "summary.csv"), pd.read_csv(bt / "trades.csv", parse_dates=["date", "exit_date"]), chain)


# ------------------------------------------------------------------ headline numbers
def idx(d: Data, tenor, m):
    return int(np.where(d.tenors == tenor)[0][0]), int(np.argmin(np.abs(d.mon - m)))


def kpis(d: Data, h):
    i = len(d.dates) - 1
    t30, k0 = idx(d, 30, 0.0)
    _, kp = idx(d, 30, -0.05)
    _, kc = idx(d, 30, 0.05)
    t7, _ = idx(d, 7, 0)
    t91, _ = idx(d, 91, 0)
    s = d.iv[i]
    prev = d.iv[i - 1]
    f = d.forecast[(d.forecast.horizon_days == h) & (d.forecast.tenor_days == 30) & (d.forecast.moneyness.abs() < 1e-9)]
    F = d.diag.set_index("date").F_near
    return {
        "as_of": d.dates[i].date(),
        "forward": float(F.iloc[-1]), "forward_chg": float(F.iloc[-1] / F.iloc[-2] - 1),
        "atm30": s[t30, k0] * 100, "atm30_chg": (s[t30, k0] - prev[t30, k0]) * 100,
        "skew30": (s[t30, kp] - s[t30, kc]) * 100,
        "term": (s[t91, k0] - s[t7, k0]) * 100,
        "fc_atm30": float(f.iv_forecast.iloc[0]) * 100 if len(f) else np.nan,
        "fc_chg": (float(f.iv_forecast.iloc[0]) - s[t30, k0]) * 100 if len(f) else np.nan,
    }


def grid_frame(d: Data, values, mask=None):
    df = pd.DataFrame(values * 100, index=[f"{t}d" for t in d.tenors], columns=[f"{m:+.1%}" for m in d.mon])
    if mask is not None:
        df = df.where(mask)
    return df


def surface_today(d: Data):
    return d.iv[-1], d.mask[-1]


def forecast_grid(d: Data, h, col="iv_forecast"):
    f = d.forecast[d.forecast.horizon_days == h].sort_values(["tenor_days", "moneyness"])
    return f[col].values.reshape(len(d.tenors), len(d.mon))


def model_rmse_by_tenor(d: Data, h, model=BLEND_KEY, normal_markets=True):
    """Out-of-sample RMSE (vol pts) per tenor from the walk-forward test, used for forecast bands.
    normal_markets=True rescales by (RMSE excluding the 2020 crash / overall RMSE) so one extreme
    year does not blow up every band; crash-level uncertainty is shown separately in the app."""
    tab = d.results["by_horizon"][str(h)]["table"][model]
    f = tab["excl. COVID crash"] / tab["overall"] if normal_markets else 1.0
    return {int(t): tab[f"{int(t)}d"] * f for t in d.tenors}


def smile(d: Data, tenor, h):
    i, _ = idx(d, tenor, 0)
    today = d.iv[-1, i] * 100
    fc = forecast_grid(d, h)[i] * 100
    band = 1.2816 * model_rmse_by_tenor(d, h)[int(tenor)]           # 80% band, assuming normal errors
    return pd.DataFrame({"moneyness": d.mon * 100, "today": today, "forecast": fc,
                         "lo": fc - band, "hi": fc + band, "quoted": d.mask[-1, i]})


def history(d: Data):
    t7, k0 = idx(d, 7, 0)
    t30, _ = idx(d, 30, 0)
    t91, _ = idx(d, 91, 0)
    _, kp = idx(d, 30, -0.05)
    _, kc = idx(d, 30, 0.05)
    return pd.DataFrame({"ATM 7d": d.iv[:, t7, k0] * 100, "ATM 30d": d.iv[:, t30, k0] * 100,
                         "ATM 91d": d.iv[:, t91, k0] * 100,
                         "Skew 30d (-5% minus +5%)": (d.iv[:, t30, kp] - d.iv[:, t30, kc]) * 100,
                         "Term slope (91d minus 7d)": (d.iv[:, t91, k0] - d.iv[:, t7, k0]) * 100}, index=d.dates)


# ------------------------------------------------------------------ strike explorer
def expiries(d: Data):
    s = d.slices[d.slices.date == d.dates[-1]].sort_values("days")
    return s


def svi_iv(row, K):
    k = np.log(K / row.F)
    x = k - row.m
    w = row.a + row.b * (row.rho * x + np.sqrt(x * x + row.sigma ** 2))
    return float(np.sqrt(max(w, 1e-10) / row["T"]))


def interp_grid(d: Data, grid, days, k):
    """Bilinear interpolation of a (tenor x moneyness) grid; clamps outside the grid."""
    t = np.clip(days, d.tenors.min(), d.tenors.max())
    kk = np.clip(k, d.mon.min(), d.mon.max())
    col = np.array([np.interp(kk, d.mon, grid[i]) for i in range(len(d.tenors))])
    return float(np.interp(t, d.tenors, col))


def strike_view(d: Data, expiry_row, K, is_call, h):
    r = expiry_row
    dfac = np.exp(-r.rate * r["T"])
    iv_now = svi_iv(r, K)
    k = np.log(K / r.F)
    today_grid, fc_grid = d.iv[-1], forecast_grid(d, h)
    # move the option's own smile by the forecast change of the grid at its maturity and moneyness
    days_after = max(r.days - h * 7 / 5, 1)
    change = interp_grid(d, fc_grid, days_after, k) - interp_grid(d, today_grid, days_after, k)
    iv_fc = max(iv_now + change, 0.01)
    band = 1.2816 * interp_rmse(d, h, r.days) / 100
    T_after = days_after / 365
    price_now = float(black76(r.F, K, r["T"], iv_now, is_call, dfac))
    price_fc = float(black76(r.F, K, T_after, iv_fc, is_call, np.exp(-r.rate * T_after)))
    lo = float(black76(r.F, K, r["T"], max(iv_now + change - band, 0.01), is_call, dfac))
    hi = float(black76(r.F, K, r["T"], iv_now + change + band, is_call, dfac))
    last = None
    if len(d.chain):
        c = d.chain[(d.chain.expiry == r.expiry) & (d.chain.strike == K) & (d.chain.option_type == ("CE" if is_call else "PE"))]
        if len(c):
            last = float(c.close.iloc[0])
    return dict(iv_now=iv_now * 100, iv_fc=iv_fc * 100, change=change * 100, model_price_now=price_now,
                price_if_forecast_today=(lo + hi) / 2, range_lo=lo, range_hi=hi, price_after_h=price_fc,
                last_traded=last, moneyness=k * 100, days=int(r.days), forward=float(r.F))


def interp_rmse(d: Data, h, days):
    r = model_rmse_by_tenor(d, h)
    t = np.array(sorted(r))
    return float(np.interp(days, t, [r[x] for x in t]))


# ------------------------------------------------------------------ track record
def pooled_rmse(errors, by=("horizon_days", "model")):
    e = errors.dropna(subset=["mse_volpts"]).copy()
    e["sse"] = e.mse_volpts * e.n_cells
    g = e.groupby(list(by))[["sse", "n_cells"]].sum()
    return np.sqrt(g.sse / g.n_cells)


def track_table(d: Data, exclude_covid=False):
    e = d.errors
    if exclude_covid:
        e = e[~e.origin_date.between("2020-02-15", "2020-06-30")]
    return pooled_rmse(e).unstack("model")


def track_by_year(d: Data, h):
    e = d.errors[d.errors.horizon_days == h].copy()
    e["year"] = e.origin_date.dt.year
    return pooled_rmse(e, by=("year", "model")).unstack("model")


def live_track_record(d: Data):
    """Forecasts logged by the daily pipeline, scored once their target day exists."""
    log = d.forecast_log
    if log.empty:
        return pd.DataFrame()
    pos = {dt: i for i, dt in enumerate(d.dates)}
    nK = len(d.mon)
    tix = {t: i for i, t in enumerate(d.tenors)}
    out = []
    for (as_of, h), g in log.groupby(["as_of", "horizon_days"]):
        i = pos.get(pd.Timestamp(as_of))
        if i is None or i + h >= len(d.dates):
            continue
        actual = d.iv[i + h].reshape(-1)
        m = d.mask[i + h].reshape(-1)
        cells = np.array([tix[t] * nK + int(np.argmin(np.abs(d.mon - k))) for t, k in zip(g.tenor_days, g.moneyness)])
        for name, col in (("NeuraVol", "iv_forecast"), ("No change", "iv_today")):
            e = (g[col].values - actual[cells])[m[cells]] * 100
            out.append(dict(as_of=as_of, horizon_days=h, target_date=d.dates[i + h], model=name,
                            rmse=float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan))
    return pd.DataFrame(out)
