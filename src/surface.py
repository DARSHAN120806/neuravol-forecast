"""Turn one day's option chain into an arbitrage-checked implied-volatility surface.

Steps per expiry slice:
  1. keep contracts that actually traded
  2. forward F from put-call parity (fallback: futures close for that expiry)
  3. keep OTM options only (puts below F, calls above), compute Black-76 IV
  4. fit a raw-SVI smile to total variance w = iv^2 * T
Then across slices:
  5. enforce calendar no-arbitrage (w non-decreasing in T at each moneyness)
  6. interpolate total variance in T onto the fixed (tenor x moneyness) grid
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import config as C
from .pricing import implied_vol

_RATES = pd.Series(
    [r for _, r in C.REPO_RATE_HISTORY],
    index=pd.to_datetime([d for d, _ in C.REPO_RATE_HISTORY]),
)


def load_rates() -> pd.Series:
    """Use data/rates.csv (date,rate as decimal) if present, else the repo-rate proxy."""
    f = C.DATA / "rates.csv"
    if f.exists():
        s = pd.read_csv(f, parse_dates=["date"]).set_index("date")["rate"].sort_index()
        return s
    return _RATES


def rate_on(date, rates: pd.Series) -> float:
    s = rates[rates.index <= pd.Timestamp(date)]
    return float(s.iloc[-1]) if len(s) else float(rates.iloc[0])


# ---------------------------------------------------------------- SVI ---------
def svi_w(k, p):
    a, b, rho, m, sig = p
    x = k - m
    return a + b * (rho * x + np.sqrt(x * x + sig * sig))


def svi_derivs(k, p):
    a, b, rho, m, sig = p
    x = k - m
    r = np.sqrt(x * x + sig * sig)
    w = a + b * (rho * x + r)
    w1 = b * (rho + x / r)
    w2 = b * sig * sig / r**3
    return w, w1, w2


def durrleman_g(k, p):
    """Butterfly-arbitrage density condition: must be >= 0 everywhere."""
    w, w1, w2 = svi_derivs(k, p)
    w = np.maximum(w, 1e-12)
    return (1 - k * w1 / (2 * w)) ** 2 - (w1**2 / 4) * (1 / w + 0.25) + w2 / 2


def fit_svi(k, iv, T, weights=None):
    """Least-squares raw-SVI fit; residuals in vol units. Returns (params, rmse_vol)."""
    w = iv**2 * T
    wt = np.ones_like(k) if weights is None else weights / weights.mean()

    def resid(p):
        wf = np.maximum(svi_w(k, p), 1e-10)
        r = (np.sqrt(wf / T) - iv) * np.sqrt(wt)
        a, b, rho, m, sig = p
        pen = max(0.0, -(a + b * sig * np.sqrt(1 - rho * rho))) * 100  # min variance >= 0
        return np.append(r, pen)

    lb = [-0.5 * w.max(), 1e-6, -0.999, -0.5, 1e-4]
    ub = [w.max(), 5.0, 0.999, 0.5, 1.5]
    best = None
    i0 = np.argmin(np.abs(k))
    for rho0, m0, s0 in [(-0.5, 0.0, 0.1), (-0.3, 0.02, 0.05), (-0.7, -0.02, 0.2), (0.0, 0.0, 0.1)]:
        b0 = max(1e-3, (w.max() - w.min()) / max(0.1, np.ptp(k)))
        a0 = float(np.clip(w[i0] - b0 * s0, lb[0], ub[0] - 1e-9))
        x0 = np.clip([a0, b0, rho0, m0, s0], np.array(lb) + 1e-9, np.array(ub) - 1e-9)
        try:
            sol = least_squares(resid, x0, bounds=(lb, ub), method="trf", max_nfev=400)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        return None, np.nan
    fitted = np.sqrt(np.maximum(svi_w(k, best.x), 1e-10) / T)
    rmse = float(np.sqrt(np.mean((fitted - iv) ** 2)))
    return best.x, rmse


def slice_w(K, row):
    """SVI total variance inside the quoted range; beyond it, extend linearly from the
    edge with the edge slope capped, so sparse wings cannot explode."""
    p = row[["a", "b", "rho", "m", "sigma"]].values.astype(float)
    lo, hi = row.k_min - C.EXTRAP_MARGIN, row.k_max + C.EXTRAP_MARGIN
    w = svi_w(K, p)
    for edge, side in ((lo, K < lo), (hi, K > hi)):
        if side.any():
            w0, w1, _ = svi_derivs(np.array([edge]), p)
            cap = C.MAX_WING_SLOPE * row["T"]
            w[side] = w0[0] + np.clip(w1[0], -cap, cap) * (K[side] - edge)
    return w


# ------------------------------------------------------------ one day -------
def _forward(sl: pd.DataFrame, df: float, fut_close: float | None):
    piv = sl.pivot_table(index="strike", columns="option_type", values="close", aggfunc="first")
    if {"CE", "PE"} <= set(piv.columns):
        both = piv.dropna(subset=["CE", "PE"])
        if len(both):
            diff = (both["CE"] - both["PE"]).abs().nsmallest(3).index
            Fs = diff.values + (both.loc[diff, "CE"] - both.loc[diff, "PE"]).values / df
            F = float(np.median(Fs))
            if fut_close is None or abs(F / fut_close - 1) < 0.02:
                return F, "parity"
    if fut_close is not None:
        return float(fut_close), "futures"
    return None, None


def build_day(day: pd.DataFrame, date, rates: pd.Series):
    """Return (grid_iv[T,K], mask[T,K], slices DataFrame, diagnostics dict) or None."""
    date = pd.Timestamp(date)
    r = rate_on(date, rates)
    opt = day[(day.instrument == "OPT") & (day.volume > 0) & (day.close > 0)].copy()
    fut = day[day.instrument == "FUT"].set_index("expiry")["close"]
    opt["days"] = (opt.expiry - date).dt.days
    opt = opt[(opt.days >= C.MIN_DAYS_TO_EXPIRY) & (opt.days <= C.MAX_DAYS_TO_EXPIRY)]

    rows = []
    for exp, sl in opt.groupby("expiry"):
        T = sl.days.iloc[0] / 365.0
        dfac = np.exp(-r * T)
        F, fsrc = _forward(sl, dfac, float(fut[exp]) if exp in fut.index else None)
        if F is None:
            continue
        q = sl[((sl.option_type == "PE") & (sl.strike < F)) | ((sl.option_type == "CE") & (sl.strike >= F))]
        q = q[q.close >= C.MIN_PRICE]
        if len(q) < C.MIN_POINTS_PER_SLICE:
            continue
        iv = implied_vol(q.close.values, F, q.strike.values, T, (q.option_type == "CE").values, dfac)
        k = np.log(q.strike.values / F)
        ok = ~np.isnan(iv) & (iv > C.IV_BOUNDS[0]) & (iv < C.IV_BOUNDS[1]) & (np.abs(k) <= C.MAX_ABS_MONEYNESS)
        k, iv, vol = k[ok], iv[ok], q.volume.values[ok]
        if len(k) < C.MIN_POINTS_PER_SLICE:
            continue
        p, rmse = fit_svi(k, iv, T, weights=np.sqrt(vol.astype(float)) + 1)
        if p is None:
            continue
        kk = np.linspace(k.min(), k.max(), 200)
        rows.append(dict(
            date=date, expiry=exp, days=int(sl.days.iloc[0]), T=T, F=F, fwd_src=fsrc, rate=r,
            a=p[0], b=p[1], rho=p[2], m=p[3], sigma=p[4], n=len(k), k_min=k.min(), k_max=k.max(),
            svi_rmse=rmse, atm_iv=float(np.sqrt(max(svi_w(0.0, p), 1e-12) / T)),
            butterfly_ok=bool(durrleman_g(kk, p).min() >= -1e-6),
        ))
    if not rows:
        return None
    sl = pd.DataFrame(rows)
    n_fit = len(sl)
    sl = sl[sl.svi_rmse <= C.MAX_REL_SVI_RMSE * sl.atm_iv]          # drop badly fitted smiles
    if len(sl) < 2:
        return None
    sl = sl.sort_values("T").reset_index(drop=True)

    K = np.array(C.MONEYNESS)
    Ts = sl["T"].values
    W, S = [], []
    for _, row in sl.iterrows():
        W.append(slice_w(K, row))
        S.append((K >= row.k_min - C.EXTRAP_MARGIN) & (K <= row.k_max + C.EXTRAP_MARGIN))
    W, S = np.maximum(np.array(W), 1e-8), np.array(S)
    # calendar no-arbitrage: w must not fall as T grows. Where a shorter slice is only
    # extrapolated at that strike, pull it down to the longer slice; else push later ones up.
    Wfix = W.copy()
    for i in range(len(Ts) - 2, -1, -1):
        Wfix[i] = np.where(~S[i], np.minimum(Wfix[i], Wfix[i + 1]), Wfix[i])
    Wfix = np.maximum.accumulate(Wfix, axis=0)
    cal_fixes = int((np.abs(Wfix - W) > 1e-9).sum())

    grid = np.full((len(C.TENORS_DAYS), len(K)), np.nan)
    mask = np.zeros_like(grid, dtype=bool)
    for i, td in enumerate(C.TENORS_DAYS):
        tg = td / 365.0
        if tg <= Ts[0]:
            w = Wfix[0] * tg / Ts[0]
            nb = [0]
        elif tg >= Ts[-1]:
            w = Wfix[-1] * tg / Ts[-1]
            nb = [len(Ts) - 1]
        else:
            j = np.searchsorted(Ts, tg)
            a = (tg - Ts[j - 1]) / (Ts[j] - Ts[j - 1])
            w = (1 - a) * Wfix[j - 1] + a * Wfix[j]
            nb = [j - 1, j]
        grid[i] = np.sqrt(w / tg)
        kmin = sl.loc[nb, "k_min"].min() - 0.01
        kmax = sl.loc[nb, "k_max"].max() + 0.01
        in_t = (tg >= 0.5 * Ts[0]) and (tg <= 1.5 * Ts[-1])
        mask[i] = in_t & (K >= kmin) & (K <= kmax)

    diag = dict(date=date, n_slices=len(sl), n_slices_dropped=n_fit - len(sl), n_points=int(sl.n.sum()),
                svi_rmse_median=float(sl.svi_rmse.median()), calendar_fixes=cal_fixes,
                butterfly_bad_slices=int((~sl.butterfly_ok).sum()), coverage=float(mask.mean()),
                F_near=float(sl.F.iloc[0]), parity_share=float((sl.fwd_src == "parity").mean()))
    return grid, mask, sl, diag
