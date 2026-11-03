"""Phase 7: does the volatility forecast make money? Delta-hedged straddle backtest on real NIFTY option closes.

Rules (decided before looking at results):
  * Every `hold` trading days (non-overlapping), read the model's forecast of the change in ATM implied vol
    for the maturity being traded (forecast surface minus today's surface, interpolated in tenor).
  * forecast change > +threshold  -> BUY one ATM straddle;  < -threshold -> SELL one;  otherwise stay flat.
  * Contract: the listed expiry closest to the target maturity, strike nearest the forward, priced at the
    actual NSE closing price. Exit at the actual close `hold` days later (SVI model price if it did not trade).
  * Delta-hedged every day with the forward (futures) using Black-76 delta from that day's fitted smile.
  * Costs per side: options `OPT_COST` x premium, futures `FUT_COST` x notional on every hedge change.
  * P&L reported in vol points (P&L / vega at entry) so trades of different sizes are comparable,
    and in rupees for one straddle on one NIFTY unit (multiply by the lot size for a real lot).

Benchmarks on the same days: always short, always long, PCA-VAR signal, and an oracle that knows the
actual future vol change (an upper bound, not achievable).

    python backtest.py
"""
import json

import numpy as np
import pandas as pd
from scipy.special import ndtr

from src import config as C
from src.evaluation import load_grid
from src.io_utils import load_table
from src.pricing import black76, vega76

OPT_COST = 0.003     # 0.3% of premium per option per side (spread + STT + exchange charges, conservative for ATM NIFTY)
FUT_COST = 0.0002    # 0.02% of notional per futures hedge adjustment
THRESHOLD = 0.25     # vol points
SETUPS = {5: (20, 50, 30), 20: (45, 80, 60)}   # hold days -> (min days to expiry, max, target)
SIGNALS = {"NN v2 + PCA-VAR blend": "NN_v2_shrunk_+_PCA-VAR_avg", "NN v2": "NN_v2_(shrunk)", "PCA-VAR": "PCA-VAR"}
REPORT = C.REPORTS / "backtest"


def slice_iv(row, K):
    k = np.log(K / row.F)
    x = k - row.m
    w = row.a + row.b * (row.rho * x + np.sqrt(x * x + row.sigma ** 2))
    return float(np.sqrt(max(w, 1e-8) / row["T"]))


def atm_forecast_change(pred, today, days):
    """Forecast change (vol pts) of ATM IV at maturity `days`, interpolating the grid's tenor rows."""
    tenors = np.array(C.TENORS_DAYS, float)
    k0 = int(np.argmin(np.abs(np.array(C.MONEYNESS))))
    nK = len(C.MONEYNESS)
    p = np.interp(days, tenors, pred.reshape(-1, nK)[:, k0])
    t = np.interp(days, tenors, today.reshape(-1, nK)[:, k0])
    return (p - t) * 100


def run():
    dates, iv, mask, _, _ = load_grid(C.SURFACES / "NIFTY_grid.npz")
    z = np.load(C.REPORTS / "nn_v2" / "nn_v2_test_predictions.npz")
    origins = z["origins"]
    slices = load_table(C.SURFACES / "NIFTY_slices")
    slices = slices.set_index(["date", "expiry"]).sort_index()
    opts = load_table(C.PROCESSED / "fo_index")
    opts = opts[(opts.symbol == C.SYMBOL) & (opts.instrument == "OPT")]
    px = opts.set_index(["date", "expiry", "strike", "option_type"])["close"].sort_index()
    px = px[~px.index.duplicated()]
    by_day = {d: g.reset_index(level=0, drop=True) for d, g in slices.groupby(level=0)}

    trades = []
    for hold, (dmin, dmax, dtarget) in SETUPS.items():
        for j in range(0, len(origins), hold):
            t = origins[j]
            if t + hold >= len(dates):
                break
            d0, d1 = dates[t], dates[t + hold]
            if d0 not in by_day:
                continue
            sl0 = by_day[d0]
            cand = sl0[(sl0.days >= dmin) & (sl0.days <= dmax)]
            if cand.empty:
                continue
            exp = (cand.days - dtarget).abs().idxmin()
            row0 = cand.loc[exp]
            strikes = [k for k in sorted(set(opts.loc[(opts.date == d0) & (opts.expiry == exp), "strike"]))]
            both = [k for k in strikes if (d0, exp, k, "CE") in px.index and (d0, exp, k, "PE") in px.index]
            if not both:
                continue
            K = min(both, key=lambda k: abs(k - row0.F))
            entry = px[(d0, exp, K, "CE")] + px[(d0, exp, K, "PE")]
            iv0 = slice_iv(row0, K)
            df0 = np.exp(-row0.rate * row0["T"])
            vega = 2 * vega76(row0.F, K, row0["T"], iv0, df0) / 100      # rupees per vol point
            # daily delta hedge from d0 to d1 (position = +1 straddle; sign applied later)
            hedge_pnl, hedge_cost, prev_delta, prevF, fallback = 0.0, 0.0, 0.0, None, False
            for s in range(t, t + hold + 1):
                ds = dates[s]
                rs = by_day.get(ds)
                if rs is None or exp not in rs.index:
                    continue
                r = rs.loc[exp]
                if prevF is not None:
                    hedge_pnl += -prev_delta * (r.F - prevF)
                if s < t + hold:
                    sig = slice_iv(r, K)
                    d1_ = (np.log(r.F / K) + 0.5 * sig ** 2 * r["T"]) / (sig * np.sqrt(r["T"]))
                    delta = np.exp(-r.rate * r["T"]) * (2 * ndtr(d1_) - 1)
                    hedge_cost += abs(delta - prev_delta) * r.F * FUT_COST
                    prev_delta = delta
                prevF = r.F
            hedge_cost += abs(prev_delta) * (prevF or row0.F) * FUT_COST              # unwind
            if (d1, exp, K, "CE") in px.index and (d1, exp, K, "PE") in px.index:
                exit_ = px[(d1, exp, K, "CE")] + px[(d1, exp, K, "PE")]
            elif d1 in by_day and exp in by_day[d1].index:
                r1 = by_day[d1].loc[exp]
                s1 = slice_iv(r1, K)
                dfa = np.exp(-r1.rate * r1["T"])
                exit_ = float(black76(r1.F, K, r1["T"], s1, True, dfa) + black76(r1.F, K, r1["T"], s1, False, dfa))
                fallback = True
            else:
                continue
            gross_long = (exit_ - entry) + hedge_pnl
            costs = OPT_COST * (entry + exit_) + hedge_cost
            sigs = {name: atm_forecast_change(z[f"h{hold}_{key}"][j], z[f"h{hold}_No_change"][j], row0.days)
                    for name, key in SIGNALS.items()}
            actual = atm_forecast_change(z[f"h{hold}_target"][j], z[f"h{hold}_No_change"][j], row0.days)
            trades.append(dict(hold=hold, date=d0, exit_date=d1, expiry=exp, days=int(row0.days), strike=K,
                               F=row0.F, entry=entry, exit=exit_, iv0=iv0, vega=vega, gross_long=gross_long,
                               costs=costs, exit_model_priced=fallback, actual_dvol=actual,
                               **{f"sig_{k}": v for k, v in sigs.items()}))
    return pd.DataFrame(trades)


def evaluate(tr, cost_mult=1.0, threshold=THRESHOLD):
    rows, curves = [], {}
    for hold, g in tr.groupby("hold"):
        per_year = 252 / hold
        strategies = {f"{k} signal": np.where(g[f"sig_{k}"] > threshold, 1, np.where(g[f"sig_{k}"] < -threshold, -1, 0))
                      for k in SIGNALS}
        strategies["Always short straddle"] = -np.ones(len(g), int)
        strategies["Always long straddle"] = np.ones(len(g), int)
        strategies["Oracle (knows future vol)"] = np.where(g.actual_dvol > threshold, 1, np.where(g.actual_dvol < -threshold, -1, 0))
        for name, pos in strategies.items():
            pnl_rs = pos * g.gross_long.values - np.abs(pos) * g.costs.values * cost_mult
            pnl_vp = pnl_rs / g.vega.values
            active = pos != 0
            eq = np.cumsum(pnl_vp)
            dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
            sharpe = pnl_vp.mean() / pnl_vp.std() * np.sqrt(per_year) if pnl_vp.std() > 0 else np.nan
            rows.append({"hold_days": hold, "strategy": name, "trades": int(active.sum()), "of_periods": len(g),
                         "hit_rate": float((pnl_vp[active] > 0).mean()) if active.any() else np.nan,
                         "avg_pnl_volpts_per_trade": float(pnl_vp[active].mean()) if active.any() else np.nan,
                         "total_pnl_volpts": float(pnl_vp.sum()), "total_pnl_rs_per_unit": float(pnl_rs.sum()),
                         "sharpe_annualised": float(sharpe), "max_drawdown_volpts": dd})
            curves[(hold, name)] = pd.Series(eq, index=g.date.values)
    return pd.DataFrame(rows), curves


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORT.mkdir(parents=True, exist_ok=True)
    tr = run()
    tr.to_csv(REPORT / "trades.csv", index=False)
    print(f"{len(tr)} trade periods built | exit priced from model (contract untraded): {tr.exit_model_priced.mean():.1%}")
    res, curves = evaluate(tr)
    res.to_csv(REPORT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))

    sens = []
    for cm in (0.0, 1.0, 2.0):
        for th in (0.0, 0.25, 0.5, 1.0):
            r, _ = evaluate(tr, cm, th)
            r = r[r.strategy == "NN v2 + PCA-VAR blend signal"]
            for _, x in r.iterrows():
                sens.append({"hold_days": x.hold_days, "cost_x": cm, "threshold": th, "trades": x.trades,
                             "sharpe": x.sharpe_annualised, "total_volpts": x.total_pnl_volpts})
    sens = pd.DataFrame(sens)
    sens.to_csv(REPORT / "sensitivity.csv", index=False)
    print("\nSensitivity (blend signal):\n" + sens.round(2).to_string(index=False))

    yearly = []
    for hold, g in tr.groupby("hold"):
        for y, gy in g.groupby(g.date.dt.year):
            r, _ = evaluate(gy.assign(hold=hold))
            for _, x in r.iterrows():
                yearly.append({"hold_days": hold, "year": y, "strategy": x.strategy, "total_volpts": x.total_pnl_volpts})
    yearly = pd.DataFrame(yearly).pivot_table(index=["hold_days", "year"], columns="strategy", values="total_volpts")
    yearly.to_csv(REPORT / "by_year.csv")
    print("\nP&L by year (vol points):\n" + yearly.round(1).to_string())

    holds = sorted(tr.hold.unique())
    fig, ax = plt.subplots(len(holds), 1, figsize=(13, 4.2 * len(holds)))
    for a, hold in zip(np.atleast_1d(ax), holds):
        for name in ["NN v2 + PCA-VAR blend signal", "PCA-VAR signal", "Always short straddle", "Oracle (knows future vol)"]:
            c = curves[(hold, name)]
            a.plot(c.index, c.values, lw=1.4 if "blend" in name else 0.9, label=name,
                   ls="--" if "Oracle" in name else "-")
        a.axhline(0, color="k", lw=0.5)
        a.set_title(f"Delta-hedged ATM straddles, {hold}-day holds, after costs")
        a.set_ylabel("cumulative P&L (vol points)"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(REPORT / "equity_curves.png", dpi=130); plt.close()

    with open(REPORT / "assumptions.json", "w") as f:
        json.dump({"option_cost_per_side": OPT_COST, "futures_cost_per_side": FUT_COST, "threshold_volpts": THRESHOLD,
                   "setups": {str(k): v for k, v in SETUPS.items()}}, f, indent=2)
    print("saved", REPORT)


if __name__ == "__main__":
    main()
