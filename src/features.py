"""Model input features. Must match the v2 Colab notebook exactly (the trained weights depend on it)."""
import numpy as np
import pandas as pd

BUDGET = ["2016-02-29", "2017-02-01", "2018-02-01", "2019-02-01", "2019-07-05", "2020-02-01", "2021-02-01",
          "2022-02-01", "2023-02-01", "2024-02-01", "2024-07-23", "2025-02-01", "2026-02-01"]
ELECTION_RESULTS = ["2019-05-23", "2024-06-04"]


def market_features(fwd):
    ret = np.clip(np.r_[0.0, np.diff(np.log(fwd))], -0.15, 0.15)
    rs = pd.Series(ret)
    return np.c_[ret * 100, np.abs(ret) * 100,
                 rs.rolling(5, min_periods=1).std().fillna(0).values * np.sqrt(252) * 100,
                 rs.rolling(21, min_periods=1).std().fillna(0).values * np.sqrt(252) * 100]


def event_features(dts, ev_dates, ahead_scale=10.0, after_scale=5.0):
    ev_dates = pd.DatetimeIndex(sorted(pd.to_datetime(ev_dates)))
    nxt = np.full(len(dts), np.inf)
    prv = np.full(len(dts), np.inf)
    for i, d in enumerate(dts):
        j = ev_dates.searchsorted(d)
        if j < len(ev_dates):
            nxt[i] = (ev_dates[j] - d).days
        if j > 0:
            prv[i] = (d - ev_dates[j - 1]).days
    return np.c_[np.exp(-nxt / ahead_scale), np.exp(-prv / after_scale)]


def default_events():
    return {"budget": BUDGET, "election": ELECTION_RESULTS}


def exogenous(dates, fwd, events):
    """dates: DatetimeIndex; fwd: near forward per day; events: {name: [dates]} in the order used in training."""
    evt = [event_features(dates, v) for v in events.values()]
    month_end = dates + pd.offsets.BMonthEnd(0)
    evt.append(np.c_[np.exp(-(month_end - dates).days.values / 5.0)])
    evt.append(np.eye(5)[np.clip(dates.dayofweek.values, 0, 4)])
    return np.c_[market_features(fwd), np.concatenate(evt, axis=1)].astype(np.float64)
