import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pricing import black76, implied_vol  # noqa: E402


def test_round_trip():
    rng = np.random.default_rng(0)
    n = 20000
    F = 23000.0
    K = F * np.exp(rng.uniform(-0.4, 0.4, n))
    T = rng.uniform(3 / 365, 2.0, n)
    sig = rng.uniform(0.05, 1.2, n)
    call = rng.random(n) < 0.5
    df = np.exp(-0.06 * T)
    p = black76(F, K, T, sig, call, df)
    iv = implied_vol(p, F, K, T, call, df)
    good = ~np.isnan(iv)
    # options with ~zero time value carry no vol information; ignore those
    intrinsic = df * np.where(call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    meaningful = (p - intrinsic) > 1e-6 * F
    assert good[meaningful].mean() > 0.999
    err = np.abs(iv - sig)[good & meaningful]
    assert np.quantile(err, 0.999) < 1e-6, err.max()


def test_put_call_parity_consistency():
    F, K, T, s, df = 23000.0, 22500.0, 30 / 365, 0.14, 0.995
    c = black76(F, K, T, s, True, df)
    p = black76(F, K, T, s, False, df)
    assert abs((c - p) - df * (F - K)) < 1e-8


def test_arbitrage_prices_return_nan():
    assert np.isnan(implied_vol(10.0, 23000, 22000, 0.1, True))   # below intrinsic
    assert np.isnan(implied_vol(30000.0, 23000, 22000, 0.1, True))  # above forward


if __name__ == "__main__":
    test_round_trip(); test_put_call_parity_consistency(); test_arbitrage_prices_return_nan()
    print("all pricing tests passed")
