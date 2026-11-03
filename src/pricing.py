"""Black-76 pricing and a vectorised, robust implied-volatility solver."""
import numpy as np
from scipy.special import ndtr


def black76(F, K, T, sigma, is_call, df=1.0):
    """Discounted Black-76 price. All args broadcast; is_call is a bool array."""
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    sqT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    return df * np.where(is_call, call, put)


def vega76(F, K, T, sigma, df=1.0):
    sqT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqT)
    return df * F * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * sqT


def implied_vol(price, F, K, T, is_call, df=1.0, lo=1e-4, hi=5.0, tol=1e-10, iters=100):
    """Implied vol by vectorised bisection (guaranteed to converge) polished with Newton.

    Returns NaN where the price is outside no-arbitrage bounds.
    """
    price, F, K, T, df = np.broadcast_arrays(*map(lambda x: np.asarray(x, float), (price, F, K, T, df)))
    is_call = np.broadcast_to(np.asarray(is_call, bool), price.shape)
    intrinsic = df * np.where(is_call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    upper = df * np.where(is_call, F, K)
    ok = (price > intrinsic + 1e-12) & (price < upper) & (T > 0)

    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    ok &= black76(F, K, T, b, is_call, df) >= price          # root bracketed
    for _ in range(iters):
        m = 0.5 * (a + b)
        hi_side = black76(F, K, T, m, is_call, df) > price
        b = np.where(hi_side, m, b)
        a = np.where(hi_side, a, m)
        if np.nanmax(np.where(ok, b - a, 0)) < 1e-7:
            break
    s = 0.5 * (a + b)
    for _ in range(3):  # Newton polish
        v = vega76(F, K, T, s, df)
        step = np.where(v > 1e-12, (black76(F, K, T, s, is_call, df) - price) / np.maximum(v, 1e-12), 0)
        s = np.clip(s - step, lo, hi)
    return np.where(ok, s, np.nan)
