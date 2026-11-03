"""Load the trained v2 model and forecast from any day (CPU is enough).

Production forecast = average of (NN v2 shrunk) and (PCA-VAR), the best model in walk-forward tests.
"""
import numpy as np
import pandas as pd

from .features import exogenous

LOOKBACK_DEFAULT = 20
HMAX_SPLIT = 20


def fit_pca_var(X, k=3):
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    V = Vt[:k].T
    F = (X - mu) @ V
    B = np.linalg.lstsq(np.c_[np.ones(len(F) - 1), F[:-1]], F[1:], rcond=None)[0]
    return mu, V, B


def pca_var_forecast(ivf, t, horizons):
    """PCA-VAR forecast from day t using all surfaces up to t. Returns (H, cells)."""
    mu, V, B = fit_pca_var(ivf[: t + 1])
    out = []
    for h in horizons:
        f = (ivf[t] - mu) @ V
        for _ in range(h):
            f = B[0] + f @ B[1:]
        out.append(mu + f @ V.T)
    return np.array(out)


def _split_train(d_train, lookback, val_frac=0.10, hmax=HMAX_SPLIT):
    usable = np.arange(lookback - 1, d_train - hmax)
    n_val = max(60, int(len(usable) * val_frac))
    val = usable[-n_val:]
    return usable[usable + hmax < val[0]]


class Forecaster:
    """Wraps neuravol_v2.pt. Needs torch (CPU build is fine)."""

    def __init__(self, bundle_path):
        import torch
        import torch.nn as nn

        self.torch = torch
        b = torch.load(bundle_path, map_location="cpu", weights_only=False)
        self.b = b
        self.h = list(b["horizons"])
        self.L = int(b["lookback"])
        self.k = int(b["n_factors"])
        self.V = np.array(b["factors"]["V"])
        self.mu_f = np.array(b["factors"]["mu"])
        self.ysd = np.array(b["ysd"])
        self.alpha = np.array(b["alpha"])
        self.events = b["events"]
        n_exo = None
        H = len(self.h)
        k = self.k

        class FactorNet(nn.Module):
            def __init__(s, n_exo, hid=64, drop=0.25):
                super().__init__()
                s.enc = nn.Sequential(nn.Linear(k + n_exo, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, hid), nn.GELU())
                s.gru = nn.GRU(hid, hid, batch_first=True)
                s.head = nn.Sequential(nn.Linear(hid + k, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, H * k))

            def forward(s, seq, last):
                h, _ = s.gru(s.enc(seq))
                return s.head(torch.cat([h[:, -1], last], dim=1)).view(-1, H, k)

        sd0 = b["state_dicts"][0]
        n_exo = sd0["enc.0.weight"].shape[1] - k
        self.models = []
        for sd in b["state_dicts"]:
            m = FactorNet(n_exo)
            m.load_state_dict(sd)
            m.eval()
            self.models.append(m)
        self.n_exo = n_exo

    def _inputs(self, dates, ivf, fwd):
        logiv = np.log(ivf)
        exo = exogenous(pd.DatetimeIndex(dates), fwd, self.events)
        if exo.shape[1] != self.n_exo:
            raise ValueError(f"feature mismatch: model expects {self.n_exo} exogenous features, got {exo.shape[1]}")
        feats = np.c_[(logiv - self.mu_f) @ self.V, exo]
        last_train = int(np.searchsorted(pd.DatetimeIndex(dates), pd.Timestamp(self.b["last_date"]), side="right"))
        train = _split_train(last_train, self.L)
        rows = np.unique(np.concatenate([train - j for j in range(self.L)]))
        mu, sd = feats[rows].mean(0), feats[rows].std(0) + 1e-6
        return logiv, (feats - mu) / sd

    def nn_forecast(self, dates, ivf, fwd, t):
        """NN v2 (shrunk) forecast from day t: (H, cells)."""
        torch = self.torch
        logiv, S = self._inputs(dates, ivf, fwd)
        seq = torch.tensor(S[t - self.L + 1: t + 1][None], dtype=torch.float32)
        last = torch.tensor(S[t, : self.k][None], dtype=torch.float32)
        with torch.no_grad():
            dF = torch.stack([m(seq, last) for m in self.models]).mean(0)[0].numpy() * self.ysd   # (H, k)
        dlog = dF @ self.V.T
        return np.exp(logiv[t][None, :] + self.alpha[:, None] * dlog)

    def forecast(self, dates, ivf, fwd, t):
        nn_ = self.nn_forecast(dates, ivf, fwd, t)
        pv = pca_var_forecast(ivf, t, self.h)
        return {"nn": nn_, "pcavar": pv, "blend": 0.5 * (nn_ + pv)}


def forecast_table(as_of, horizons, tenors, moneyness, iv_today, mask_today, preds):
    rows = []
    nK = len(moneyness)
    for j, h in enumerate(horizons):
        for i, tn in enumerate(tenors):
            for kk, m in enumerate(moneyness):
                c = i * nK + kk
                rows.append(dict(as_of=pd.Timestamp(as_of).date(), horizon_days=int(h), tenor_days=int(tn),
                                 moneyness=float(m), iv_today=float(iv_today[c]),
                                 iv_nn=float(preds["nn"][j, c]), iv_pcavar=float(preds["pcavar"][j, c]),
                                 iv_forecast=float(preds["blend"][j, c]), backed_by_quotes=bool(mask_today[c])))
    return pd.DataFrame(rows)
