# %% [markdown]
# # NeuraVol Forecast v3: one model trained on NIFTY, BANKNIFTY and FINNIFTY
#
# v2 learned from NIFTY alone (about 2,600 days). v3 trains **one shared model on several indices** so it sees more
# volatility regimes and shocks. Each surface is tagged with its index, so the model can learn what is common
# (how skew and term structure usually move) and what is index-specific.
#
# What is the same as v2: 8 surface factors, GRU over 20 days, 1/5/20-day horizons, spike-filtered training targets,
# event calendar, Huber loss, calendar-arbitrage penalty, shrinkage, 5-seed ensemble, walk-forward 2020-2026, and
# scoring on raw surfaces (quoted cells only). **NIFTY results are directly comparable with v2.**
#
# New in v3:
# 1. Shared factor basis and scaling fitted on all indices in the training window.
# 2. All indices on one trading calendar. Days where an index has no surface are carried forward as inputs
#    (with a "days since last real surface" feature) and never used as targets.
# 3. Index one-hot input, per-index shrinkage, per-index results.
# 4. A comparison of training sets: NIFTY only (= v2 setup), NIFTY + BANKNIFTY, all three.
#    The final model uses the set with the best *validation* score (test years are never used to choose).
#
# **Data note.** FINNIFTY options were thinly traded in 2021-22 and lost weekly expiries in late 2024, so only
# ~340 FINNIFTY days have a reliable surface (mostly 2023-24). BANKNIFTY has ~2,640 good days.
#
# **Run:** `Runtime > Change runtime type > T4 GPU`, then `Runtime > Run all`. Upload the six files from
# `data/surfaces/` (select all in one dialog): `NIFTY_grid.npz`, `NIFTY_diagnostics.csv.gz`, `BANKNIFTY_grid.npz`,
# `BANKNIFTY_diagnostics.csv.gz`, `FINNIFTY_grid.npz`, `FINNIFTY_diagnostics.csv.gz`. Optional: `events.csv`.
# Set `QUICK = True` first for a ~5 minute check that everything runs.

# %%
QUICK = False            # True = quick smoke test (one test year, one seed, few epochs, no comparison)
COMPARE = True           # also train "NIFTY only" and "NIFTY + BANKNIFTY" to measure what the extra data adds
USE_DRIVE = False
DRIVE_DIR = "/content/drive/MyDrive/NEURAVOLAFORECAST"

INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY"]   # NIFTY must be first: it is the product and the reference
HORIZONS = [1, 5, 20]
LOOKBACK = 20
N_FACTORS = 8
TEST_YEARS = list(range(2020, 2027)) if not QUICK else [2024]
SEEDS = [0, 1, 2, 3, 4] if not QUICK else [0]
MAX_EPOCHS = 200 if not QUICK else 8
PATIENCE = 25
COVID = ("2020-02-15", "2020-06-30")
SPIKE_THRESHOLD = 0.12
V2_REFERENCE = {   # NIFTY RMSE (vol points) of v2 from its own walk-forward run, for comparison
    1: {"NN v2": 1.90, "v2 blend": 1.82}, 5: {"NN v2": 3.61, "v2 blend": 3.48}, 20: {"NN v2": 6.24, "v2 blend": 6.02}}

# %%
import os, json, time
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
except Exception as e:  # plots are optional
    plt = None
    print("matplotlib unavailable, plots skipped:", e)
from scipy.stats import norm

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

NEEDED = [f"{s}_{x}" for s in INDICES for x in ("grid.npz", "diagnostics.csv.gz")]
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_DIR = DRIVE_DIR
elif IN_COLAB:
    from google.colab import files
    DATA_DIR = "/content"
    while True:
        missing = [f for f in NEEDED if not os.path.exists(os.path.join(DATA_DIR, f))]
        if not missing:
            break
        print("Please upload (select several files in one dialog):", ", ".join(missing), "(+ optional events.csv)")
        files.upload()
else:
    DATA_DIR = os.environ.get("NEURAVOL_DATA", ".")
for f in NEEDED + ["events.csv"]:
    p = os.path.join(DATA_DIR, f)
    print(f"{'found' if os.path.exists(p) else ('MISSING' if f in NEEDED else 'not provided (optional)')}: {f}")
OUT_DIR = os.environ.get("NEURAVOL_OUT", os.path.join("/content" if IN_COLAB else DATA_DIR, "nn_v3_outputs"))
os.makedirs(OUT_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Data: all indices on one trading calendar
# Every index becomes a block of rows on the common calendar, from its first surface to the last trading day.
# Rows are stacked into global arrays, so row `g + h` is always the same index `h` trading days later.

# %%
def despike(x, thr):
    x = x.copy()
    d_prev = x[1:-1] - x[:-2]
    d_next = x[1:-1] - x[2:]
    spike = (np.sign(d_prev) == np.sign(d_next)) & (np.minimum(np.abs(d_prev), np.abs(d_next)) > thr)
    x[1:-1] = np.where(spike, 0.5 * (x[:-2] + x[2:]), x[1:-1])
    return x, spike


raw = {}
for s in INDICES:
    z = np.load(os.path.join(DATA_DIR, f"{s}_grid.npz"), allow_pickle=True)
    dg = pd.read_csv(os.path.join(DATA_DIR, f"{s}_diagnostics.csv.gz"), parse_dates=["date"]).set_index("date")
    raw[s] = dict(dates=pd.to_datetime(z["dates"]), iv=z["iv"].astype(np.float64), mask=z["mask"].astype(bool),
                  tenors=z["tenors"].astype(float), mon=z["moneyness"].astype(float), diag=dg)
TENORS, MON = raw[INDICES[0]]["tenors"], raw[INDICES[0]]["mon"]
for s in INDICES:
    assert np.allclose(raw[s]["tenors"], TENORS) and np.allclose(raw[s]["mon"], MON), f"{s}: grid differs from NIFTY"
NT, NK = len(TENORS), len(MON)
NC = NT * NK
TAU = np.repeat(TENORS / 365.0, NK)
CAL = pd.DatetimeIndex(sorted(set().union(*[set(raw[s]["dates"]) for s in INDICES])))

blocks = []
for sid, s in enumerate(INDICES):
    r = raw[s]
    cal = CAL[CAL >= r["dates"][0]]
    n = len(cal)
    pos_in = cal.get_indexer(r["dates"])
    logiv = np.full((n, NC), np.nan)
    logiv[pos_in] = np.log(r["iv"].reshape(len(pos_in), NC))
    mask = np.zeros((n, NC), bool)
    mask[pos_in] = r["mask"].reshape(len(pos_in), NC)
    present = np.zeros(n, bool)
    present[pos_in] = True
    # spike filter on the calendar array: a day is only compared with real surfaces on the trading days
    # directly before and after it (NaN neighbours never count as a spike); then carry forward over gaps
    with np.errstate(invalid="ignore"):
        clean, sp = despike(logiv, SPIKE_THRESHOLD)
    sp = sp[present[1:-1]]
    logiv_f = pd.DataFrame(logiv).ffill().values
    clean_f = pd.DataFrame(clean).ffill().values
    last_real = pd.Series(np.where(present, np.arange(n), np.nan)).ffill().values
    stale = np.minimum(np.arange(n) - last_real, 20) / 20.0
    fwd = r["diag"]["F_near"].reindex(cal).ffill().values
    ret = np.clip(np.r_[0.0, np.diff(np.log(fwd))], -0.15, 0.15)
    rs = pd.Series(ret)
    mkt = np.c_[ret * 100, np.abs(ret) * 100,
                rs.rolling(5, min_periods=1).std().fillna(0).values * np.sqrt(252) * 100,
                rs.rolling(21, min_periods=1).std().fillna(0).values * np.sqrt(252) * 100, stale]
    blocks.append(dict(sid=np.full(n, sid), pos=np.arange(n), length=np.full(n, n), dates=cal, logiv=logiv_f,
                       clean=clean_f, mask=mask, present=present, mkt=mkt))
    print(f"{s:10s} {present.sum():5d} real surfaces on {n} calendar days ({cal[0].date()} -> {cal[-1].date()}), "
          f"spike cells replaced {sp.mean():.2%}")

SID = np.concatenate([b["sid"] for b in blocks])
POS = np.concatenate([b["pos"] for b in blocks])
LEN = np.concatenate([b["length"] for b in blocks])
DATES = pd.DatetimeIndex(np.concatenate([b["dates"].values for b in blocks]))
LOGIV = np.concatenate([b["logiv"] for b in blocks])
LOGIV_CLEAN = np.concatenate([b["clean"] for b in blocks])
MASKF = np.concatenate([b["mask"] for b in blocks])
PRESENT = np.concatenate([b["present"] for b in blocks])
MKT = np.concatenate([b["mkt"] for b in blocks])
IVF = np.exp(LOGIV)            # carried forward on missing days; scoring masks those days out (MASKF is False)
G = len(SID)

# --- event calendar features (shared by all indices) ---
BUDGET = ["2016-02-29", "2017-02-01", "2018-02-01", "2019-02-01", "2019-07-05", "2020-02-01", "2021-02-01",
          "2022-02-01", "2023-02-01", "2024-02-01", "2024-07-23", "2025-02-01", "2026-02-01"]
ELECTION_RESULTS = ["2019-05-23", "2024-06-04"]
events = {"budget": pd.to_datetime(BUDGET), "election": pd.to_datetime(ELECTION_RESULTS)}
ev_path = os.path.join(DATA_DIR, "events.csv")
if os.path.exists(ev_path):
    ev = pd.read_csv(ev_path, parse_dates=["date"])
    for name, g in ev.groupby("event"):
        events[str(name).lower()] = pd.DatetimeIndex(g["date"].sort_values())


def event_features(dts, ev_dates, ahead_scale=10.0, after_scale=5.0):
    ev_dates = pd.DatetimeIndex(sorted(ev_dates))
    j = ev_dates.searchsorted(dts)
    nxt = np.where(j < len(ev_dates), (ev_dates[np.minimum(j, len(ev_dates) - 1)] - dts).days, np.inf)
    prv = np.where(j > 0, (dts - ev_dates[np.maximum(j - 1, 0)]).days, np.inf)
    return np.c_[np.exp(-nxt / ahead_scale), np.exp(-prv / after_scale)]


EVT = [event_features(DATES, v) for v in events.values()]
month_end = DATES + pd.offsets.BMonthEnd(0)
EVT.append(np.c_[np.exp(-(month_end - DATES).days.values / 5.0)])
EVT.append(np.eye(5)[np.clip(DATES.dayofweek.values, 0, 4)])
EVT = np.concatenate(EVT, axis=1)
ONEHOT = np.eye(len(INDICES))[SID]
EXO = np.c_[MKT, EVT, ONEHOT].astype(np.float64)
NE = EXO.shape[1]
ONEHOT_COLS = N_FACTORS + NE - len(INDICES) + np.arange(len(INDICES))   # positions inside [factors, EXO]

HMAX = max(HORIZONS)
HZ = np.array(HORIZONS)
VALID = (POS >= LOOKBACK - 1) & (POS + HMAX < LEN) & PRESENT     # usable forecast origins
VALID[VALID] &= np.stack([PRESENT[np.where(VALID)[0] + h] for h in HORIZONS], 1).any(1)
print(f"\n{G} rows in total, {VALID.sum()} usable forecast origins "
      f"({', '.join(f'{s} {int((VALID & (SID == i)).sum())}' for i, s in enumerate(INDICES))}) | exogenous features {NE}")

# %% [markdown]
# ## 2. Scoring helpers and baselines (identical to v2, applied per index)

# %%
def rmse_vp(pred_iv, true_iv, mask):
    e = (pred_iv - true_iv)[mask] * 100
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def daily_loss(pred_iv, true_iv, mask):
    se = ((pred_iv - true_iv) * 100) ** 2
    return np.array([se[i][mask[i]].mean() if mask[i].any() else np.nan for i in range(len(se))])


def diebold_mariano(loss_a, loss_b, h):
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 30:
        return float("nan"), float("nan")
    var = np.var(d)
    for l in range(1, max(h - 1, 0) + 1):
        var += 2 * (1 - l / h) * np.cov(d[l:], d[:-l], ddof=0)[0, 1]
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(norm.cdf(stat))


def region_table(pred_iv, true_iv, mask):
    P, T_, M = (a.reshape(-1, NT, NK) for a in (pred_iv, true_iv, mask))
    out = {"overall": rmse_vp(pred_iv, true_iv, mask)}
    for i, t in enumerate(TENORS):
        out[f"{int(t)}d"] = rmse_vp(P[:, i], T_[:, i], M[:, i])
    for name, sel in {"put wing": MON <= -0.05, "near ATM": np.abs(MON) <= 0.025, "call wing": MON >= 0.05}.items():
        out[name] = rmse_vp(P[:, :, sel], T_[:, :, sel], M[:, :, sel])
    return out


def fit_pca_var(X, k=3):
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    V = Vt[:k].T
    F = (X - mu) @ V
    B = np.linalg.lstsq(np.c_[np.ones(len(F) - 1), F[:-1]], F[1:], rcond=None)[0]
    return mu, V, B


def pca_var_forecast(x, model, h):
    mu, V, B = model
    f = (x - mu) @ V
    for _ in range(h):
        f = B[0] + f @ B[1:]
    return mu + f @ V.T


def block_start(g):
    return int(g - POS[g])


def test_indices(year, sid):
    return np.where(VALID & (SID == sid) & (DATES.year == year))[0]


def baselines(h, origins):
    """No change and PCA-VAR (refit monthly on that index's own history)."""
    rw = IVF[origins]
    pv = np.empty_like(rw)
    model, last_fit = None, -10 ** 9
    for j, g in enumerate(origins):
        if g - last_fit >= 21:
            hist = np.arange(block_start(g), g + 1)
            hist = hist[PRESENT[hist]]
            model, last_fit = (fit_pca_var(IVF[hist]) if len(hist) > 30 else None), g
        pv[j] = pca_var_forecast(IVF[g], model, h) if model is not None else IVF[g]
    return {"No change": rw, "PCA-VAR": pv}


def split_train_val(first_test, sids, val_frac=0.15):
    """Validation = last 15% of NIFTY's usable history before `first_test` (a date). Training = every index's
    origins whose targets end before validation starts (embargo), so no information leaks across the split."""
    usable = np.where(VALID & np.isin(SID, sids))[0]
    usable = usable[DATES[usable + HMAX] < first_test]
    nifty = usable[SID[usable] == 0]
    n_val = max(60, int(len(nifty) * val_frac))
    val_start = DATES[nifty[-n_val]]
    train = usable[DATES[usable + HMAX] < val_start]
    val = usable[DATES[usable] >= val_start]
    return train, val


# %% [markdown]
# ## 3. Factor space (shared by all indices)

# %%
def fit_factors(rows, k=N_FACTORS):
    rows = rows[PRESENT[rows]]
    X = LOGIV_CLEAN[rows]
    W = np.where(MASKF[rows], 1.0, 0.1)
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd((X - mu) * W, full_matrices=False)
    return dict(mu=mu, V=Vt[:k].T, explained=float((S[:k] ** 2).sum() / (S ** 2).sum()))


def factor_scores(fac, logiv):
    return (logiv - fac["mu"]) @ fac["V"]


_tr, _va = split_train_val(pd.Timestamp("2024-01-01"), [0, 1, 2])
_fac = fit_factors(np.where(DATES <= DATES[_tr].max())[0])
print(f"{N_FACTORS} shared factors explain {_fac['explained']:.1%} of surface variance (data before 2024)")
print(f"split before 2024: train {len(_tr)} ({np.bincount(SID[_tr], minlength=len(INDICES))}), "
      f"val {len(_va)} ({np.bincount(SID[_va], minlength=len(INDICES))})")
for sid, s in enumerate(INDICES):
    _o = test_indices(2024, sid)
    if len(_o):
        b5 = baselines(5, _o)
        print(f"{s} 2024 h=5: " + ", ".join(f"{k} {rmse_vp(v, IVF[_o + 5], MASKF[_o + 5]):.2f}" for k, v in b5.items()))

# %% [markdown]
# ## 4. The v3 model
# Same network as v2, plus the index one-hot in the inputs and in the output head.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as Fn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEV, torch.cuda.get_device_name(0) if DEV.type == "cuda" else "")
H = len(HORIZONS)
NI = len(INDICES)
LAMBDA_ARB = 1.0
TAU_T = torch.tensor(TAU, dtype=torch.float32, device=DEV)


class FactorNet(nn.Module):
    def __init__(self, k=N_FACTORS, n_exo=NE, n_idx=NI, hid=64, drop=0.25):
        super().__init__()
        self.k = k
        self.enc = nn.Sequential(nn.Linear(k + n_exo, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, hid), nn.GELU())
        self.gru = nn.GRU(hid, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + k + n_idx, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, H * k))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, seq, last):
        h, _ = self.gru(self.enc(seq))
        return self.head(torch.cat([h[:, -1], last], dim=1)).view(-1, H, self.k)


def make_prep(train_idx, sids):
    rows = np.where(np.isin(SID, sids) & (DATES <= DATES[train_idx].max()))[0]
    fac = fit_factors(rows)
    F_all = factor_scores(fac, LOGIV)
    feats = np.c_[F_all, EXO]
    lag_rows = np.unique(np.concatenate([train_idx - j for j in range(LOOKBACK)]))
    mu, sd = feats[lag_rows].mean(0), feats[lag_rows].std(0) + 1e-6
    mu[ONEHOT_COLS], sd[ONEHOT_COLS] = 0.0, 1.0                     # keep the index flag as a clean 0/1
    ysd, scale = [], []
    for h in HORIZONS:
        ok = train_idx[PRESENT[train_idx + h]]
        ysd.append((factor_scores(fac, LOGIV_CLEAN[ok + h]) - F_all[ok]).std(0) + 1e-4)
        scale.append(rmse_vp(IVF[ok], np.exp(LOGIV_CLEAN[ok + h]), MASKF[ok + h]))
    return dict(fac=fac, S=(feats - mu) / sd, mu=mu, sd=sd, ysd=np.array(ysd), scale=np.array(scale))


LAST_COLS = np.r_[np.arange(N_FACTORS), ONEHOT_COLS]


def tensors(idx, prep):
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=DEV)
    S = prep["S"]
    b = dict(seq=t(S[idx[:, None] - np.arange(LOOKBACK - 1, -1, -1)[None, :]]),
             last=t(S[idx][:, LAST_COLS]), logiv=t(LOGIV[idx]))
    if VALID[idx].all():
        tgt = idx[:, None] + HZ[None, :]
        pres = PRESENT[tgt][:, :, None]
        b["clean"] = t(np.exp(LOGIV_CLEAN[tgt]))
        b["raw"] = t(IVF[tgt])
        b["w"] = t(np.where(MASKF[tgt], 1.0, 0.1) * pres)            # missing target days carry no weight
        b["m"] = t((MASKF[tgt] & pres).astype(np.float32))
    return b


class Consts:
    def __init__(self, prep):
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=DEV)
        self.V, self.ysd, self.scale = t(prep["fac"]["V"]), t(prep["ysd"]), t(prep["scale"])


def pred_dlog(model, b, c):
    dF = model(b["seq"], b["last"]) * c.ysd
    return torch.einsum("bhk,ck->bhc", dF, c.V)


def loss_fn(model, b, c):
    iv = torch.exp(b["logiv"][:, None, :] + pred_dlog(model, b, c))
    err = (iv - b["clean"]) * 100 / c.scale[None, :, None]
    hub = Fn.smooth_l1_loss(err, torch.zeros_like(err), beta=1.0, reduction="none")
    loss = (b["w"] * hub).sum() / b["w"].sum().clamp_min(1e-6)
    wt = (iv ** 2 * TAU_T).view(-1, H, NT, NK)
    return loss + LAMBDA_ARB * 1e4 * torch.relu(wt[:, :, :-1] - wt[:, :, 1:]).sum(dim=(2, 3)).mean()


def rel_rmse(dlog, b):
    iv = torch.exp(b["logiv"][:, None, :] + dlog)
    rw = torch.exp(b["logiv"])[:, None, :].expand_as(iv)
    r = []
    for j in range(H):
        m = b["m"][:, j]
        e1 = (((iv[:, j] - b["raw"][:, j]) * 100) ** 2 * m).sum() / m.sum().clamp_min(1e-6)
        e0 = (((rw[:, j] - b["raw"][:, j]) * 100) ** 2 * m).sum() / m.sum().clamp_min(1e-6)
        r.append(torch.sqrt(e1 / e0.clamp_min(1e-9)))
    return torch.stack(r)


def subset(b, i):
    return {k: v[i] for k, v in b.items()}


def fit_one(train_idx, val_idx, prep, seed, bs=64, lr=1e-3):
    """Early stopping on NIFTY validation error (NIFTY is the product)."""
    torch.manual_seed(seed); np.random.seed(seed)
    c = Consts(prep)
    model = FactorNet().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    tr, va = tensors(train_idx, prep), tensors(val_idx[SID[val_idx] == 0], prep)
    best, best_state, wait, n = float("inf"), None, 0, len(train_idx)
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, bs):
            loss = loss_fn(model, subset(tr, perm[s:s + bs]), c)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = rel_rmse(pred_dlog(model, va, c), va).mean().item()
        if vl < best - 1e-5:
            best, wait = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, ep + 1, best


def ens_dlog(models, idx, prep):
    c, b = Consts(prep), tensors(idx, prep)
    with torch.no_grad():
        return torch.stack([pred_dlog(m, b, c) for m in models]).mean(0), b


def choose_alpha(models, val_idx, prep, sids):
    """Shrinkage per index and horizon, picked on validation. Indices with <60 validation days use NIFTY's."""
    grid = np.round(np.arange(0, 1.21, 0.05), 2)
    out = {}
    for sid in [0] + [s for s in sids if s != 0]:
        v = val_idx[SID[val_idx] == sid]
        if len(v) < 60:
            out[sid] = out[0]
            continue
        dl, b = ens_dlog(models, v, prep)
        out[sid] = np.array([grid[int(np.argmin([rel_rmse(dl * a, b)[j].item() for a in grid]))] for j in range(H)])
    return out


# %% [markdown]
# ## 5. Walk-forward evaluation 2020-2026
# For each test year: retrain on everything before it, 5-seed ensemble, shrinkage chosen on validation.
# Run for each training set in the comparison.

# %%
CONFIGS = {"All three": [0, 1, 2]}
if COMPARE and not QUICK:
    CONFIGS = {"NIFTY only": [0], "NIFTY + BANKNIFTY": [0, 1], "All three": [0, 1, 2]}
CONFIGS = {k: [s for s in v if s < NI] for k, v in CONFIGS.items()}


def walk_forward(sids, label):
    t0 = time.time()
    origins = {sid: np.concatenate([test_indices(y, sid) for y in TEST_YEARS]) for sid in sids}
    pred = {sid: np.full((len(o), H, NC), np.nan) for sid, o in origins.items()}
    val_scores, alpha_log = [], {}
    for y in TEST_YEARS:
        first_test = pd.Timestamp(f"{y}-01-01")
        train, val = split_train_val(first_test, sids)
        prep = make_prep(train, sids)
        fits = [fit_one(train, val, prep, s) for s in SEEDS]
        models = [f[0] for f in fits]
        alpha = choose_alpha(models, val, prep, sids)
        alpha_log[y] = {INDICES[k]: v.tolist() for k, v in alpha.items()}
        val_scores.append(np.mean([f[2] for f in fits]))
        for sid in sids:
            test = test_indices(y, sid)
            if not len(test):
                continue
            dl, _ = ens_dlog(models, test, prep)
            sel = np.isin(origins[sid], test)
            pred[sid][sel] = np.exp(LOGIV[test][:, None, :] + dl.cpu().numpy() * alpha[sid][None, :, None])
        print(f"[{label}] {y}: train {len(train)} {np.bincount(SID[train], minlength=NI)} | val {len(val)} | "
              f"factors {prep['fac']['explained']:.1%} | epochs {[f[1] for f in fits]} | NIFTY val rel-RMSE "
              f"{val_scores[-1]:.3f} | alpha NIFTY {dict(zip(HORIZONS, alpha[0].tolist()))} | {time.time() - t0:.0f}s")
    return dict(origins=origins, pred=pred, val=float(np.mean(val_scores)), alpha=alpha_log)


RUNS = {label: walk_forward(sids, label) for label, sids in CONFIGS.items()}

# %% [markdown]
# ## 6. Results
# NIFTY first (comparable with v2), then the other indices.

# %%
def evaluate(sid, h, extra):
    """Table for one index and horizon. `extra` = {name: predictions on that index's origins}."""
    j = HORIZONS.index(h)
    o = RUNS[MAIN]["origins"][sid]
    tgt, msk = IVF[o + h], MASKF[o + h] & PRESENT[o + h][:, None]
    preds = {**baselines(h, o), **{k: v[:, j] for k, v in extra.items()}}
    preds["NeuraVol v3 (NN + PCA-VAR avg)"] = 0.5 * (preds[f"NN v3 ({MAIN})"] + preds["PCA-VAR"])
    tab = pd.DataFrame({k: region_table(v, tgt, msk) for k, v in preds.items()})
    calm = ~((DATES[o] >= COVID[0]) & (DATES[o] <= COVID[1]))
    tab.loc["excl. COVID crash"] = [rmse_vp(v[calm], tgt[calm], msk[calm]) for v in preds.values()]
    losses = {k: daily_loss(v, tgt, msk) for k, v in preds.items()}
    dm = {f"{a} vs {b_}": diebold_mariano(losses[a], losses[b_], h)
          for a in [f"NN v3 ({MAIN})", "NeuraVol v3 (NN + PCA-VAR avg)"] for b_ in ["No change", "PCA-VAR"]}
    yr = DATES[o].year
    yearly = pd.DataFrame({y: {k: rmse_vp(v[yr == y], tgt[yr == y], msk[yr == y]) for k, v in preds.items()}
                           for y in TEST_YEARS if (yr == y).any()}).T
    return tab, dm, yearly, preds


print("Validation score per training set (NIFTY, lower is better; used to pick the final set):")
for k, r in RUNS.items():
    print(f"  {k:18s} {r['val']:.4f}")
MAIN = min(RUNS, key=lambda k: RUNS[k]["val"])
print(f"-> final model uses: {MAIN}\n")

results, store = {}, {}
for sid in CONFIGS[MAIN]:
    name = INDICES[sid]
    for h in HORIZONS:
        extra = {f"NN v3 ({k})": r["pred"][sid] for k, r in RUNS.items() if sid in r["pred"]}
        tab, dm, yearly, preds = evaluate(sid, h, extra)
        results[(name, h)] = dict(table=tab, dm=dm, yearly=yearly)
        for k, v in preds.items():
            store[f"{name}_h{h}_{k}"] = v
        print(f"\n=== {name}, {h}-day horizon: RMSE in vol points, {len(RUNS[MAIN]['origins'][sid])} forecasts ===")
        show = tab.loc[["overall", "excl. COVID crash", "7d", "30d", "91d", "put wing", "near ATM", "call wing"]]
        if name == "NIFTY" and TEST_YEARS == list(range(2020, 2027)):   # v2 numbers cover 2020-2026 only
            show = show.copy()
            for k, v in V2_REFERENCE[h].items():
                show.loc["overall", f"[{k}]"] = v
        print(show.round(2).to_string())
        print("by year:\n" + yearly.round(2).to_string())
        for k, (s_, p_) in dm.items():
            print(f"DM {k}: stat {s_:.2f}, p = {p_:.3f} {'<- significantly better' if p_ < 0.05 else ''}")

# %%
nifty = pd.DataFrame({f"{h}d": results[("NIFTY", h)]["table"].loc["overall"] for h in HORIZONS}).T
if plt is not None:
    fig, ax = plt.subplots(figsize=(12, 4))
    nifty.plot.bar(ax=ax, rot=0, title="NIFTY forecast error by training set (walk-forward 2020-2026)")
    ax.set_ylabel("RMSE, vol points"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "v3_nifty_rmse.png"), dpi=130); plt.show()
print(nifty.round(3).to_string())

with open(os.path.join(OUT_DIR, "nn_v3_results.json"), "w") as f:
    json.dump({"main_config": MAIN, "val_scores": {k: r["val"] for k, r in RUNS.items()},
               "alphas_by_year": {k: r["alpha"] for k, r in RUNS.items()},
               "by_index_horizon": {f"{n}_{h}": dict(table=r["table"].round(4).to_dict(), dm=r["dm"],
                                                     by_year=r["yearly"].round(4).to_dict())
                                    for (n, h), r in results.items()}}, f, indent=2, default=str)
np.savez_compressed(os.path.join(OUT_DIR, "nn_v3_test_predictions.npz"),
                    **{k.replace(" ", "_"): v for k, v in store.items()},
                    **{f"{INDICES[s]}_origin_dates": DATES[o].values.astype("datetime64[D]")
                       for s, o in RUNS[MAIN]["origins"].items()})

# %% [markdown]
# ## 7. Final model on all data + latest forecast for each index
# Saved as `neuravol_v3.pt`, with everything inference needs (factors, scaling, alphas, index list, weights).

# %%
sids = CONFIGS[MAIN]
end = DATES.max() + pd.Timedelta(days=1)
train, val = split_train_val(end, sids, val_frac=0.10)
prep = make_prep(train, sids)
models = [fit_one(train, val, prep, s)[0] for s in SEEDS]
alpha = choose_alpha(models, val, prep, sids)
rows = []
for sid in sids:
    g = np.where(SID == sid)[0][-1]
    if not PRESENT[g]:
        print(f"{INDICES[sid]}: no surface on the last day ({DATES[g].date()}), no forecast")
        continue
    dl, _ = ens_dlog(models, np.array([g]), prep)
    fc = np.exp(LOGIV[g][None, :] + dl.cpu().numpy()[0] * alpha[sid][:, None])
    for j, h in enumerate(HORIZONS):
        for i, tn in enumerate(TENORS):
            for kk, m in enumerate(MON):
                c = i * NK + kk
                rows.append(dict(index=INDICES[sid], as_of=DATES[g].date(), horizon_days=h, tenor_days=int(tn),
                                 moneyness=m, iv_today=IVF[g, c], iv_forecast=fc[j, c],
                                 backed_by_quotes=bool(MASKF[g, c])))
fc_df = pd.DataFrame(rows)
fc_df.to_csv(os.path.join(OUT_DIR, "latest_forecast_v3.csv"), index=False)
torch.save({"version": 3, "indices": [INDICES[s] for s in sids], "all_indices": INDICES, "horizons": HORIZONS,
            "lookback": LOOKBACK, "n_factors": N_FACTORS, "tenors": TENORS.tolist(), "moneyness": MON.tolist(),
            "last_date": str(DATES.max().date()), "events": {k: [str(d.date()) for d in v] for k, v in events.items()},
            "factors": {"mu": prep["fac"]["mu"].tolist(), "V": prep["fac"]["V"].tolist()},
            "feat_mu": prep["mu"].tolist(), "feat_sd": prep["sd"].tolist(),
            "ysd": prep["ysd"].tolist(), "scale": prep["scale"].tolist(),
            "alpha": {INDICES[k]: v.tolist() for k, v in alpha.items()},
            "state_dicts": [{k: v.cpu() for k, v in m.state_dict().items()} for m in models]},
           os.path.join(OUT_DIR, "neuravol_v3.pt"))
print("final alphas:", {INDICES[k]: dict(zip(HORIZONS, v.tolist())) for k, v in alpha.items()})
print(fc_df[fc_df.moneyness == 0].round(4).to_string(index=False))

# %%
if IN_COLAB:
    import shutil
    from google.colab import files
    shutil.make_archive("/content/nn_v3_outputs", "zip", OUT_DIR)
    files.download("/content/nn_v3_outputs.zip")
    print("Downloaded nn_v3_outputs.zip: put it in the NEURAVOLAFORECAST folder")
