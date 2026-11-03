"""Sanity-check plots for the surface history -> reports/*.png (+ an interactive 3D html if plotly is installed)."""
import numpy as np
import pandas as pd

from src import config as C
from src.evaluation import load_grid
from src.io_utils import load_table


def main(symbol=C.SYMBOL):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dates, iv, mask, tenors, mon = load_grid(C.SURFACES / f"{symbol}_grid.npz")
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    ti = {t: i for i, t in enumerate(tenors)}
    k0 = int(np.argmin(np.abs(mon)))
    kp, kc = int(np.argmin(np.abs(mon + 0.05))), int(np.argmin(np.abs(mon - 0.05)))

    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for t in (7, 30, 91):
        ax[0].plot(dates, iv[:, ti[t], k0] * 100, lw=0.8, label=f"{t}d ATM")
    ax[0].set_ylabel("IV %"); ax[0].legend(); ax[0].set_title(f"{symbol} ATM implied volatility")
    ax[1].plot(dates, (iv[:, ti[30], kp] - iv[:, ti[30], kc]) * 100, lw=0.8, color="tab:red")
    ax[1].set_ylabel("vol pts"); ax[1].set_title("30d skew: IV(-5%) minus IV(+5%)")
    ax[2].plot(dates, (iv[:, ti[91], k0] - iv[:, ti[7], k0]) * 100, lw=0.8, color="tab:green")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylabel("vol pts"); ax[2].set_title("Term slope: 91d minus 7d ATM (negative = stress / inversion)")
    fig.tight_layout(); fig.savefig(C.REPORTS / f"{symbol}_history.png", dpi=130); plt.close(fig)

    diag = load_table(C.SURFACES / f"{symbol}_diagnostics", date_cols=("date",))
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(diag.date, diag.svi_rmse_median * 100, lw=0.6); ax[0].set_ylabel("vol pts")
    ax[0].set_title("Median SVI fit error per day")
    ax[1].plot(diag.date, diag.n_points, lw=0.6); ax[1].set_title("Clean option quotes used per day")
    fig.tight_layout(); fig.savefig(C.REPORTS / f"{symbol}_quality.png", dpi=130); plt.close(fig)

    for label, d in (("latest", dates[-1]), ("covid", pd.Timestamp("2020-03-23"))):
        i = int(np.argmin(np.abs(dates - d)))
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(np.where(mask[i], iv[i], np.nan) * 100, cmap="magma", aspect="auto")
        ax.set_xticks(range(len(mon)), [f"{m:+.1%}" for m in mon]); ax.set_yticks(range(len(tenors)), [f"{t}d" for t in tenors])
        for r in range(len(tenors)):
            for c in range(len(mon)):
                ax.text(c, r, f"{iv[i, r, c] * 100:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if mask[i, r, c] else "grey")
        ax.set_title(f"{symbol} surface {dates[i].date()} (grey = extrapolated)")
        fig.colorbar(im, label="IV %"); fig.tight_layout()
        fig.savefig(C.REPORTS / f"{symbol}_surface_{label}.png", dpi=130); plt.close(fig)

    try:
        import plotly.graph_objects as go
        i = len(dates) - 1
        fig = go.Figure(go.Surface(z=iv[i] * 100, x=mon * 100, y=tenors, colorscale="Viridis"))
        fig.update_layout(title=f"{symbol} implied volatility surface {dates[i].date()}",
                          scene=dict(xaxis_title="ln(K/F) %", yaxis_title="days", zaxis_title="IV %"))
        fig.write_html(C.REPORTS / f"{symbol}_surface_3d.html")
    except ImportError:
        print("plotly not installed: skipped 3D html")
    print("plots saved in", C.REPORTS)


if __name__ == "__main__":
    main()
