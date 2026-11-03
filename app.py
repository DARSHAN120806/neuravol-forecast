"""NeuraVol: NIFTY implied-volatility surface and forecast dashboard.

    pip install -r requirements-app.txt
    streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dashboard_data as dd

st.set_page_config(page_title="NeuraVol: NIFTY volatility forecasts", layout="wide")

AMBER, BLUE, GREY, GREEN = "#F2A33A", "#4C9AFF", "#8B949E", "#5FD39A"
SEQ = [[0, "#1E2A3A"], [0.5, "#8A5A2B"], [1, AMBER]]
DIV = [[0, BLUE], [0.5, "#161B22"], [1, AMBER]]
LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13))
DISCLAIMER = ("Research and education only. Forecasts of implied volatility, not buy/sell recommendations. "
              "Not investment advice. Past accuracy does not guarantee future accuracy.")


@st.cache_resource
def get_data():
    return dd.load()


d = get_data()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("NeuraVol")
st.sidebar.caption("Neural forecasts of the NIFTY 50 implied-volatility surface")
page = st.sidebar.radio("Page", ["Surface & forecast", "Strike explorer", "History", "Track record", "About"])
h = st.sidebar.radio("Forecast horizon", dd.HORIZONS, index=1,
                     format_func=lambda x: {1: "1 day", 5: "1 week (5 days)", 20: "1 month (20 days)"}[x])
st.sidebar.markdown(f"**Data as of:** {d.dates[-1]:%d %b %Y} (end of day)")
st.sidebar.caption("Source: NSE F&O bhavcopy. Updated daily after market close.")
st.sidebar.caption(DISCLAIMER)
H_LABEL = {1: "1 day", 5: "1 week", 20: "1 month"}[h]


def heatmap(df, title, colorscale, zmid=None, fmt=".1f", zmin=None, zmax=None):
    fig = go.Figure(go.Heatmap(z=df.values, x=df.columns, y=df.index, colorscale=colorscale, zmid=zmid,
                               zmin=zmin, zmax=zmax, texttemplate="%{z:" + fmt + "}", hoverongaps=False,
                               colorbar=dict(thickness=10)))
    fig.update_layout(title=title, height=380, yaxis=dict(autorange="reversed"), **LAYOUT)
    fig.update_xaxes(title="moneyness ln(K/F)")
    return fig


# ------------------------------------------------------------------ page: surface & forecast
if page == "Surface & forecast":
    st.title("NIFTY implied volatility")
    k = dd.kpis(d, h)
    st.caption(f"End of day {k['as_of']:%d %b %Y}. All volatilities in % per year.")
    c = st.columns(5)
    c[0].metric("NIFTY near forward", f"{k['forward']:,.1f}", f"{k['forward_chg']:+.2%}")
    c[1].metric("ATM IV, 30 days", f"{k['atm30']:.2f}%", f"{k['atm30_chg']:+.2f} pts vs prev. day", delta_color="off")
    c[2].metric("Put skew, 30 days", f"{k['skew30']:+.2f} pts", help="IV of 5% OTM put minus 5% OTM call")
    c[3].metric("Term slope, 7d to 91d", f"{k['term']:+.2f} pts", help="Negative = short-dated vol above long-dated (stress)")
    c[4].metric(f"Forecast ATM 30d, {H_LABEL}", f"{k['fc_atm30']:.2f}%", f"{k['fc_chg']:+.2f} pts", delta_color="off")

    today, mask = dd.surface_today(d)
    left, right = st.columns([1.15, 1])
    with left:
        view = st.radio("Surface view", ["Heatmap", "3D"], horizontal=True)
        if view == "Heatmap":
            st.plotly_chart(heatmap(dd.grid_frame(d, today, mask), "Today's surface (blank = extrapolated)", SEQ))
        else:
            fig = go.Figure(go.Surface(z=today * 100, x=d.mon * 100, y=d.tenors, colorscale="Viridis",
                                       colorbar=dict(thickness=10)))
            fig.update_layout(title="Today's surface", height=430,
                              scene=dict(xaxis_title="ln(K/F) %", yaxis_title="days", zaxis_title="IV %"), **LAYOUT)
            st.plotly_chart(fig)
    with right:
        tenor = st.radio("Smile for maturity", list(d.tenors), index=2, horizontal=True, format_func=lambda x: f"{x}d")
        sm = dd.smile(d, tenor, h)
        fig = go.Figure()
        fig.add_trace(go.Scatter(mode="lines", x=sm.moneyness, y=sm.hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(mode="lines", x=sm.moneyness, y=sm.lo, fill="tonexty", fillcolor="rgba(242,163,58,0.15)",
                                 line=dict(width=0), name="80% band (normal markets)"))
        fig.add_trace(go.Scatter(mode="lines", x=sm.moneyness, y=sm.today, name="today", line=dict(color=BLUE, width=3)))
        fig.add_trace(go.Scatter(mode="lines", x=sm.moneyness, y=sm.forecast, name=f"forecast, {H_LABEL}",
                                 line=dict(color=AMBER, width=3, dash="dash")))
        fig.update_layout(title=f"{tenor}-day smile: today vs forecast", height=380, xaxis_title="moneyness %",
                          yaxis_title="IV %", legend=dict(orientation="h", y=-0.25), **LAYOUT)
        st.plotly_chart(fig)

    st.subheader(f"Expected change over {H_LABEL}")
    delta = dd.forecast_grid(d, h) - today
    lim = max(0.5, float(np.nanmax(np.abs(delta * 100))))
    st.plotly_chart(heatmap(dd.grid_frame(d, delta), "Forecast minus today, vol points (amber = rise, blue = fall)",
                            DIV, zmid=0, fmt=".2f", zmin=-lim, zmax=lim))
    st.caption("Forecast = average of the NN v2 model and a PCA-VAR factor model, the most accurate combination in "
               "out-of-sample tests 2020-2026 (see Track record). Bands use typical out-of-sample error outside the "
               "2020 crash; in a crash, errors were 2-4x larger.")

# ------------------------------------------------------------------ page: strike explorer
elif page == "Strike explorer":
    st.title("Strike explorer")
    st.caption("Pick a listed contract to see its implied vol today, the forecast for it, and what that means for its "
               "premium. Analysis only: this is not a trade recommendation.")
    ex = dd.expiries(d)
    c1, c2, c3 = st.columns(3)
    labels = {i: f"{r.expiry:%d %b %Y} ({int(r.days)} days)" for i, r in ex.iterrows()}
    default = int(np.argmin(np.abs(ex.days.values - 30)))
    pick = c1.selectbox("Expiry", list(labels), index=default, format_func=lambda i: labels[i])
    row = ex.loc[pick]
    atm = int(round(row.F / 50) * 50)
    K = c2.number_input("Strike", min_value=int(row.F * 0.7) // 50 * 50, max_value=int(row.F * 1.3) // 50 * 50,
                        value=atm, step=50)
    typ = c3.radio("Type", ["Put (PE)", "Call (CE)"], horizontal=True)
    v = dd.strike_view(d, row, float(K), typ.startswith("Call"), h)
    m = st.columns(4)
    m[0].metric("Implied vol today", f"{v['iv_now']:.2f}%")
    m[1].metric(f"Forecast IV, {H_LABEL}", f"{v['iv_fc']:.2f}%", f"{v['change']:+.2f} pts", delta_color="off")
    m[2].metric("Model premium today", f"₹{v['model_price_now']:,.2f}",
                f"last traded ₹{v['last_traded']:,.2f}" if v["last_traded"] is not None else "no trade today",
                delta_color="off")
    m[3].metric(f"Premium if IV moves as forecast", f"₹{v['range_lo']:,.0f} to ₹{v['range_hi']:,.0f}",
                help="Today's premium re-priced at the forecast IV, with the 80% normal-market error band")
    where = ("below" if v["last_traded"] and v["last_traded"] < v["range_lo"] else
             "above" if v["last_traded"] and v["last_traded"] > v["range_hi"] else "inside")
    st.info(f"Strike {K:,} is {v['moneyness']:+.1f}% from the forward ({v['forward']:,.1f}), {v['days']} days to expiry. "
            f"The model expects this option's implied vol to move {v['change']:+.2f} points over the next {H_LABEL}. "
            + (f"Today's last traded premium sits {where} the model's range. " if v["last_traded"] else "")
            + "Analysis only, not a recommendation to buy or sell.")
    st.caption(f"With time decay over the {H_LABEL} included, the model premium would be about ₹{v['price_after_h']:,.2f} "
               "if the forward stays where it is. " + DISCLAIMER)

# ------------------------------------------------------------------ page: history
elif page == "History":
    st.title("Ten years of NIFTY implied volatility")
    hist = dd.history(d)
    start, end = st.slider("Period", min_value=hist.index[0].date(), max_value=hist.index[-1].date(),
                           value=(hist.index[0].date(), hist.index[-1].date()))
    hs = hist.loc[str(start):str(end)]
    fig = go.Figure()
    for col, colr in zip(["ATM 7d", "ATM 30d", "ATM 91d"], [BLUE, AMBER, GREEN]):
        fig.add_trace(go.Scatter(mode="lines", x=hs.index, y=hs[col], name=col, line=dict(width=1.2, color=colr)))
    fig.update_layout(title="At-the-money implied vol (%)", height=380, **LAYOUT)
    st.plotly_chart(fig)
    c1, c2 = st.columns(2)
    for box, col in zip((c1, c2), ["Skew 30d (-5% minus +5%)", "Term slope (91d minus 7d)"]):
        f = go.Figure(go.Scatter(mode="lines", x=hs.index, y=hs[col], line=dict(width=1, color=AMBER)))
        f.add_hline(y=0, line_color=GREY, line_width=0.5)
        f.update_layout(title=col + ", vol points", height=300, **LAYOUT)
        box.plotly_chart(f)
    st.caption("Visible events: COVID crash (Mar 2020), general elections (May 2019, Jun 2024). "
               "Negative term slope = short-dated fear above long-dated.")

# ------------------------------------------------------------------ page: track record
elif page == "Track record":
    st.title("Track record")
    st.caption("Walk-forward test: for each year 2020-2026 the model was trained only on earlier data, then "
               "forecast every day of that year. Error = RMSE in vol points on cells backed by real quotes. Lower is better.")
    t_all, t_calm = dd.track_table(d), dd.track_table(d, exclude_covid=True)
    order = ["No change", "PCA-VAR", "NN v2", dd.BLEND]
    fmt = lambda t: t[order].rename(index=lambda x: f"{x} day" + ("s" if x > 1 else "")).rename_axis("horizon").rename(columns={dd.BLEND: "NeuraVol"})
    c1, c2 = st.columns(2)
    c1.markdown("**All test days**")
    c1.dataframe(fmt(t_all).style.format("{:.2f}").highlight_min(axis=1, color="#3d2c12"))
    c2.markdown("**Excluding the Feb-Jun 2020 crash**")
    c2.dataframe(fmt(t_calm).style.format("{:.2f}").highlight_min(axis=1, color="#3d2c12"))
    sig = d.significance.get(str(h), {})
    p_rw = sig.get("NN v2 shrunk + PCA-VAR avg vs No change", [np.nan, np.nan])[1]
    st.markdown(f"**Honest verdict ({H_LABEL}):** NeuraVol has the lowest error, but the Diebold-Mariano test vs "
                f"'no change' gives p = {p_rw:.2f}, so the improvement is **not statistically significant** yet.")

    a = d.atm30[d.atm30.horizon_days == h]
    fig = go.Figure()
    fig.add_trace(go.Scatter(mode="lines", x=a.target_date, y=a.actual * 100, name="actual", line=dict(color=BLUE, width=1.3)))
    fig.add_trace(go.Scatter(mode="lines", x=a.target_date, y=a[dd.BLEND] * 100, name=f"NeuraVol forecast made {H_LABEL} earlier",
                             line=dict(color=AMBER, width=1.1)))
    fig.update_layout(title="30-day ATM implied vol: forecast vs actual (%)", height=360,
                      legend=dict(orientation="h", y=-0.2), **LAYOUT)
    st.plotly_chart(fig)

    st.markdown(f"**Error by year, {H_LABEL} horizon**")
    st.dataframe(dd.track_by_year(d, h)[order].rename(columns={dd.BLEND: "NeuraVol"}).style.format("{:.2f}").highlight_min(axis=1, color="#3d2c12"))

    st.subheader("Does it make money? Delta-hedged straddle backtest")
    st.caption("Buy (sell) an ATM straddle when the forecast says vol will rise (fall), hedge daily with futures, "
               "0.3% cost per option trade. P&L in vol points. Includes an oracle that knows the future (upper bound).")
    bt = d.bt_summary.rename(columns={"hold_days": "hold (days)", "strategy": "strategy", "trades": "trades",
                                      "hit_rate": "hit rate", "sharpe_annualised": "Sharpe",
                                      "total_pnl_volpts": "total P&L (vol pts)", "max_drawdown_volpts": "max drawdown"})
    st.dataframe(bt[["hold (days)", "strategy", "trades", "hit rate", "Sharpe", "total P&L (vol pts)", "max drawdown"]]
                 .round(2), hide_index=True)
    st.markdown("**Result:** the forecasts are more accurate than the baselines but do **not** produce trading profit "
                "after costs; simply selling volatility did better. The edge is too small to pay for trading costs.")

    live = dd.live_track_record(d)
    st.subheader("Live forecasts since launch")
    if live.empty:
        st.caption("Forecasts are logged every day by the pipeline and scored here once the target day arrives.")
    else:
        st.dataframe(live.pivot_table(index="horizon_days", columns="model", values="rmse", aggfunc="mean").round(2))

# ------------------------------------------------------------------ page: about
else:
    st.title("About NeuraVol")
    st.markdown("""
**What it does.** Every trading day NeuraVol rebuilds the NIFTY implied-volatility surface from NSE end-of-day
option prices and forecasts how it will look 1 day, 1 week and 1 month ahead.

**How the surface is built.** Forward from put-call parity; OTM options only; Black-76 implied vol;
an SVI smile per expiry; calendar-arbitrage fixes; interpolation of total variance onto a fixed grid of
7 maturities x 9 moneyness levels. Cells not backed by quotes are shown blank in the heatmap.

**The model.** A GRU network reads the last 20 days of surface factors (8 PCA factors), market returns,
realised volatility and an event calendar (Budget, election results, expiry week). It predicts the change
of each factor for all three horizons at once, shrunk toward "no change" by an amount chosen on validation
data. The published forecast averages it with a PCA-VAR factor model.

**How it was tested.** Walk-forward 2020-2026, retraining every year on past data only, compared with
"no change" and PCA-VAR on the same days, with Diebold-Mariano significance tests and a trading backtest.
All results, including the weak ones, are on the Track record page.
""")
    st.caption(DISCLAIMER)
