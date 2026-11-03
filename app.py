"""NeuraVol: NIFTY implied-volatility surface and forecast dashboard.

    pip install -r requirements.txt
    streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dashboard_data as dd

st.set_page_config(page_title="NeuraVol: NIFTY volatility forecasts", page_icon="🌊", layout="wide")

# ------------------------------------------------------------------ palette (pastel + one strong accent)
INK, MUTED, LINE = "#1E2235", "#5B6178", "#E3E6F0"
BG, CARD, SIDE = "#F5F6FB", "#FFFFFF", "#EEF0FA"
INDIGO, INDIGO_SOFT = "#5B5BD6", "#E4E4FB"
CORAL, CORAL_SOFT = "#E8806F", "#FDE7E2"
TEAL, TEAL_SOFT = "#2FA39A", "#DDF3F0"
SKY, SKY_SOFT = "#5AA9E6", "#E2F0FB"
BUTTER_SOFT = "#FFF5D9"
SEQ = [[0, "#F3F4FF"], [0.5, "#C9CCF7"], [1, "#8C8FEA"]]          # IV level heatmap
DIV = [[0, "#8CC8F2"], [0.5, "#FFFFFF"], [1, "#F4A493"]]          # forecast change: blue fall, coral rise
HILITE = "#DDF3F0"
LAYOUT = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              margin=dict(l=10, r=10, t=48, b=10), font=dict(size=13, color=INK, family="Poppins, sans-serif"),
              title_font=dict(size=15, color=INK))
DISCLAIMER = ("Research and education only. Forecasts of implied volatility, not buy/sell recommendations. "
              "Not investment advice. Past accuracy does not guarantee future accuracy.")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp *:not(.material-icons):not([data-testid="stIconMaterial"]):not(span[class*="material"]) {{ font-family: 'Poppins', sans-serif; }}
.stApp {{ background: {BG}; color: {INK}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.02em; }}
h1 {{ font-weight: 700; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: {SIDE}; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] * {{ color: {INK}; }}
[data-testid="stSidebar"] .stButton > button {{
  width: 100%; justify-content: flex-start; text-align: left; border: 1px solid transparent;
  background: transparent; border-radius: 12px; padding: 0.55rem 0.85rem; min-height: 44px;
  font-weight: 500; color: {MUTED}; box-shadow: none; }}
[data-testid="stSidebar"] .stButton > button > div {{ justify-content: flex-start; }}
[data-testid="stSidebar"] .stButton > button p {{ font-size: 0.95rem; }}
[data-testid="stSidebar"] .stButton > button:hover {{ background: {INDIGO_SOFT}; color: {INDIGO}; border-color: transparent; }}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {{
  background: {INDIGO}; color: #FFFFFF; border-color: {INDIGO}; }}
[data-testid="stSidebar"] .stButton > button[kind="primary"] *,
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] * {{ color: #FFFFFF; }}
.nv-brand {{ display: flex; align-items: center; gap: 10px; margin: 0.2rem 0 0.2rem; }}
.nv-brand b {{ font-size: 1.25rem; font-weight: 700; color: {INK}; }}
.nv-tag {{ color: {MUTED}; font-size: 0.85rem; line-height: 1.45; margin-bottom: 1.1rem; }}
.nv-section {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: {MUTED}; margin: 1.1rem 0 0.4rem 0.2rem; }}
.nv-note {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; padding: 0.75rem 0.85rem;
  font-size: 0.8rem; color: {MUTED}; line-height: 1.5; }}

/* cards */
.nv-kpis {{ display: grid; gap: 14px; margin: 0.6rem 0 1.2rem; }}
@media (max-width: 1100px) {{ .nv-kpis {{ grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)) !important; }} }}
.nv-kpi {{ border-radius: 16px; padding: 16px 18px; border: 1px solid rgba(30,34,53,0.05); }}
.nv-kpi .l {{ font-size: 0.8rem; color: {MUTED}; font-weight: 500; }}
.nv-kpi .v {{ font-size: 1.6rem; white-space: nowrap; font-weight: 700; color: {INK}; margin-top: 4px; letter-spacing: -0.02em; }}
.nv-kpi .s {{ font-size: 0.8rem; margin-top: 6px; color: {MUTED}; }}
.nv-pill {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-weight: 600; font-size: 0.78rem; }}
.nv-callout {{ background: {CARD}; border: 1px solid {LINE}; border-left: 0; border-radius: 16px;
  padding: 16px 18px; color: {INK}; line-height: 1.6; margin: 0.6rem 0; }}
.nv-verdict {{ background: {BUTTER_SOFT}; border-radius: 16px; padding: 14px 18px; color: {INK}; line-height: 1.6; margin: 0.4rem 0 1rem; }}
.nv-sub {{ color: {MUTED}; font-size: 0.92rem; margin-top: -0.4rem; margin-bottom: 0.8rem; }}

.nv-zone {{ display:inline-block; padding:3px 10px; border-radius:999px; font-weight:600; font-size:0.78rem; }}
.nv-card {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px; padding:16px 18px; margin-bottom:12px; }}
.nv-card h4 {{ margin:0 0 6px 0; font-size:1.05rem; color:{INK}; }}
.nv-row {{ display:flex; justify-content:space-between; gap:10px; font-size:0.88rem; padding:3px 0; color:{MUTED}; }}
.nv-row b {{ color:{INK}; font-weight:600; }}
.nv-what {{ font-size:0.85rem; color:{INK}; margin-top:8px; line-height:1.5; }}
.nv-bar {{ position:relative; height:12px; border-radius:999px; background:linear-gradient(90deg,#BFE8E2 0 25%,#C9E2F7 25% 75%,#FBE6A6 75% 90%,#F6C2B7 90% 100%); margin:14px 0 6px; }}
.nv-dot {{ position:absolute; top:-5px; width:22px; height:22px; border-radius:50%; background:{INDIGO}; border:3px solid #fff; transform:translateX(-50%); }}
.nv-scale {{ position:relative; height:16px; font-size:0.72rem; color:{MUTED}; }}
.nv-scale span {{ position:absolute; transform:translateX(-50%); white-space:nowrap; }}
.nv-mood {{ font-size:2rem; font-weight:700; color:{INK}; }}
.nv-sum {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px; padding:14px 16px; margin-bottom:10px; }}
.nv-sum .t {{ font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:{INDIGO}; }}
.nv-sum .x {{ font-size:0.93rem; color:{INK}; line-height:1.55; margin-top:4px; }}
.nv-def {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:6px 0 16px; }}
.nv-def div {{ border-radius:14px; padding:12px 14px; font-size:0.86rem; line-height:1.5; color:{INK}; }}
[data-testid="stExpander"] {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; }}

/* charts and tables as cards */
[data-testid="stPlotlyChart"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 6px 8px; }}
[data-testid="stDataFrame"] {{ background: {CARD}; border-radius: 12px; }}
div[data-testid="stCaptionContainer"] {{ color: {MUTED}; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_data():
    return dd.load()


d = get_data()


def choice(label, options, default, fmt=str, key=None, hide_label=False):
    """Segmented control where available (Streamlit >= 1.40), horizontal radio otherwise."""
    vis = "collapsed" if hide_label else "visible"
    seg = getattr(st, "segmented_control", None)
    if seg is not None:
        v = seg(label, options, default=default, format_func=fmt, key=key, label_visibility=vis)
        return default if v is None else v
    return st.radio(label, options, index=list(options).index(default), format_func=fmt, horizontal=True, key=key,
                    label_visibility=vis)


def kpi(label, value, sub="", bg=INDIGO_SOFT, pill=None, pill_bg=None, pill_fg=None):
    p = f'<span class="nv-pill" style="background:{pill_bg};color:{pill_fg}">{pill}</span> ' if pill else ""
    return f'<div class="nv-kpi" style="background:{bg}"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{p}{sub}</div></div>'


def kpi_row(cards):
    return f'<div class="nv-kpis" style="grid-template-columns:repeat({len(cards)},minmax(0,1fr))">' + "".join(cards) + "</div>"


def chg_pill(x, unit=" pts"):
    up = x >= 0
    return dict(pill=f"{'▲' if up else '▼'} {abs(x):.2f}{unit}", pill_bg="#FFFFFF", pill_fg=CORAL if up else SKY)


# ------------------------------------------------------------------ sidebar navigation
PAGES = [("Today", ":material/home:"), ("Volatility surface", ":material/ssid_chart:"),
         ("Strike explorer", ":material/target:"), ("History", ":material/history:"),
         ("Track record", ":material/verified:"), ("Learn", ":material/school:")]
if st.session_state.get("page") not in [p for p, _ in PAGES]:
    st.session_state["page"] = PAGES[0][0]

with st.sidebar:
    st.markdown(f"""<div class="nv-brand">
      <svg width="30" height="30" viewBox="0 0 30 30" aria-hidden="true"><rect width="30" height="30" rx="9" fill="{INDIGO}"/>
      <path d="M6 19 C10 9, 13 9, 15 15 S20 21, 24 9" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round"/></svg>
      <b>NeuraVol</b></div>
      <div class="nv-tag">Neural forecasts of the NIFTY 50 implied-volatility surface</div>""", unsafe_allow_html=True)
    st.markdown('<div class="nv-section">Menu</div>', unsafe_allow_html=True)
    for name, icon in PAGES:
        active = st.session_state["page"] == name
        try:
            clicked = st.button(name, key=f"nav_{name}", icon=icon, type="primary" if active else "secondary",
                                use_container_width=True)
        except TypeError:   # older Streamlit without icon support
            clicked = st.button(name, key=f"nav_{name}", type="primary" if active else "secondary")
        if clicked and not active:
            st.session_state["page"] = name
            st.rerun()
    st.markdown('<div class="nv-section">Forecast horizon</div>', unsafe_allow_html=True)
    h = choice("Forecast horizon", dd.HORIZONS, 5, fmt=lambda x: {1: "1 day", 5: "1 week", 20: "1 month"}[x], key="horizon",
               hide_label=True)
    st.markdown(f'<div class="nv-section">Data</div><div class="nv-note"><b style="color:{INK}">As of {d.dates[-1]:%d %b %Y}</b>'
                f'<br>Source: NSE F&amp;O bhavcopy, updated after market close.<br><br>{DISCLAIMER}</div>', unsafe_allow_html=True)

page = st.session_state["page"]
H_LABEL = {1: "1 day", 5: "1 week", 20: "1 month"}[h]


def heatmap(df, title, colorscale, zmid=None, fmt=".1f", zmin=None, zmax=None):
    fig = go.Figure(go.Heatmap(z=df.values, x=df.columns, y=df.index, colorscale=colorscale, zmid=zmid,
                               zmin=zmin, zmax=zmax, texttemplate="%{z:" + fmt + "}", textfont=dict(color=INK, size=12),
                               hoverongaps=False, xgap=2, ygap=2, colorbar=dict(thickness=10, outlinewidth=0)))
    fig.update_layout(title=title, height=390, yaxis=dict(autorange="reversed"), **LAYOUT)
    fig.update_xaxes(title="moneyness ln(K/F)", showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# ------------------------------------------------------------------ page: today (beginner home)
ZONE_STYLE = {"ITM": (TEAL_SOFT, TEAL), "ATM": (INDIGO_SOFT, INDIGO), "OTM": (CORAL_SOFT, CORAL)}
PRICE_STYLE = {"low": (TEAL_SOFT, TEAL), "mid": (SKY_SOFT, SKY), "high": (CORAL_SOFT, CORAL)}
MOOD_BG = {"Calm": TEAL_SOFT, "Normal": SKY_SOFT, "Elevated": BUTTER_SOFT, "High fear": CORAL_SOFT}


def how_to_read(text):
    with st.expander("How to read this", icon=":material/help:") if _expander_icon_ok() else st.expander("How to read this"):
        st.markdown(text)


def _expander_icon_ok():
    import inspect
    try:
        return "icon" in inspect.signature(st.expander).parameters
    except (TypeError, ValueError):
        return False


if page == "Today":
    st.title("Today's options market")
    st.markdown(f'<div class="nv-sub">NIFTY 50 options, end of day {d.dates[-1]:%d %b %Y}. Plain-language summary for '
                'beginners and experts alike.</div>', unsafe_allow_html=True)
    mood = dd.market_mood(d)
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        pos = min(max(mood["pct"], 2), 98)
        st.markdown(f"""<div class="nv-kpi" style="background:{MOOD_BG[mood['mood']]};padding:20px 22px">
          <div class="l">Market mood (how expensive options are)</div>
          <div class="nv-mood">{mood['mood']}</div>
          <div class="s" style="color:{INK};font-size:0.9rem">{mood['text']}</div>
          <div class="nv-bar"><div class="nv-dot" style="left:{pos}%"></div></div>
          <div class="nv-scale"><span style="left:12%">Calm</span><span style="left:50%">Normal</span><span style="left:82%">Elevated</span><span style="left:95%">Fear</span></div>
          <div class="s" style="margin-top:12px">30-day implied volatility is <b style="color:{INK}">{mood['now']:.2f}%</b>,
          higher than on {mood['pct']:.0f}% of days in the past year (range {mood['low']:.1f}% to {mood['high']:.1f}%,
          average {mood['avg']:.1f}%).</div></div>""", unsafe_allow_html=True)
    with right:
        for title, text in dd.plain_summary(d, h):
            st.markdown(f'<div class="nv-sum"><div class="t">{title}</div><div class="x">{text}</div></div>',
                        unsafe_allow_html=True)

    st.subheader("Strike guide: ITM, ATM and OTM with today's prices")
    ex = dd.expiries(d)
    ex = ex[ex.days >= 7]
    labels = {i: f"{r.expiry:%d %b %Y} ({int(r.days)} days)" for i, r in ex.iterrows()}
    g0 = dd.guide_expiry(d)
    pick = st.selectbox("Expiry", list(labels), index=list(labels).index(g0.name), format_func=lambda i: labels[i],
                        help="Options expire on this date. Monthly expiries about a month away are the most traded.")
    row, g = dd.strike_guide(d, h, ex.loc[pick])
    F = float(row.F)
    st.markdown(f"""<div class="nv-def">
      <div style="background:{TEAL_SOFT}"><b>ITM, in the money</b><br>Would be worth something if it expired now.
      Call ITM = strike below NIFTY ({F:,.0f}); put ITM = strike above it. Costly, behaves like the index.</div>
      <div style="background:{INDIGO_SOFT}"><b>ATM, at the money</b><br>Strike closest to NIFTY's expected price
      ({F:,.0f}). Most affected by changes in volatility and by time decay.</div>
      <div style="background:{CORAL_SOFT}"><b>OTM, out of the money</b><br>Worth nothing if it expired now; needs a move
      to pay off. Cheap, but most expire worthless.</div></div>""", unsafe_allow_html=True)

    def guide_card(r):
        zbg, zfg = ZONE_STYLE[r.zone]
        pbg, pfg = PRICE_STYLE[r.pricing_level]
        last = f"₹{r.last_traded:,.2f}" if pd.notna(r.last_traded) else "no trade"
        return f"""<div class="nv-card">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <h4>{r.type} · {r.strike:,}</h4><span class="nv-zone" style="background:{zbg};color:{zfg}">{r.zone}</span></div>
          <div class="nv-row"><span>Fair price (model)</span><b>₹{r.premium:,.2f}</b></div>
          <div class="nv-row"><span>Last traded on NSE</span><b>{last}</b></div>
          <div class="nv-row"><span>Implied volatility</span><b>{r.iv:.2f}%</b></div>
          <div class="nv-row"><span>Price level vs past year</span>
            <span class="nv-zone" style="background:{pbg};color:{pfg}">{r.pricing}</span></div>
          <div class="nv-row"><span>Chance to finish in the money*</span><b>{r.prob_itm:.0f}%</b></div>
          <div class="nv-row"><span>Moves per ₹1 NIFTY move (delta)</span><b>{r.delta:+.2f}</b></div>
          <div class="nv-row"><span>Expected IV change, {H_LABEL}</span><b>{r.iv_change:+.2f} pts</b></div>
          <div class="nv-what">{r.what}</div></div>"""

    cc, cp = st.columns(2, gap="large")
    with cc:
        st.markdown("**Calls** · profit if NIFTY rises")
        st.markdown("".join(guide_card(r) for r in g[g.type == "Call"].itertuples()), unsafe_allow_html=True)
    with cp:
        st.markdown("**Puts** · profit if NIFTY falls")
        st.markdown("".join(guide_card(r) for r in g[g.type == "Put"].itertuples()), unsafe_allow_html=True)
    st.markdown(f"""<div class="nv-verdict"><b>Why there is no "best strike" here:</b> the right choice depends on your view
      of the market, your budget and how much risk you can take, and suggesting specific trades requires SEBI registration.
      This guide shows the facts for each type so you can decide, or discuss with a SEBI-registered adviser.</div>""",
                unsafe_allow_html=True)
    st.caption("*Chance to finish in the money is the probability implied by today's option prices (risk-neutral), not a "
               "prediction of what NIFTY will do. Strikes are examples about 3% from the expected price, rounded to 50. "
               + DISCLAIMER)

# ------------------------------------------------------------------ page: surface & forecast
elif page == "Volatility surface":
    st.title("NIFTY implied volatility")
    k = dd.kpis(d, h)
    st.markdown(f'<div class="nv-sub">End of day {k["as_of"]:%d %b %Y} · all volatilities in % per year</div>', unsafe_allow_html=True)
    st.markdown(kpi_row([
        kpi("NIFTY near forward", f"{k['forward']:,.1f}", "vs previous day", INDIGO_SOFT, **chg_pill(k['forward_chg'] * 100, "%")),
        kpi("ATM IV · 30 days", f"{k['atm30']:.2f}%", "vs previous day", TEAL_SOFT, **chg_pill(k['atm30_chg'])),
        kpi("Put skew · 30 days", f"{k['skew30']:+.2f} pts", "5% OTM put minus call", CORAL_SOFT),
        kpi("Term slope · 7d to 91d", f"{k['term']:+.2f} pts", "negative = stress", SKY_SOFT),
        kpi(f"Forecast ATM 30d · {H_LABEL}", f"{k['fc_atm30']:.2f}%", "expected change", BUTTER_SOFT, **chg_pill(k['fc_chg'])),
    ]), unsafe_allow_html=True)

    today, mask = dd.surface_today(d)
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        view = choice("Surface view", ["Heatmap", "3D"], "Heatmap", key="view")
        if view == "Heatmap":
            st.plotly_chart(heatmap(dd.grid_frame(d, today, mask), "Today's surface (blank = extrapolated)", SEQ))
        else:
            fig = go.Figure(go.Surface(z=today * 100, x=d.mon * 100, y=d.tenors, colorscale=SEQ,
                                       colorbar=dict(thickness=10, outlinewidth=0)))
            fig.update_layout(title="Today's surface", height=430,
                              scene=dict(xaxis_title="ln(K/F) %", yaxis_title="days", zaxis_title="IV %"), **LAYOUT)
            st.plotly_chart(fig)
    with right:
        tenor = choice("Smile for maturity", list(d.tenors), 30, fmt=lambda x: f"{x}d", key="tenor")
        sm = dd.smile(d, tenor, h)
        fig = go.Figure()
        fig.add_trace(go.Scatter(mode="lines", x=sm.moneyness, y=sm.hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(mode="lines", x=sm.moneyness, y=sm.lo, fill="tonexty", fillcolor="rgba(232,128,111,0.14)",
                                 line=dict(width=0), name="80% band (normal markets)"))
        fig.add_trace(go.Scatter(mode="lines", x=sm.moneyness, y=sm.today, name="today", line=dict(color=INDIGO, width=3)))
        fig.add_trace(go.Scatter(mode="lines", x=sm.moneyness, y=sm.forecast, name=f"forecast, {H_LABEL}",
                                 line=dict(color=CORAL, width=3, dash="dash")))
        fig.update_layout(title=f"{tenor}-day smile: today vs forecast", height=390, xaxis_title="moneyness %",
                          yaxis_title="IV %", legend=dict(orientation="h", y=-0.25), **LAYOUT)
        st.plotly_chart(fig)

    how_to_read("""**Heatmap (left):** each box is the implied volatility of one kind of option. Rows are time to
expiry (7 days at the top, 1 year at the bottom); columns are how far the strike is from NIFTY (left = puts below
the market, right = calls above it). Darker = more expensive. Blank boxes have no traded options, so we estimate them.

**Smile (right):** the same thing for one expiry as a line. It usually slopes down from left to right because
protection against a fall (puts) costs more. The blue line is today, the dashed coral line is the forecast, and the
shaded band shows the usual size of the forecast error.""")
    st.subheader(f"Expected change over {H_LABEL}")
    delta = dd.forecast_grid(d, h) - today
    lim = max(0.5, float(np.nanmax(np.abs(delta * 100))))
    st.plotly_chart(heatmap(dd.grid_frame(d, delta), "Forecast minus today, vol points (coral = rise, blue = fall)",
                            DIV, zmid=0, fmt=".2f", zmin=-lim, zmax=lim))
    how_to_read("""Each box shows how much the model expects that option type's implied volatility to change.
Coral = expected to get pricier, blue = expected to get cheaper, white = little change. Numbers are in volatility
points (e.g. +0.50 means 14.0% becoming 14.5%).""")
    st.caption("Forecast = average of the NN v2 model and a PCA-VAR factor model, the most accurate combination in "
               "out-of-sample tests 2020-2026 (see Track record). Bands use typical out-of-sample error outside the "
               "2020 crash; in a crash, errors were 2-4x larger.")

# ------------------------------------------------------------------ page: strike explorer
elif page == "Strike explorer":
    st.title("Strike explorer")
    st.markdown('<div class="nv-sub">Pick a listed contract to see its implied vol today, the forecast for it, and what that '
                'means for its premium. Analysis only, not a trade recommendation.</div>', unsafe_allow_html=True)
    ex = dd.expiries(d)
    c1, c2, c3 = st.columns(3)
    labels = {i: f"{r.expiry:%d %b %Y} ({int(r.days)} days)" for i, r in ex.iterrows()}
    default = int(np.argmin(np.abs(ex.days.values - 30)))
    pick = c1.selectbox("Expiry", list(labels), index=default, format_func=lambda i: labels[i])
    row = ex.loc[pick]
    atm = int(round(row.F / 50) * 50)
    K = c2.number_input("Strike", min_value=int(row.F * 0.7) // 50 * 50, max_value=int(row.F * 1.3) // 50 * 50,
                        value=atm, step=50)
    with c3:
        typ = choice("Type", ["Put (PE)", "Call (CE)"], "Put (PE)", key="opt_type")
    v = dd.strike_view(d, row, float(K), typ.startswith("Call"), h)
    last = f"last traded ₹{v['last_traded']:,.2f}" if v["last_traded"] is not None else "no trade today"
    st.markdown(kpi_row([
        kpi("Implied vol today", f"{v['iv_now']:.2f}%", f"{v['days']} days to expiry", INDIGO_SOFT),
        kpi(f"Forecast IV · {H_LABEL}", f"{v['iv_fc']:.2f}%", "expected change", BUTTER_SOFT, **chg_pill(v['change'])),
        kpi("Model premium today", f"₹{v['model_price_now']:,.2f}", last, TEAL_SOFT),
        kpi("Premium if IV moves as forecast", f"₹{v['range_lo']:,.0f} – ₹{v['range_hi']:,.0f}", "80% normal-market band", CORAL_SOFT),
    ]), unsafe_allow_html=True)
    where = ("below" if v["last_traded"] and v["last_traded"] < v["range_lo"] else
             "above" if v["last_traded"] and v["last_traded"] > v["range_hi"] else "inside")
    st.markdown(f'<div class="nv-callout">Strike <b>{K:,}</b> is {v["moneyness"]:+.1f}% from the forward '
                f'({v["forward"]:,.1f}), {v["days"]} days to expiry. The model expects this option\'s implied vol to move '
                f'<b>{v["change"]:+.2f} points</b> over the next {H_LABEL}. '
                + (f"Today's last traded premium sits <b>{where}</b> the model's range. " if v["last_traded"] else "")
                + "Analysis only, not a recommendation to buy or sell.</div>", unsafe_allow_html=True)
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
    for col, colr in zip(["ATM 7d", "ATM 30d", "ATM 91d"], [CORAL, INDIGO, TEAL]):
        fig.add_trace(go.Scatter(mode="lines", x=hs.index, y=hs[col], name=col, line=dict(width=1.3, color=colr)))
    fig.update_layout(title="At-the-money implied vol (%)", height=400, legend=dict(orientation="h", y=1.08, x=1, xanchor="right"), **LAYOUT)
    st.plotly_chart(fig)
    c1, c2 = st.columns(2, gap="large")
    for box, col, colr in zip((c1, c2), ["Skew 30d (-5% minus +5%)", "Term slope (91d minus 7d)"], [CORAL, SKY]):
        f = go.Figure(go.Scatter(mode="lines", x=hs.index, y=hs[col], line=dict(width=1.1, color=colr)))
        f.add_hline(y=0, line_color=LINE, line_width=1)
        f.update_layout(title=col + ", vol points", height=320, **LAYOUT)
        box.plotly_chart(f)
    how_to_read("""**Top chart:** how expensive at-the-money options were each day, for 7-day, 30-day and 91-day
expiries. Spikes are fear: the COVID crash in March 2020 and the 2019 and 2024 election results.

**Skew:** how much more crash protection (puts) cost than calls. Higher = more fear of a fall.

**Term slope:** long-dated minus short-dated volatility. Below zero means short-term fear is unusually high.""")
    st.caption("Visible events: COVID crash (Mar 2020), general elections (May 2019, Jun 2024). "
               "Negative term slope = short-dated fear above long-dated.")

# ------------------------------------------------------------------ page: track record
elif page == "Track record":
    st.title("Track record")
    st.markdown('<div class="nv-sub">Walk-forward test: for each year 2020-2026 the model was trained only on earlier data, '
                'then forecast every day of that year. Error = RMSE in vol points on cells backed by real quotes. '
                'Lower is better; the best model in each row is highlighted.</div>', unsafe_allow_html=True)
    t_all, t_calm = dd.track_table(d), dd.track_table(d, exclude_covid=True)
    order = ["No change", "PCA-VAR", "NN v2", dd.BLEND]
    fmt = lambda t: (t[order].rename(index=lambda x: f"{x} day" + ("s" if x > 1 else "")).rename_axis("horizon")
                     .rename(columns={dd.BLEND: "NeuraVol"}))
    c1, c2 = st.columns(2, gap="large")
    c1.markdown("**All test days**")
    c1.dataframe(fmt(t_all).style.format("{:.2f}").highlight_min(axis=1, color=HILITE))
    c2.markdown("**Excluding the Feb-Jun 2020 crash**")
    c2.dataframe(fmt(t_calm).style.format("{:.2f}").highlight_min(axis=1, color=HILITE))
    sig = d.significance.get(str(h), {})
    p_rw = sig.get("NN v2 shrunk + PCA-VAR avg vs No change", [np.nan, np.nan])[1]
    st.markdown(f'<div class="nv-verdict"><b>Honest verdict ({H_LABEL}):</b> NeuraVol has the lowest error, but the '
                f"Diebold-Mariano test vs 'no change' gives p = {p_rw:.2f}, so the improvement is <b>not statistically "
                "significant</b> yet.</div>", unsafe_allow_html=True)

    how_to_read("""Each number is the typical forecast error in volatility points: lower is better. "No change"
simply assumes tomorrow looks like today, a tough benchmark because volatility moves slowly. PCA-VAR is a classic
statistical model. NeuraVol is our model. The highlighted cell is the most accurate in each row.""")
    a = d.atm30[d.atm30.horizon_days == h]
    fig = go.Figure()
    fig.add_trace(go.Scatter(mode="lines", x=a.target_date, y=a.actual * 100, name="actual", line=dict(color=INDIGO, width=1.4)))
    fig.add_trace(go.Scatter(mode="lines", x=a.target_date, y=a[dd.BLEND] * 100, name=f"NeuraVol forecast made {H_LABEL} earlier",
                             line=dict(color=CORAL, width=1.2)))
    fig.update_layout(title="30-day ATM implied vol: forecast vs actual (%)", height=380,
                      legend=dict(orientation="h", y=-0.2), **LAYOUT)
    st.plotly_chart(fig)

    st.markdown(f"**Error by year, {H_LABEL} horizon**")
    st.dataframe(dd.track_by_year(d, h)[order].rename(columns={dd.BLEND: "NeuraVol"}).style.format("{:.2f}")
                 .highlight_min(axis=1, color=HILITE))

    st.subheader("Does it make money? Delta-hedged straddle backtest")
    st.caption("Buy (sell) an ATM straddle when the forecast says vol will rise (fall), hedge daily with futures, "
               "0.3% cost per option trade. P&L in vol points. Includes an oracle that knows the future (upper bound).")
    bt = d.bt_summary.rename(columns={"hold_days": "hold (days)", "hit_rate": "hit rate", "sharpe_annualised": "Sharpe",
                                      "total_pnl_volpts": "total P&L (vol pts)", "max_drawdown_volpts": "max drawdown"})
    st.dataframe(bt[["hold (days)", "strategy", "trades", "hit rate", "Sharpe", "total P&L (vol pts)", "max drawdown"]]
                 .round(2), hide_index=True)
    st.markdown('<div class="nv-verdict"><b>Result:</b> the forecasts are more accurate than the baselines but do '
                '<b>not</b> produce trading profit after costs; simply selling volatility did better. The edge is too '
                'small to pay for trading costs.</div>', unsafe_allow_html=True)

    live = dd.live_track_record(d)
    st.subheader("Live forecasts since launch")
    if live.empty:
        st.caption("Forecasts are logged every day by the pipeline and scored here once the target day arrives.")
    else:
        st.dataframe(live.pivot_table(index="horizon_days", columns="model", values="rmse", aggfunc="mean").round(2))

# ------------------------------------------------------------------ page: learn
else:
    st.title("Learn: options and volatility in 5 minutes")
    st.markdown('<div class="nv-sub">New to options? Start here. Each section takes about a minute.</div>', unsafe_allow_html=True)
    k = dd.kpis(d, h)
    F = k["forward"]
    with st.expander("1. What is an option?", expanded=True):
        st.markdown(f"""An option is a contract that gives you the **right, but not the duty**, to buy or sell NIFTY at a fixed
price (the **strike**) on a fixed date (the **expiry**).

- A **call** gains when NIFTY rises above the strike.
- A **put** gains when NIFTY falls below the strike.
- The price you pay for the option is the **premium**. A buyer can lose at most the premium; a seller collects it
  but can lose much more.""")
    with st.expander("2. ITM, ATM and OTM"):
        st.markdown(f"""Today NIFTY's expected price is about **{F:,.0f}**.

| | Call | Put |
|---|---|---|
| **ITM** (in the money) | strike below {F:,.0f}, e.g. {int(round(F*0.97/50)*50):,} | strike above, e.g. {int(round(F*1.03/50)*50):,} |
| **ATM** (at the money) | strike near {int(round(F/50)*50):,} | strike near {int(round(F/50)*50):,} |
| **OTM** (out of the money) | strike above, e.g. {int(round(F*1.03/50)*50):,} | strike below, e.g. {int(round(F*0.97/50)*50):,} |

ITM options cost more and behave like the index; OTM options are cheap but usually expire worthless; ATM options are in
between and react most to changes in volatility. See them with today's prices on the **Today** page.""")
    with st.expander("3. What is implied volatility (IV)?"):
        st.markdown(f"""IV is the market's guess of how much NIFTY will move, written as a % per year and worked out from
option prices. Higher IV = more expensive options.

Today 30-day IV is **{k['atm30']:.2f}%**. Roughly, that means the market expects NIFTY to move about
**±{k['atm30']/np.sqrt(12):.1f}%** over the next month (IV divided by the square root of 12). In March 2020 it was above 80%.""")
    with st.expander("4. Smile, skew and term structure"):
        st.markdown("""- **Smile / skew:** IV is different for every strike. Puts below the market usually have higher IV
  because people pay extra for crash protection.
- **Term structure:** IV is different for every expiry. Normally long-dated IV is a bit higher. When short-dated IV
  jumps above it, the market is nervous about something soon.
- Put all strikes and expiries together and you get the **volatility surface** (the heatmap and 3D chart).""")
    with st.expander("5. What does NeuraVol forecast?"):
        st.markdown("""It forecasts how the whole volatility surface will look **1 day, 1 week and 1 month ahead**: whether
options will get pricier or cheaper. It does **not** forecast whether NIFTY goes up or down.

Use the horizon buttons in the sidebar to switch between 1 day, 1 week and 1 month. The Track record page shows how
accurate it has been, including where it did not beat simple methods.""")
    with st.expander("6. Glossary"):
        st.markdown("""- **Strike:** the fixed price in the option contract.
- **Expiry:** the date the option ends.
- **Premium:** the option's price.
- **Delta:** how much the option price moves when NIFTY moves ₹1.
- **Theta / time decay:** options lose value every day as expiry approaches.
- **Vol point:** one percentage point of IV (14% to 15% = +1 vol point).
- **Forward:** NIFTY's expected price at expiry, slightly above today's level because of interest.""")
    st.subheader("About NeuraVol")
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
