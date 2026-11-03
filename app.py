"""NeuraVol: NIFTY implied-volatility surface and forecast dashboard.

    pip install -r requirements.txt
    streamlit run app.py
"""
import time as _time
_T0 = _time.perf_counter()
_MARKS = []


def _mark(name):
    _MARKS.append((name, _time.perf_counter() - _T0))


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dashboard_data as dd
from src.pricing import black76
_mark('imports')


def _chart(fig, **k):
    _mark('chart build')
    st.plotly_chart(fig, **k)
    _mark('chart sent')

st.set_page_config(page_title="NeuraVol · NIFTY volatility intelligence", page_icon="📈", layout="wide")

# ------------------------------------------------------------------ palette
INK, MUTED, LINE, BG, CARD = "#1C1B33", "#5E5C78", "#E6E4F2", "#F6F5FB", "#FFFFFF"
SIDE, SIDE_TXT, SIDE_MUTED = "#231E4D", "#E4E1FA", "#A49FD0"
VIOLET, VIOLET_SOFT = "#6D5DFC", "#EEEBFF"
TEAL, TEAL_SOFT = "#0E9F90", "#E2F6F3"
CORAL, CORAL_SOFT = "#E8566C", "#FFE8EC"
AMBER, AMBER_SOFT = "#D98A0B", "#FFF3DC"
SKY, SKY_SOFT = "#2B93D6", "#E2F1FB"
SEQ = [[0, "#F3F1FF"], [0.5, "#C6BDFF"], [1, "#8E80FF"]]
DIV = [[0, "#8FCFF4"], [0.5, "#FFFFFF"], [1, "#F7A1AE"]]
HILITE = "#DDF4F0"
LAYOUT = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              margin=dict(l=10, r=10, t=50, b=10), font=dict(size=13, color=INK, family="Poppins, sans-serif"),
              title_font=dict(size=15, color=INK), hoverlabel=dict(font_family="Poppins, sans-serif"))
DISCLAIMER = ("NeuraVol provides research and analysis of implied volatility. It does not provide buy or sell "
              "recommendations and is not investment advice. Past accuracy does not guarantee future results.")
H_NAMES = {1: "1 day", 5: "1 week", 20: "1 month"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0..1,0&display=block');
html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not([class*="material"]) {{ font-family: 'Poppins', sans-serif; }}
.stApp {{ background: {BG}; color: {INK}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1320px; }}
h1 {{ font-weight: 700; color: {INK}; letter-spacing: -0.02em; }}
h2, h3 {{ color: {INK}; font-weight: 600; letter-spacing: -0.01em; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: {SIDE}; }}
[data-testid="stSidebar"] * {{ color: {SIDE_TXT}; }}
[data-testid="stSidebar"] .stButton > button {{
  width: 100%; justify-content: flex-start; text-align: left; border: 1px solid transparent;
  background: transparent; border-radius: 12px; padding: 0.5rem 0.85rem; min-height: 44px; font-weight: 500; box-shadow: none; }}
[data-testid="stSidebar"] .stButton > button > div {{ justify-content: flex-start; }}
[data-testid="stSidebar"] .stButton > button:hover {{ background: rgba(255,255,255,0.08); border-color: transparent; }}
[data-testid="stSidebar"] .stButton > button:hover * {{ color: #FFFFFF; }}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {{ background: {VIOLET}; border-color: {VIOLET}; }}
[data-testid="stSidebar"] .stButton > button[kind="primary"] *,
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] * {{ color: #FFFFFF; }}
[data-testid="stSidebar"] [class*="st-key-hz_"] button {{ justify-content: center !important; min-height: 38px !important;
  padding: 0.3rem 0.2rem !important; border: 1px solid rgba(255,255,255,0.2) !important; }}
[data-testid="stSidebar"] [class*="st-key-hz_"] button > div {{ justify-content: center !important; }}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{ gap: 0.4rem; }}
.nv-brand {{ display: flex; align-items: center; gap: 10px; margin: 0.1rem 0 0.2rem; }}
.nv-brand b {{ font-size: 1.3rem; font-weight: 700; color: #FFFFFF; }}
.nv-tag {{ color: {SIDE_MUTED}; font-size: 0.82rem; line-height: 1.45; margin-bottom: 1rem; }}
.nv-section {{ font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: {SIDE_MUTED} !important;
  margin: 1.1rem 0 0.4rem 0.3rem; }}
.nv-note {{ background: rgba(255,255,255,0.06); border-radius: 12px; padding: 0.75rem 0.85rem; font-size: 0.78rem; line-height: 1.55; }}
.nv-note, .nv-note * {{ color: {SIDE_MUTED} !important; }}
.nv-note b {{ color: #FFFFFF !important; }}

/* content */
.nv-sub {{ color: {MUTED}; font-size: 0.92rem; margin-top: -0.4rem; margin-bottom: 1rem; }}
.nv-kpis {{ display: grid; gap: 14px; margin: 0.4rem 0 1.3rem; }}
@media (max-width: 1100px) {{ .nv-kpis {{ grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)) !important; }} }}
.nv-kpi {{ border-radius: 16px; padding: 16px 18px; background: {CARD}; border: 1px solid {LINE}; position: relative; overflow: hidden; }}
.nv-kpi::before {{ content: ""; position: absolute; left: 0; top: 0; right: 0; height: 4px; background: var(--acc); }}
.nv-kpi .l {{ font-size: 0.78rem; color: {MUTED}; font-weight: 500; }}
.nv-kpi .v {{ font-size: 1.55rem; font-weight: 700; color: {INK}; margin-top: 4px; letter-spacing: -0.02em; white-space: nowrap; }}
.nv-kpi .s {{ font-size: 0.78rem; margin-top: 6px; color: {MUTED}; }}
.nv-pill {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-weight: 600; font-size: 0.76rem; }}
.nv-panel {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 18px 20px; margin-bottom: 12px; }}
.nv-panel h4 {{ margin: 0 0 6px 0; font-size: 1rem; color: {INK}; }}
.nv-insight {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 14px 16px; margin-bottom: 10px; display: flex; gap: 12px; }}
.nv-insight .ic {{ flex: 0 0 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 700; }}
.nv-insight .t {{ font-size: 0.74rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: {MUTED}; }}
.nv-insight .x {{ font-size: 0.9rem; color: {INK}; line-height: 1.55; margin-top: 2px; }}
.nv-regime {{ border-radius: 18px; padding: 20px 22px; color: {INK}; }}
.nv-regime .big {{ font-size: 2rem; font-weight: 700; margin: 2px 0 4px; }}
.nv-bar {{ position: relative; height: 10px; border-radius: 999px; margin: 16px 0 6px;
  background: linear-gradient(90deg,#9EE0D6 0 25%,#A9D3F3 25% 75%,#FAD58A 75% 90%,#F6A3B0 90% 100%); }}
.nv-dot {{ position: absolute; top: -6px; width: 22px; height: 22px; border-radius: 50%; background: {INK}; border: 4px solid #fff;
  transform: translateX(-50%); box-sizing: border-box; }}
.nv-scale {{ position: relative; height: 16px; font-size: 0.72rem; color: {MUTED}; }}
.nv-scale span {{ position: absolute; transform: translateX(-50%); white-space: nowrap; }}
.nv-defs {{ display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 12px; margin: 4px 0 16px; }}
.nv-defs div {{ border-radius: 14px; padding: 12px 14px; font-size: 0.85rem; line-height: 1.5; color: {INK}; }}
.nv-card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 16px 18px; margin-bottom: 12px; }}
.nv-card h4 {{ margin: 0; font-size: 1.02rem; color: {INK}; }}
.nv-row {{ display: flex; justify-content: space-between; gap: 10px; font-size: 0.86rem; padding: 3px 0; color: {MUTED}; }}
.nv-row b {{ color: {INK}; font-weight: 600; }}
.nv-what {{ font-size: 0.83rem; color: {MUTED}; margin-top: 8px; line-height: 1.5; border-top: 1px dashed {LINE}; padding-top: 8px; }}
.nv-notice {{ background: {VIOLET_SOFT}; border-radius: 14px; padding: 13px 16px; color: {INK}; font-size: 0.86rem; line-height: 1.6; margin: 0.6rem 0 1rem; }}
.nv-step {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 16px 18px; height: 100%; }}
.nv-step .n {{ width: 30px; height: 30px; border-radius: 9px; background: {VIOLET}; color: #fff; display: flex; align-items: center;
  justify-content: center; font-weight: 700; margin-bottom: 8px; }}
.nv-step b {{ color: {INK}; }}
.nv-step p {{ color: {MUTED}; font-size: 0.86rem; line-height: 1.55; margin: 4px 0 0; }}
[data-testid="stPlotlyChart"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 6px 8px; }}
[data-testid="stExpander"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 10px; padding: 6px 14px; }}
.stTabs [aria-selected="true"] {{ background: {VIOLET_SOFT}; border-color: {VIOLET}; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_data():
    return dd.load()


d = get_data()
_mark('data')
K0 = dd.kpis(d, 5)


# ------------------------------------------------------------------ small UI helpers
def choice(label, options, default, fmt=str, key=None, hide_label=False):
    vis = "collapsed" if hide_label else "visible"
    seg = getattr(st, "segmented_control", None)
    if seg is not None:
        v = seg(label, options, default=default, format_func=fmt, key=key, label_visibility=vis)
        return default if v is None else v
    return st.radio(label, options, index=list(options).index(default), format_func=fmt, horizontal=True, key=key,
                    label_visibility=vis)


def kpi(label, value, sub="", acc=VIOLET, pill=None, pill_bg=None, pill_fg=None):
    p = f'<span class="nv-pill" style="background:{pill_bg};color:{pill_fg}">{pill}</span> ' if pill else ""
    return (f'<div class="nv-kpi" style="--acc:{acc}"><div class="l">{label}</div><div class="v">{value}</div>'
            f'<div class="s">{p}{sub}</div></div>')


def kpi_row(cards):
    return f'<div class="nv-kpis" style="grid-template-columns:repeat({len(cards)},minmax(0,1fr))">' + "".join(cards) + "</div>"


def chg(x, unit=" pts"):
    up = x >= 0
    return dict(pill=f"{'▲' if up else '▼'} {abs(x):.2f}{unit}", pill_bg=CORAL_SOFT if up else SKY_SOFT,
                pill_fg=CORAL if up else SKY)


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def nav_button(label, key, active, icon=None):
    try:
        return st.button(label, key=key, icon=icon, type="primary" if active else "secondary", use_container_width=True)
    except TypeError:
        return st.button(label, key=key, type="primary" if active else "secondary")


def heatmap(df, title, colorscale, zmid=None, fmt=".1f", zmin=None, zmax=None, height=400):
    fig = go.Figure(go.Heatmap(z=df.values, x=df.columns, y=df.index, colorscale=colorscale, zmid=zmid, zmin=zmin,
                               zmax=zmax, texttemplate="%{z:" + fmt + "}", textfont=dict(color=INK, size=12),
                               hovertemplate="maturity %{y}<br>strike %{x} from forward<br>%{z:.2f}<extra></extra>",
                               hoverongaps=False, xgap=3, ygap=3, colorbar=dict(thickness=10, outlinewidth=0)))
    fig.update_layout(title=title, height=height, yaxis=dict(autorange="reversed"), **LAYOUT)
    fig.update_xaxes(title="strike distance from forward", showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# ------------------------------------------------------------------ sidebar
PAGES = [("Overview", ":material/space_dashboard:"), ("Volatility surface", ":material/ssid_chart:"),
         ("Strike explorer", ":material/tune:"), ("History", ":material/timeline:"),
         ("Track record", ":material/verified:"), ("User guide", ":material/menu_book:"),
         ("Admin", ":material/admin_panel_settings:")]
if st.session_state.get("page") not in [p for p, _ in PAGES]:
    st.session_state["page"] = PAGES[0][0]
if st.session_state.get("h") not in dd.HORIZONS:
    st.session_state["h"] = 5

with st.sidebar:
    html(f"""<div class="nv-brand">
      <svg width="32" height="32" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="10" fill="{VIOLET}"/>
      <path d="M6 20 C10 9, 14 9, 16 16 S22 23, 26 10" fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round"/></svg>
      <b>NeuraVol</b></div><div class="nv-tag">NIFTY 50 volatility intelligence</div>""")
    html('<div class="nv-section">Navigate</div>')
    for name, icon in PAGES:
        active = st.session_state["page"] == name
        if nav_button(name, f"nav_{name}", active, icon) and not active:
            st.session_state["page"] = name
            st.rerun()
    html('<div class="nv-section">Forecast horizon</div>')
    cols = st.columns(3)
    for c, hz in zip(cols, dd.HORIZONS):
        with c:
            if nav_button({1: "1D", 5: "1W", 20: "1M"}[hz], f"hz_{hz}", st.session_state["h"] == hz) and st.session_state["h"] != hz:
                st.session_state["h"] = hz
                st.rerun()
    html(f'<div class="nv-section">Data</div><div class="nv-note"><b>Updated {d.dates[-1]:%d %b %Y}</b><br>'
         f'Source: NSE F&amp;O end-of-day data, refreshed every trading evening.<br><br>{DISCLAIMER}</div>')

_mark('sidebar')
page = st.session_state["page"]
h = st.session_state["h"]
HL = H_NAMES[h]

ZONE = {"ITM": (TEAL_SOFT, TEAL), "ATM": (VIOLET_SOFT, VIOLET), "OTM": (CORAL_SOFT, CORAL)}
PRICE = {"low": (TEAL_SOFT, TEAL), "mid": (SKY_SOFT, SKY), "high": (CORAL_SOFT, CORAL)}
REGIME_BG = {"Calm": TEAL_SOFT, "Normal": SKY_SOFT, "Elevated": AMBER_SOFT, "High fear": CORAL_SOFT}
INSIGHT_STYLE = {"Forecast": ("↗", VIOLET_SOFT, VIOLET), "Crash protection": ("⛨", CORAL_SOFT, CORAL),
                 "Short vs long term": ("⇆", TEAL_SOFT, TEAL)}
INSIGHT_TITLE = {"Forecast": "Volatility outlook", "Crash protection": "Downside protection",
                 "Short vs long term": "Term structure"}

# ================================================================== OVERVIEW
if page == "Overview":
    st.title("Market overview")
    html(f'<div class="nv-sub">NIFTY 50 options · end of day {d.dates[-1]:%d %b %Y} · forecast horizon: {HL}</div>')
    k = dd.kpis(d, h)
    html(kpi_row([
        kpi("NIFTY forward", f"{k['forward']:,.1f}", "vs previous session", VIOLET, **chg(k['forward_chg'] * 100, "%")),
        kpi("30-day ATM volatility", f"{k['atm30']:.2f}%", "vs previous session", TEAL, **chg(k['atm30_chg'])),
        kpi(f"Forecast, {HL}", f"{k['fc_atm30']:.2f}%", "30-day ATM volatility", CORAL, **chg(k['fc_chg'])),
        kpi("Downside skew", f"{k['skew30']:+.2f} pts", "5% OTM put minus 5% OTM call", AMBER),
    ]))
    mood = dd.market_mood(d)
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        pos = min(max(mood["pct"], 3), 97)
        html(f"""<div class="nv-regime" style="background:{REGIME_BG[mood['mood']]}">
          <div style="font-size:0.8rem;color:{MUTED};font-weight:500">Volatility regime</div>
          <div class="big">{mood['mood']}</div>
          <div style="font-size:0.9rem;line-height:1.55">{mood['text']}</div>
          <div class="nv-bar"><div class="nv-dot" style="left:{pos}%"></div></div>
          <div class="nv-scale"><span style="left:12%">Calm</span><span style="left:50%">Normal</span>
          <span style="left:82%">Elevated</span><span style="left:95%">High</span></div>
          <div style="font-size:0.82rem;color:{MUTED};margin-top:12px;line-height:1.55">30-day volatility of
          <b style="color:{INK}">{mood['now']:.2f}%</b> is above {mood['pct']:.0f}% of sessions in the past year
          (range {mood['low']:.1f}% to {mood['high']:.1f}%, average {mood['avg']:.1f}%).</div></div>""")
    with right:
        for title, text in dd.plain_summary(d, h):
            ic, bg, fg = INSIGHT_STYLE[title]
            html(f'<div class="nv-insight"><div class="ic" style="background:{bg};color:{fg}">{ic}</div><div>'
                 f'<div class="t">{INSIGHT_TITLE[title]}</div><div class="x">{text}</div></div></div>')

    st.subheader("Strike guide")
    html('<div class="nv-sub">Representative in-the-money, at-the-money and out-of-the-money contracts for the selected '
         'expiry, with fair value, live market price and how today\'s pricing compares with the past year.</div>')
    ex = dd.expiries(d)
    ex = ex[ex.days >= 7]
    labels = {i: f"{r.expiry:%d %b %Y} · {int(r.days)} days" for i, r in ex.iterrows()}
    g0 = dd.guide_expiry(d)
    pick = st.selectbox("Expiry", list(labels), index=list(labels).index(g0.name), format_func=lambda i: labels[i])
    row, g = dd.strike_guide(d, h, ex.loc[pick])
    F = float(row.F)
    html(f"""<div class="nv-defs">
      <div style="background:{TEAL_SOFT}"><b>In the money (ITM)</b><br>Has intrinsic value today. Higher premium;
      price moves closely with NIFTY.</div>
      <div style="background:{VIOLET_SOFT}"><b>At the money (ATM)</b><br>Strike nearest the forward ({F:,.0f}). Most
      sensitive to volatility and to time decay.</div>
      <div style="background:{CORAL_SOFT}"><b>Out of the money (OTM)</b><br>No intrinsic value today. Lowest premium;
      pays off only after a sizeable move.</div></div>""")

    def guide_card(r):
        zb, zf = ZONE[r.zone]
        pb, pf = PRICE[r.pricing_level]
        last = f"₹{r.last_traded:,.2f}" if pd.notna(r.last_traded) else "not traded"
        return f"""<div class="nv-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <h4>{r.type} {r.strike:,}</h4><span class="nv-pill" style="background:{zb};color:{zf}">{r.zone}</span></div>
          <div class="nv-row"><span>Fair value</span><b>₹{r.premium:,.2f}</b></div>
          <div class="nv-row"><span>Last traded price</span><b>{last}</b></div>
          <div class="nv-row"><span>Implied volatility</span><b>{r.iv:.2f}%</b></div>
          <div class="nv-row"><span>Pricing vs past year</span><span class="nv-pill" style="background:{pb};color:{pf}">{r.pricing}</span></div>
          <div class="nv-row"><span>Market-implied chance to expire ITM</span><b>{r.prob_itm:.0f}%</b></div>
          <div class="nv-row"><span>Delta</span><b>{r.delta:+.2f}</b></div>
          <div class="nv-row"><span>Expected volatility change, {HL}</span><b>{r.iv_change:+.2f} pts</b></div>
          <div class="nv-what">{r.what}</div></div>"""

    cc, cp = st.columns(2, gap="large")
    with cc:
        st.markdown("**Calls** · gain when NIFTY rises")
        html("".join(guide_card(r) for r in g[g.type == "Call"].itertuples()))
    with cp:
        st.markdown("**Puts** · gain when NIFTY falls")
        html("".join(guide_card(r) for r in g[g.type == "Put"].itertuples()))

    st.subheader("Scenario: profit or loss at expiry")
    html('<div class="nv-sub">Move the slider to see what each contract above would be worth at expiry if NIFTY ends '
         'at that level. Values are per unit of NIFTY, after the premium paid.</div>')
    move = st.slider("NIFTY at expiry, change from today's forward", -10.0, 10.0, 0.0, 0.5, format="%+.1f%%")
    ST = F * (1 + move / 100)
    grid_m = np.linspace(-10, 10, 161)
    fig = go.Figure()
    colors = {("Call", "ITM"): TEAL, ("Call", "ATM"): VIOLET, ("Call", "OTM"): CORAL,
              ("Put", "ITM"): TEAL, ("Put", "ATM"): VIOLET, ("Put", "OTM"): CORAL}
    rows_now = []
    for r in g.itertuples():
        S = F * (1 + grid_m / 100)
        pay = (np.maximum(S - r.strike, 0) if r.type == "Call" else np.maximum(r.strike - S, 0)) - r.premium
        fig.add_trace(go.Scatter(mode="lines", x=grid_m, y=pay, name=f"{r.type} {r.strike:,} ({r.zone})",
                                 line=dict(color=colors[(r.type, r.zone)], width=2.4, dash="solid" if r.type == "Call" else "dot"),
                                 hovertemplate="NIFTY %{x:+.1f}%<br>P&L ₹%{y:,.0f}<extra></extra>"))
        v = (max(ST - r.strike, 0) if r.type == "Call" else max(r.strike - ST, 0)) - r.premium
        rows_now.append((f"{r.type} {r.strike:,} · {r.zone}", v))
    fig.add_vline(x=move, line_color=INK, line_dash="dash", line_width=1)
    fig.add_hline(y=0, line_color=LINE, line_width=1)
    fig.update_layout(title=f"P&L per unit at expiry, NIFTY at {ST:,.0f}", height=420, xaxis_title="NIFTY change at expiry (%)",
                      yaxis_title="₹ per unit", legend=dict(orientation="h", y=-0.25), **LAYOUT)
    _chart(fig)
    html(kpi_row([kpi(n, f"₹{v:+,.0f}", "per unit at this level", TEAL if v >= 0 else CORAL) for n, v in rows_now[:3]]))
    html(kpi_row([kpi(n, f"₹{v:+,.0f}", "per unit at this level", TEAL if v >= 0 else CORAL) for n, v in rows_now[3:]]))
    html(f'<div class="nv-notice"><b>Analysis, not advice.</b> The right contract depends on your market view, budget and '
         'risk tolerance. Recommending specific trades requires SEBI registration, so NeuraVol presents the facts for '
         'each type and leaves the decision to you or your SEBI-registered adviser.</div>')
    st.caption("Market-implied chance is derived from option prices (risk-neutral) and is not a prediction. Example "
               "strikes are about 3% from the forward, rounded to 50. " + DISCLAIMER)

# ================================================================== VOLATILITY SURFACE
elif page == "Volatility surface":
    st.title("Volatility surface")
    k = dd.kpis(d, h)
    html(f'<div class="nv-sub">Implied volatility for every maturity and strike, today and as forecast {HL} ahead.</div>')
    html(kpi_row([
        kpi("30-day ATM volatility", f"{k['atm30']:.2f}%", "today", VIOLET),
        kpi(f"Forecast, {HL}", f"{k['fc_atm30']:.2f}%", "30-day ATM", CORAL, **chg(k['fc_chg'])),
        kpi("Downside skew", f"{k['skew30']:+.2f} pts", "5% put minus 5% call", AMBER),
        kpi("Term slope", f"{k['term']:+.2f} pts", "91-day minus 7-day", TEAL),
    ]))
    today, mask = dd.surface_today(d)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        view = choice("View", ["Heatmap", "3D"], "Heatmap", key="view")
        if view == "Heatmap":
            _chart(heatmap(dd.grid_frame(d, today, mask), "Implied volatility today (%)", SEQ))
        else:
            fig = go.Figure(go.Surface(z=today * 100, x=d.mon * 100, y=d.tenors, colorscale=SEQ,
                                       colorbar=dict(thickness=10, outlinewidth=0)))
            fig.update_layout(title="Implied volatility today (%)", height=440,
                              scene=dict(xaxis_title="strike vs forward %", yaxis_title="days", zaxis_title="IV %"), **LAYOUT)
            _chart(fig)
    with right:
        c1, c2 = st.columns(2)
        with c1:
            tenor = st.selectbox("Maturity", list(d.tenors), index=2, format_func=lambda x: f"{x} days")
        with c2:
            cmp_opt = {"None": None, "1 week ago": 5, "1 month ago": 21, "3 months ago": 63, "1 year ago": 252}
            cmp = st.selectbox("Compare with", list(cmp_opt), index=0)
        sm = dd.smile(d, tenor, h)
        fig = go.Figure()
        fig.add_trace(go.Scatter(mode="lines", x=sm.moneyness, y=sm.hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(mode="lines", x=sm.moneyness, y=sm.lo, fill="tonexty", fillcolor="rgba(232,86,108,0.12)",
                                 line=dict(width=0), name="typical forecast range"))
        if cmp_opt[cmp]:
            j = max(0, len(d.dates) - 1 - cmp_opt[cmp])
            ti = int(np.where(d.tenors == tenor)[0][0])
            fig.add_trace(go.Scatter(mode="lines", x=d.mon * 100, y=d.iv[j, ti] * 100, name=f"{d.dates[j]:%d %b %Y}",
                                     line=dict(color=MUTED, width=2, dash="dot")))
        fig.add_trace(go.Scatter(mode="lines+markers", x=sm.moneyness, y=sm.today, name="today",
                                 line=dict(color=VIOLET, width=3), marker=dict(size=6)))
        fig.add_trace(go.Scatter(mode="lines", x=sm.moneyness, y=sm.forecast, name=f"forecast, {HL}",
                                 line=dict(color=CORAL, width=3, dash="dash")))
        fig.update_layout(title=f"{tenor}-day volatility smile", height=400, xaxis_title="strike vs forward (%)",
                          yaxis_title="IV (%)", legend=dict(orientation="h", y=-0.25), hovermode="x unified", **LAYOUT)
        _chart(fig)
    delta = dd.forecast_grid(d, h) - today
    lim = max(0.5, float(np.nanmax(np.abs(delta * 100))))
    _chart(heatmap(dd.grid_frame(d, delta), f"Expected change over {HL} (vol points · pink = rise, blue = fall)",
                            DIV, zmid=0, fmt=".2f", zmin=-lim, zmax=lim, height=380))
    nn_ok = d.forecast["iv_nn"].notna().any() if "iv_nn" in d.forecast else True
    method = ("NeuraVol model (neural network combined with a factor model)" if nn_ok else
              "factor model only for this date (the neural model was not available when this forecast was made)")
    st.caption(f"Forecast: {method}. The shaded range reflects typical "
               "out-of-sample error in normal markets; during crises errors were two to four times larger.")

# ================================================================== STRIKE EXPLORER
elif page == "Strike explorer":
    st.title("Strike explorer")
    html('<div class="nv-sub">Analyse any listed contract: fair value, volatility outlook and what-if scenarios.</div>')
    ex = dd.expiries(d)
    c1, c2, c3 = st.columns([1.3, 1, 1])
    labels = {i: f"{r.expiry:%d %b %Y} · {int(r.days)} days" for i, r in ex.iterrows()}
    default = int(np.argmin(np.abs(ex.days.values - 30)))
    pick = c1.selectbox("Expiry", list(labels), index=default, format_func=lambda i: labels[i])
    row = ex.loc[pick]
    atm = int(round(row.F / 50) * 50)
    K = c2.number_input("Strike", min_value=int(row.F * 0.7) // 50 * 50, max_value=int(row.F * 1.3) // 50 * 50,
                        value=atm, step=50)
    with c3:
        typ = choice("Type", ["Call (CE)", "Put (PE)"], "Call (CE)", key="opt_type")
    is_call = typ.startswith("Call")
    v = dd.strike_view(d, row, float(K), is_call, h)
    zone = "ATM" if abs(K - row.F) <= 50 else ("ITM" if (K < row.F) == is_call else "OTM")
    last = f"last traded ₹{v['last_traded']:,.2f}" if v["last_traded"] is not None else "not traded today"
    html(kpi_row([
        kpi("Moneyness", zone, f"{v['moneyness']:+.1f}% vs forward {v['forward']:,.0f}", ZONE[zone][1]),
        kpi("Implied volatility", f"{v['iv_now']:.2f}%", f"{v['days']} days to expiry", VIOLET),
        kpi(f"Forecast volatility, {HL}", f"{v['iv_fc']:.2f}%", "expected", CORAL, **chg(v['change'])),
        kpi("Fair value today", f"₹{v['model_price_now']:,.2f}", last, TEAL),
    ]))

    st.subheader("What-if analysis")
    s1, s2, s3 = st.columns(3)
    mv = s1.slider("NIFTY move", -10.0, 10.0, 0.0, 0.5, format="%+.1f%%")
    dv = s2.slider("Change in implied volatility (pts)", -8.0, 8.0, float(round(v["change"] * 2) / 2), 0.5, format="%+.1f")
    dd_ = s3.slider("Days from today", 0, max(int(row.days) - 1, 1), min(int(round(h * 7 / 5)), max(int(row.days) - 1, 1)))
    F0, T0 = float(row.F), float(row["T"])
    T1 = max((row.days - dd_) / 365, 1 / 365)
    F1 = F0 * (1 + mv / 100)
    iv1 = max(v["iv_now"] / 100 + dv / 100, 0.01)
    p0 = v["model_price_now"]
    p1 = float(black76(F1, float(K), T1, iv1, is_call, np.exp(-row.rate * T1)))
    html(kpi_row([
        kpi("Scenario value", f"₹{p1:,.2f}", f"NIFTY {F1:,.0f} · IV {iv1 * 100:.1f}% · {row.days - dd_} days left", VIOLET),
        kpi("Change vs today", f"₹{p1 - p0:+,.2f}", f"{(p1 / p0 - 1) * 100:+.1f}% of today's value" if p0 > 0 else "", TEAL if p1 >= p0 else CORAL),
        kpi("Effect of time alone", f"₹{float(black76(F0, float(K), T1, v['iv_now'] / 100, is_call, np.exp(-row.rate * T1))) - p0:+,.2f}",
            f"over {dd_} days, nothing else changing", AMBER),
    ]))
    xs = np.linspace(-10, 10, 121)
    Fs = F0 * (1 + xs / 100)
    val_scn = black76(Fs, float(K), T1, iv1, is_call, np.exp(-row.rate * T1))
    val_exp = np.maximum(Fs - K, 0) if is_call else np.maximum(K - Fs, 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(mode="lines", x=xs, y=val_exp - p0, name="at expiry", line=dict(color=MUTED, width=2, dash="dot")))
    fig.add_trace(go.Scatter(mode="lines", x=xs, y=val_scn - p0, name=f"after {dd_} days at scenario IV",
                             line=dict(color=VIOLET, width=3)))
    fig.add_trace(go.Scatter(mode="markers", x=[mv], y=[p1 - p0], name="your scenario",
                             marker=dict(color=CORAL, size=13, line=dict(color="#fff", width=2))))
    fig.add_hline(y=0, line_color=LINE, line_width=1)
    fig.update_layout(title=f"Profit or loss per unit, bought at ₹{p0:,.2f}", height=420, xaxis_title="NIFTY move (%)",
                      yaxis_title="₹ per unit", hovermode="x unified", legend=dict(orientation="h", y=-0.25), **LAYOUT)
    _chart(fig)
    where = ("below" if v["last_traded"] and v["last_traded"] < v["range_lo"] else
             "above" if v["last_traded"] and v["last_traded"] > v["range_hi"] else "within")
    html(f'<div class="nv-notice">If implied volatility moves as forecast, this contract\'s fair value range is '
         f'<b>₹{v["range_lo"]:,.0f} to ₹{v["range_hi"]:,.0f}</b>'
         + (f'; the last traded price is {where} this range' if v["last_traded"] else "")
         + '. Analysis only, not a recommendation to buy or sell.</div>')

# ================================================================== HISTORY
elif page == "History":
    st.title("History")
    html('<div class="nv-sub">How NIFTY option volatility has evolved since 2016.</div>')
    hist = dd.history(d)
    span = choice("Period", ["1M", "6M", "1Y", "3Y", "5Y", "All"], "1Y", key="span")
    days = {"1M": 21, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "All": len(hist)}[span]
    hs = hist.iloc[-days:]
    s = hs["ATM 30d"]
    html(kpi_row([
        kpi("Latest 30-day volatility", f"{s.iloc[-1]:.2f}%", f"{span} change {s.iloc[-1] - s.iloc[0]:+.2f} pts", VIOLET),
        kpi(f"{span} average", f"{s.mean():.2f}%", "30-day ATM", TEAL),
        kpi(f"{span} high", f"{s.max():.2f}%", f"{s.idxmax():%d %b %Y}", CORAL),
        kpi(f"{span} low", f"{s.min():.2f}%", f"{s.idxmin():%d %b %Y}", SKY),
    ]))
    fig = go.Figure()
    for col, colr in zip(["ATM 7d", "ATM 30d", "ATM 91d"], [CORAL, VIOLET, TEAL]):
        fig.add_trace(go.Scatter(mode="lines", x=hs.index, y=hs[col], name=col.replace("ATM ", "") + " maturity",
                                 line=dict(width=1.5, color=colr)))
    fig.update_layout(title="At-the-money implied volatility (%)", height=420, hovermode="x unified",
                      legend=dict(orientation="h", y=1.1, x=1, xanchor="right"), **LAYOUT)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    _chart(fig)
    c1, c2 = st.columns(2, gap="large")
    for box, col, colr, ttl in zip((c1, c2), ["Skew 30d (-5% minus +5%)", "Term slope (91d minus 7d)"], [AMBER, SKY],
                                   ["Downside skew (vol points)", "Term slope, 91-day minus 7-day (vol points)"]):
        f = go.Figure(go.Scatter(mode="lines", x=hs.index, y=hs[col], line=dict(width=1.4, color=colr), fill="tozeroy",
                                 fillcolor="rgba(0,0,0,0.03)"))
        f.add_hline(y=0, line_color=LINE, line_width=1)
        f.update_layout(title=ttl, height=320, hovermode="x unified", **LAYOUT)
        box.plotly_chart(f)
    st.caption("Notable events: COVID-19 crash (March 2020), general election results (May 2019, June 2024).")

# ================================================================== TRACK RECORD
elif page == "Track record":
    st.title("Track record")
    html('<div class="nv-sub">Every forecast below was made using only data available at the time (walk-forward testing, '
         '2020 to 2026). Lower error is better; the best model in each row is highlighted.</div>')
    t_all, t_calm = dd.track_table(d), dd.track_table(d, exclude_covid=True)
    order = ["No change", "PCA-VAR", "NN v2", dd.BLEND]
    fmt = lambda t: (t[order].rename(index=lambda x: f"{x} day" + ("s" if x > 1 else "")).rename_axis("horizon")
                     .rename(columns={dd.BLEND: "NeuraVol", "No change": "No-change benchmark"}))
    a = d.atm30[d.atm30.horizon_days == h]
    err = (a[dd.BLEND] - a.actual).abs() * 100
    html(kpi_row([
        kpi("Forecasts evaluated", f"{len(a):,}", "trading days, 2020-2026", VIOLET),
        kpi(f"Typical error, {HL}", f"{err.mean():.2f} pts", "30-day ATM volatility", TEAL),
        kpi("Accuracy", f"{100 - (err / (a.actual * 100)).mean() * 100:.1f}%", "100% minus average % error", CORAL),
        kpi("Within 1 vol point", f"{(err <= 1).mean() * 100:.0f}%", "of forecasts", AMBER),
    ]))
    c1, c2 = st.columns(2, gap="large")
    c1.markdown("**All test days** · error in vol points")
    c1.dataframe(fmt(t_all).style.format("{:.2f}").highlight_min(axis=1, color=HILITE))
    c2.markdown("**Excluding the Feb-Jun 2020 crash**")
    c2.dataframe(fmt(t_calm).style.format("{:.2f}").highlight_min(axis=1, color=HILITE))
    sig = d.significance.get(str(h), {})
    p_rw = sig.get("NN v2 shrunk + PCA-VAR avg vs No change", [np.nan, np.nan])[1]
    html(f'<div class="nv-notice"><b>Assessment ({HL}):</b> NeuraVol has the lowest error of the models tested. A '
         f'Diebold-Mariano test against the no-change benchmark gives p = {p_rw:.2f}, so the improvement is not yet '
         'statistically significant.</div>')
    fig = go.Figure()
    fig.add_trace(go.Scatter(mode="lines", x=a.target_date, y=a.actual * 100, name="actual", line=dict(color=VIOLET, width=1.5)))
    fig.add_trace(go.Scatter(mode="lines", x=a.target_date, y=a[dd.BLEND] * 100, name=f"forecast made {HL} earlier",
                             line=dict(color=CORAL, width=1.2)))
    fig.update_layout(title="30-day ATM volatility: forecast vs actual (%)", height=400, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2), **LAYOUT)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    _chart(fig)
    st.markdown(f"**Error by year, {HL} horizon**")
    st.dataframe(dd.track_by_year(d, h)[order].rename(columns={dd.BLEND: "NeuraVol", "No change": "No-change benchmark"})
                 .style.format("{:.2f}").highlight_min(axis=1, color=HILITE))
    st.subheader("Trading backtest")
    st.caption("Delta-hedged at-the-money straddles traded on the forecast signal, 0.3% cost per option trade, 2020-2026. "
               "Includes an oracle with perfect foresight as an upper bound.")
    bt = d.bt_summary.rename(columns={"hold_days": "hold (days)", "hit_rate": "hit rate", "sharpe_annualised": "Sharpe",
                                      "total_pnl_volpts": "total P&L (vol pts)", "max_drawdown_volpts": "max drawdown"})
    st.dataframe(bt[["hold (days)", "strategy", "trades", "hit rate", "Sharpe", "total P&L (vol pts)", "max drawdown"]].round(2),
                 hide_index=True)
    html('<div class="nv-notice"><b>Result:</b> forecasts are more accurate than the benchmarks, but the edge does not '
         'survive trading costs in this strategy; systematically selling volatility performed better.</div>')
    live = dd.live_track_record(d)
    st.subheader("Live forecasts")
    if live.empty:
        st.caption("Forecasts are logged every trading day and scored here as their target dates arrive.")
    else:
        st.dataframe(live.pivot_table(index="horizon_days", columns="model", values="rmse", aggfunc="mean").round(2))

# ================================================================== ADMIN
elif page == "Admin":
    import admin_page
    admin_page.render(d, html, kpi_row, kpi, dict(VIOLET=VIOLET, TEAL=TEAL, CORAL=CORAL, AMBER=AMBER, SKY=SKY))

# ================================================================== USER GUIDE
else:
    st.title("User guide")
    html('<div class="nv-sub">How NeuraVol works and how to get the most out of each page.</div>')
    tabs = st.tabs(["Getting started", "Pages", "Reading the charts", "Options basics", "Data & accuracy"])
    with tabs[0]:
        html("""<div class="nv-panel"><h4>What NeuraVol does</h4><div style="color:#5E5C78;line-height:1.65">
        NeuraVol tracks how expensive NIFTY 50 options are and forecasts whether they are likely to become more or
        less expensive over the next day, week or month. Option prices are driven largely by <b>implied volatility</b>,
        the market's expectation of how much NIFTY will move. NeuraVol measures it for every strike and expiry, compares
        it with history and forecasts its next move.</div></div>""")
        s = st.columns(3, gap="medium")
        steps = [("1", "Check the market", "Open <b>Overview</b> to see today's volatility regime, the forecast and "
                                            "how options are priced compared with the past year."),
                 ("2", "Choose your horizon", "Use the <b>Forecast horizon</b> buttons in the sidebar (1 day, 1 week, "
                                              "1 month). Every page updates to that horizon."),
                 ("3", "Analyse a contract", "Use the <b>Strike guide</b> or <b>Strike explorer</b> to see a contract's "
                                             "fair value and test what-if scenarios before you decide.")]
        for c, (n, t, p) in zip(s, steps):
            c.markdown(f'<div class="nv-step"><div class="n">{n}</div><b>{t}</b><p>{p}</p></div>', unsafe_allow_html=True)
        html(f'<div class="nv-notice" style="margin-top:16px">Data refreshes every trading evening after NSE publishes '
             f'end-of-day prices. The date in the sidebar shows the latest session. {DISCLAIMER}</div>')
    with tabs[1]:
        pages = [("Overview", "Today's market at a glance: volatility regime, outlook, strike guide for ITM / ATM / OTM "
                              "calls and puts, and a profit-and-loss scenario slider.", "Start every session here."),
                 ("Volatility surface", "Implied volatility across all maturities and strikes, the forecast for each, "
                                        "and a comparison with any recent date.", "See where options are cheap or expensive."),
                 ("Strike explorer", "Pick any listed contract to see its fair value, volatility outlook and a what-if "
                                     "calculator for NIFTY moves, volatility changes and time passing.", "Plan and stress-test a position."),
                 ("History", "Ten years of volatility, downside skew and term structure, with period buttons and a zoom bar.",
                  "Put today's levels in context."),
                 ("Track record", "How accurate past forecasts were, compared with standard benchmarks, including a "
                                  "trading backtest.", "Judge how much to rely on the forecast."),
                 ("User guide", "This page.", "")]
        for name, what, use in pages:
            html(f'<div class="nv-panel"><h4>{name}</h4><div style="color:#5E5C78;line-height:1.6">{what}'
                 + (f'<br><b style="color:#6D5DFC">{use}</b>' if use else "") + "</div></div>")
    with tabs[2]:
        html("""<div class="nv-panel"><h4>Volatility heatmap</h4><div style="color:#5E5C78;line-height:1.65">Each cell is
        the implied volatility of one type of option. Rows are time to expiry (7 days at the top, 1 year at the
        bottom); columns are how far the strike is from NIFTY (left: below the market, right: above). Darker cells are
        more expensive. Blank cells have no traded contracts and are estimated.</div></div>
        <div class="nv-panel"><h4>Volatility smile</h4><div style="color:#5E5C78;line-height:1.65">The same information
        for one maturity as a curve. It usually slopes down from left to right because protection against a fall costs
        more. The solid line is today, the dashed line the forecast, and the shaded area the typical forecast range.
        Use <b>Compare with</b> to overlay a past date.</div></div>
        <div class="nv-panel"><h4>Expected-change map</h4><div style="color:#5E5C78;line-height:1.65">How much the model
        expects each cell to change over the selected horizon. Pink means options are expected to become more expensive,
        blue cheaper. +0.50 means, for example, 14.0% rising to 14.5%.</div></div>
        <div class="nv-panel"><h4>History charts</h4><div style="color:#5E5C78;line-height:1.65">Spikes in volatility mark
        periods of fear, such as March 2020. Rising downside skew means protection against a fall is in demand. A term
        slope below zero means short-term volatility is above long-term, usually ahead of an event.</div></div>""")
    with tabs[3]:
        F = K0["forward"]
        r50 = lambda x: f"{int(round(x / 50) * 50):,}"
        html(f"""<div class="nv-panel"><h4>Calls and puts</h4><div style="color:#5E5C78;line-height:1.65">An option gives
        the right, not the obligation, to buy (<b>call</b>) or sell (<b>put</b>) NIFTY at a fixed <b>strike</b> on a fixed
        <b>expiry</b> date. The price is the <b>premium</b>. A buyer's maximum loss is the premium; a seller collects it
        but can lose considerably more.</div></div>
        <div class="nv-panel"><h4>ITM, ATM and OTM</h4><div style="color:#5E5C78;line-height:1.65">With NIFTY's forward
        near {F:,.0f}: a call at {r50(F * 0.97)} is <b>in the money</b>, at {r50(F)} <b>at the money</b>, and at
        {r50(F * 1.03)} <b>out of the money</b>. For puts it is the reverse. ITM options cost more and track NIFTY
        closely; OTM options are cheaper but need a larger move to pay off.</div></div>
        <div class="nv-panel"><h4>Implied volatility</h4><div style="color:#5E5C78;line-height:1.65">The market's expected
        annual movement in NIFTY, derived from option prices. Today's 30-day level of {K0['atm30']:.1f}% implies a typical
        move of about ±{K0['atm30'] / np.sqrt(12):.1f}% over the next month. Higher volatility means more expensive options.
        </div></div>
        <div class="nv-panel"><h4>Key terms</h4><div style="color:#5E5C78;line-height:1.75">
        <b>Delta</b>: change in option price for a ₹1 move in NIFTY. <b>Time decay</b>: the daily loss of value as expiry
        approaches. <b>Vol point</b>: one percentage point of implied volatility. <b>Forward</b>: NIFTY's expected price at
        expiry, slightly above spot due to interest.</div></div>""")
    with tabs[4]:
        a = d.atm30[d.atm30.horizon_days == 1]
        html(f"""<div class="nv-panel"><h4>Data</h4><div style="color:#5E5C78;line-height:1.65">End-of-day prices for every
        NIFTY option and future from NSE, {d.dates[0]:%B %Y} to {d.dates[-1]:%B %Y} ({len(d.dates):,} sessions), updated
        each trading evening. Live bid and ask quotes are collected during market hours to improve future versions.</div></div>
        <div class="nv-panel"><h4>Forecast accuracy (30-day ATM volatility)</h4><div style="color:#5E5C78;line-height:1.75">
        Tested on 1,668 sessions from 2020 to 2026 using only information available at the time:<br>
        1 day ahead: about <b>95%</b> accurate · 1 week: about <b>91%</b> · 1 month: about <b>82%</b>.<br>
        Direction of meaningful moves called correctly 57-60% of the time. See <b>Track record</b> for full detail,
        including where the model did not beat simple benchmarks.</div></div>""")
        html(f'<div class="nv-notice">{DISCLAIMER}</div>')

_mark('page done')
try:
    _dbg = st.query_params.get("debug")
except Exception:
    _dbg = None
_line = " | ".join(f"{n} {t:.2f}s" for n, t in _MARKS)
print(f"[timing] page={st.session_state.get('page')} {_line}", flush=True)
if _dbg:
    import sys, plotly, scipy
    st.text(f"python {sys.version.split()[0]} streamlit {st.__version__} pandas {pd.__version__} numpy {np.__version__} "
            f"plotly {getattr(plotly, '__version__', '?')} scipy {scipy.__version__}\n" + "\n".join(f"{t:7.2f}s  {n}" for n, t in _MARKS))
