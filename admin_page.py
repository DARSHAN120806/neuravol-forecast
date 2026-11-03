"""Admin panel: password-protected data sync, health checks, live snapshots, connections and maintenance."""
import hmac
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src import sync

ROOT = C.ROOT


def secret(name, default=None):
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def _github():
    tok = secret("GITHUB_TOKEN")
    if not tok:
        return None
    from src.github_store import GitHubStore
    return GitHubStore(tok, secret("GITHUB_REPO", "DARSHAN120806/neuravol-forecast"), secret("GITHUB_BRANCH", "main"))


def _apply_upstox_secret():
    t = secret("UPSTOX_ANALYTICS_TOKEN")
    if t:
        os.environ["UPSTOX_ANALYTICS_TOKEN"] = t
    return sync.upstox_token()


def _table(df, fmt=None):
    from ui_table import html_table
    st.markdown(html_table(df, fmt, index=False), unsafe_allow_html=True)


def _status(label):
    s = getattr(st, "status", None)
    return s(label, expanded=True) if s else st.container()


def _push(files, message, ui):
    gh = _github()
    if gh is None:
        ui.write("GitHub saving skipped: no GITHUB_TOKEN configured.")
        return "not configured"
    try:
        sha = gh.commit_files(ROOT, files, message)
        ui.write(f"Saved to GitHub (commit {sha[:7]}). The public site redeploys with the new data in about a minute.")
        return f"saved {sha[:7]}"
    except Exception as e:
        ui.write(f"GitHub save failed: {e}")
        return f"failed: {e}"


def login(html, kpi_row, kpi, colors):
    pw = secret("ADMIN_PASSWORD")
    st.title("Admin")
    if not pw:
        html('<div class="nv-sub">The admin panel is disabled until an admin password is configured.</div>')
        html(f"""<div class="nv-panel"><h4>Enable the admin panel</h4><div style="color:#5E5C78;line-height:1.7">
        <b>On Streamlit Cloud:</b> App menu &rsaquo; Settings &rsaquo; Secrets, then paste the lines below.<br>
        <b>On your laptop:</b> create <code>.streamlit/secrets.toml</code> in the project folder with the same lines
        (this file is excluded from GitHub).</div></div>""")
        st.code('ADMIN_PASSWORD = "choose-a-strong-password"\n'
                'UPSTOX_ANALYTICS_TOKEN = "your Upstox analytics token"      # optional: Upstox fallback\n'
                'GITHUB_TOKEN = "github fine-grained token"                 # optional: save synced data\n'
                'GITHUB_REPO = "DARSHAN120806/neuravol-forecast"', language="toml")
        return False
    if st.session_state.get("admin_ok"):
        return True
    html('<div class="nv-sub">Restricted area. Sign in to manage data and updates.</div>')
    with st.form("admin_login"):
        entered = st.text_input("Admin password", type="password")
        ok = st.form_submit_button("Sign in", type="primary")
    if ok:
        if hmac.compare_digest(entered.encode(), pw.encode()):
            st.session_state["admin_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def render(d, html, kpi_row, kpi, colors):
    if not login(html, kpi_row, kpi, colors):
        return
    V, T, CO, A, S = colors["VIOLET"], colors["TEAL"], colors["CORAL"], colors["AMBER"], colors["SKY"]
    top = st.columns([4, 1])
    top[0].markdown('<div class="nv-sub">Data sync, quality checks, live quotes, connections and maintenance.</div>',
                    unsafe_allow_html=True)
    if top[1].button("Sign out", use_container_width=True):
        st.session_state["admin_ok"] = False
        st.rerun()

    token = _apply_upstox_secret()
    now = sync.now_ist()
    latest = sync.latest_completed_session(now)
    last = sync.last_date()
    behind = len(sync.missing_weekdays(last, latest))
    hist = pd.read_csv(sync.SYNC_LOG) if sync.SYNC_LOG.exists() else pd.DataFrame()
    fc = pd.read_csv(C.DATA / "forecasts" / "latest_forecast.csv")
    live = sorted((C.DATA / "live").glob("*.csv")) if (C.DATA / "live").exists() else []
    html(kpi_row([
        kpi("Latest stored session", f"{last:%d %b %Y}", f"latest completed session {latest:%d %b}", V),
        kpi("Sessions to sync", f"{behind}", "weekdays incl. any holidays", CO if behind else T),
        kpi("Last sync", hist.time_ist.iloc[-1][5:16] if len(hist) else "never",
            (hist.message.iloc[-1] if len(hist) else "no sync recorded yet"), A),
        kpi("Forecast as of", f"{pd.Timestamp(fc.as_of.iloc[0]):%d %b %Y}", "1 day, 1 week, 1 month", S),
        kpi("Live snapshots", f"{len(live)} days", (live[-1].stem if live else "none collected here"), T),
    ]))

    tabs = st.tabs(["Sync", "Data health", "Live quotes", "Connections", "Downloads & history", "Maintenance"])

    # ---------------------------------------------------------------- sync
    with tabs[0]:
        c1, c2, c3 = st.columns([1.4, 1, 1])
        src_label = c1.selectbox("Data source", ["Auto: NSE, then Upstox for the latest session", "NSE bhavcopy only",
                                                 "Upstox live quotes only (latest session)"])
        src = {"A": "auto", "N": "nse", "U": "upstox"}[src_label[0]]
        until = c2.date_input("Sync up to", value=latest, max_value=now.date(), min_value=last)
        with c3:
            upd = st.checkbox("Update forecast", value=True)
            save = st.checkbox("Save to GitHub", value=bool(secret("GITHUB_TOKEN")), disabled=not secret("GITHUB_TOKEN"),
                               help="Requires GITHUB_TOKEN in secrets. Without it, data synced on the cloud is lost on restart.")
        if src != "nse" and not token:
            st.warning("No Upstox token configured: the Upstox fallback is unavailable.")
        if st.button("Sync now", type="primary", icon=":material/sync:") if _btn_icon() else st.button("Sync now", type="primary"):
            box = _status("Syncing data...")
            lines = []

            def log(msg):
                lines.append(str(msg))
                box.write(msg)

            res = sync.run_sync(until=until, source=src, update_forecast=upd, log=log)
            if save and res.added:
                class _Ui:
                    write = staticmethod(log)
                pushed = _push(sync.SYNC_FILES, f"Admin sync: {', '.join(str(x) for x in res.added)}", _Ui)
                sync.mark_last_push(pushed)
            st.cache_resource.clear()
            # Keep the report across a rerun so the status cards and every page show the new data.
            st.session_state["sync_report"] = dict(message=res.message, lines=lines,
                                                   ok=bool(res.added or not res.skipped))
            st.rerun()
        rep = st.session_state.get("sync_report")
        if rep:
            with st.expander(rep["message"], expanded=True):
                for ln in rep["lines"]:
                    st.write(ln)
            (st.success if rep["ok"] else st.error)(rep["message"] + " All pages now show the updated data.")
            if st.button("Clear report"):
                st.session_state.pop("sync_report", None)
                st.rerun()
        st.caption("Sync downloads each missing session from NSE. If NSE is unreachable (common from cloud servers), "
                   "the latest completed session is built from Upstox bid/ask quotes instead. Earlier missed sessions "
                   "need NSE and can be filled by the evening scheduler on your laptop.")

    # ---------------------------------------------------------------- data health
    with tabs[1]:
        health, gaps = sync.data_health(30)
        bad = (health.status != "OK").sum()
        html(kpi_row([
            kpi("Sessions checked", f"{len(health)}", "most recent", V),
            kpi("Sessions flagged", f"{bad}", "fit error, quotes or coverage", CO if bad else T),
            kpi("Median fit error", f"{health.fit_error_pts.median():.2f} pts", "lower is better", A),
            kpi("Missing weekdays (90 days)", f"{len(gaps)}", "mostly market holidays", S),
        ]))
        show = health.sort_values("date", ascending=False).rename(columns={
            "n_slices": "expiries", "n_points": "quotes", "fit_error_pts": "fit error (pts)"})
        show["date"] = pd.to_datetime(show["date"]).dt.strftime("%d %b %Y")
        _table(show, {"fit error (pts)": "{:.2f}", "coverage": "{:.0%}"})
        if gaps:
            st.caption("Weekdays with no stored session in the last 90 days: "
                       + ", ".join(f"{g:%d %b}" for g in gaps) + ". Market holidays appear here too.")

    # ---------------------------------------------------------------- live quotes
    with tabs[2]:
        st.caption("Take an on-demand snapshot of the full NIFTY option chain with live bid and ask prices (Upstox). "
                   "Snapshots taken around 15:25 are used to build the cleaner mid-price surfaces.")
        if st.button("Take snapshot now", disabled=not token):
            try:
                snap = sync.fetch_upstox_snapshot(token)
                (C.DATA / "live").mkdir(parents=True, exist_ok=True)
                out = C.DATA / "live" / f"{now:%Y-%m-%d}.csv"
                snap.to_csv(out, mode="a", header=not out.exists(), index=False)
                q = ((snap.bid > 0) & (snap.ask > 0)).sum()
                st.success(f"Saved {len(snap):,} quotes ({q:,} two-sided) across {snap.expiry.nunique()} expiries.")
                s0 = snap.spot.iloc[0]
                near = snap[snap.expiry == sorted(snap.expiry.unique())[min(2, snap.expiry.nunique() - 1)]]
                near = near.iloc[(near.strike - s0).abs().argsort()[:8]].sort_values(["strike", "option_type"])
                near["spread %"] = (near.ask - near.bid) / ((near.ask + near.bid) / 2) * 100
                _table(near[["expiry", "strike", "option_type", "ltp", "bid", "ask", "spread %", "volume", "oi"]],
                       {"ltp": "{:.2f}", "bid": "{:.2f}", "ask": "{:.2f}", "spread %": "{:.2f}", "strike": "{:,.0f}",
                        "volume": "{:,.0f}", "oi": "{:,.0f}"})
            except Exception as e:
                st.error(f"Snapshot failed: {e}")
        if live:
            rows = []
            for f in live[-15:][::-1]:
                df = pd.read_csv(f, usecols=["ts"])
                rows.append(dict(day=f.stem, snapshots=df.ts.nunique(), first=df.ts.min()[11:16], last=df.ts.max()[11:16],
                                 rows=len(df)))
            _table(pd.DataFrame(rows))
        if st.button("Build mid-price surfaces from snapshots", disabled=not live):
            from collector import build_mid_surface
            box = _status("Building mid-price surfaces...")
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                build_mid_surface.main([])
            box.write(buf.getvalue() or "done")

    # ---------------------------------------------------------------- connections
    with tabs[3]:
        cfg = [("Admin password", bool(secret("ADMIN_PASSWORD")), "required for this panel"),
               ("Upstox analytics token", bool(token), "fallback data source and live snapshots"),
               ("GitHub token", bool(secret("GITHUB_TOKEN")), "saves synced data permanently"),
               ("GitHub repository", True, secret("GITHUB_REPO", "DARSHAN120806/neuravol-forecast"))]
        _table(pd.DataFrame([dict(setting=a, status="configured" if b else "missing", purpose=c) for a, b, c in cfg]))
        b1, b2, b3 = st.columns(3)
        if b1.button("Test NSE", use_container_width=True):
            ok, msg = sync.nse_reachable()
            (st.success if ok else st.error)(f"NSE: {'reachable' if ok else 'not reachable'} ({msg})")
        if b2.button("Test Upstox", use_container_width=True):
            ok, msg = sync.upstox_reachable(token)
            (st.success if ok else st.error)(f"Upstox: {'working' if ok else 'not working'} ({msg})")
        if b3.button("Test GitHub", use_container_width=True):
            gh = _github()
            if gh is None:
                st.error("GitHub: no GITHUB_TOKEN configured")
            else:
                ok, msg = gh.check()
                (st.success if ok else st.error)(f"GitHub: {msg}")
        st.caption("Secrets are never displayed. Add or change them in Streamlit Cloud under Settings > Secrets, or in "
                   ".streamlit/secrets.toml on your laptop.")

    # ---------------------------------------------------------------- downloads & history
    with tabs[4]:
        files = [("Latest forecast (CSV)", C.DATA / "forecasts" / "latest_forecast.csv", "text/csv"),
                 ("Forecast log (CSV)", C.DATA / "forecasts" / "forecast_log.csv", "text/csv"),
                 ("Surface diagnostics (CSV.GZ)", C.SURFACES / f"{C.SYMBOL}_diagnostics.csv.gz", "application/gzip"),
                 ("Surface history (NPZ)", C.SURFACES / f"{C.SYMBOL}_grid.npz", "application/octet-stream"),
                 ("Latest option chain (CSV)", C.DATA / "latest_chain.csv", "text/csv"),
                 ("Sync history (CSV)", sync.SYNC_LOG, "text/csv")]
        cols = st.columns(3)
        for i, (label, p, mime) in enumerate(files):
            if Path(p).exists():
                cols[i % 3].download_button(label, Path(p).read_bytes(), file_name=Path(p).name, mime=mime,
                                            use_container_width=True)
        st.markdown("**Sync history**")
        if len(hist):
            _table(hist.iloc[::-1].drop(columns=["sources"], errors="ignore"), {"seconds": "{:.1f}"})
        else:
            st.caption("No syncs recorded yet.")

    # ---------------------------------------------------------------- maintenance
    with tabs[5]:
        m1, m2 = st.columns(2, gap="large")
        with m1:
            html('<div class="nv-panel"><h4>Refresh</h4><div style="color:#5E5C78">Reload all data from disk, or recompute '
                 'the forecast for the latest stored session.</div></div>')
            if st.button("Reload data", use_container_width=True):
                st.cache_resource.clear()
                st.success("Data reloaded.")
            if st.button("Recompute forecast", use_container_width=True):
                box = _status("Recomputing forecast...")
                method = sync.refresh_forecast(box.write)
                st.cache_resource.clear()
                st.success(f"Forecast recomputed ({method}).")
        with m2:
            html(f'<div class="nv-panel"><h4>Undo last session</h4><div style="color:#5E5C78">Removes '
                 f'{last:%d %b %Y} from the stored history, for example after a sync with bad data. Run Sync again to '
                 'rebuild it.</div></div>')
            confirm = st.checkbox(f"I want to remove {last:%d %b %Y}")
            also_save = st.checkbox("Also save the change to GitHub", value=False, disabled=not secret("GITHUB_TOKEN"))
            if st.button("Remove last session", disabled=not confirm, use_container_width=True):
                removed = sync.remove_last_session()
                sync.refresh_forecast(lambda *_: None)
                if also_save:
                    _push(sync.SYNC_FILES, f"Admin: removed session {removed}", st)
                st.cache_resource.clear()
                st.success(f"Removed {removed:%d %b %Y}.")


def _btn_icon():
    import inspect
    try:
        return "icon" in inspect.signature(st.button).parameters
    except (TypeError, ValueError):
        return False
