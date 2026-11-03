"""Lightweight HTML tables.

st.dataframe loads a large JavaScript component that Streamlit Community Cloud sometimes serves very slowly,
leaving grey placeholders. These tables are plain HTML, so they render with the rest of the page.
"""
from html import escape

import numpy as np
import pandas as pd

CSS = """
.nv-table-wrap { overflow-x: auto; margin: 4px 0 14px 0; border: 1px solid #E6E8F2; border-radius: 12px; background: #fff; }
.nv-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.nv-table th { text-align: right; font-weight: 600; color: #5E5C78; background: #F7F8FC; padding: 8px 10px;
               border-bottom: 1px solid #E6E8F2; white-space: normal; vertical-align: bottom; line-height: 1.25; }
.nv-table td { text-align: right; padding: 7px 10px; border-bottom: 1px solid #F0F1F7; color: #1E2235; white-space: nowrap; }
.nv-table th:first-child, .nv-table td:first-child { text-align: left; }
.nv-table tr:last-child td { border-bottom: none; }
.nv-table td.hl { background: #E6F6F3; color: #0B7A6E; font-weight: 600; }
"""


def _fmt(v, f):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if f and isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        return f.format(v) if isinstance(f, str) else f(v)
    if isinstance(v, (float, np.floating)):
        return f"{v:,.2f}"
    return str(v)


def html_table(df: pd.DataFrame, fmt=None, highlight_min=False, index=True) -> str:
    """fmt: one format string for all numbers, or {column: format}. highlight_min: mark the lowest number per row."""
    df = df.copy()
    if not index:
        df = df.reset_index(drop=True)
    cols = list(df.columns)
    head = ("<th></th>" if index else "") + "".join(f"<th>{escape(str(c))}</th>" for c in cols)
    rows = []
    for idx, r in df.iterrows():
        num = pd.to_numeric(r, errors="coerce")
        lo = num.min() if highlight_min and num.notna().any() else None
        cells = []
        for c in cols:
            f = fmt.get(c) if isinstance(fmt, dict) else fmt
            cls = ' class="hl"' if lo is not None and pd.notna(num[c]) and num[c] == lo else ""
            cells.append(f"<td{cls}>{escape(_fmt(r[c], f))}</td>")
        rows.append("<tr>" + (f"<td>{escape(str(idx))}</td>" if index else "") + "".join(cells) + "</tr>")
    return (f'<div class="nv-table-wrap"><table class="nv-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')
