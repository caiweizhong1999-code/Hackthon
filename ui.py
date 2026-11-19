"""
ui.py
Visual part of the UI: CSS + pure functions that generate HTML snippets (no streamlit dependency, can be rendered and previewed standalone).
app.py uses them via st.markdown(..., unsafe_allow_html=True).

Security: all text from data/users (column names, values, questions) goes through html.escape before being put into HTML.
"""
from __future__ import annotations

import html
from typing import Any, Iterable

import pandas as pd

OP_PRETTY = {"==": "=", "!=": "≠", ">=": "≥", "<=": "≤", ">": ">", "<": "<", "in": "in"}
OUTCOME_LABELS = {"overall_survival": "Overall survival", "progression_free_survival": "Progression-free survival"}
PLOT_LABELS = {"km_curve": "KM curve", "histogram": "Histogram", "boxplot": "Box plot",
               "bar": "Bar chart", "scatter": "Scatter plot", "trend": "Trend plot"}
SOURCE_LABELS = {"llm": "LLM", "rules": "Rules fallback", "manual": "Edited"}

e = html.escape


# ============================================================================
# Icons (24x24 stroked SVG)
# ============================================================================
def _svg(paths: str, size: int = 22, color: str = "currentColor", sw: float = 1.8) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>')


ICONS = {
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>'
                '<path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M8 17v-5"/><path d="M13 17V8"/><path d="M18 17v-9"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "code": '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
             '<path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "table": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18"/>',
    "sparkle": '<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "chevron": '<polyline points="9 18 15 12 9 6"/>',
}


def icon(name: str, size: int = 22, color: str = "currentColor") -> str:
    return _svg(ICONS[name], size, color)


# ============================================================================
# CSS
# ============================================================================
CSS = """
<style>
:root{--navy:#171a47;--violet:#5b3fd6;--violet2:#7c5cf0;--ink:#1d2143;--muted:#6b7194;
      --panel:#eef0fb;--panel2:#e8ebfa;--line:#e3e5f3;--green:#16a34a;}
html, body, [class*="css"], .stApp{font-family:Inter,"Segoe UI",system-ui,-apple-system,sans-serif;}
.stApp{background:#f3f4fb;}
.block-container{padding-top:1.3rem;max-width:1320px;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}

/* ---- sidebar ---- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#171a47 0%,#1e2461 100%);}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label p,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] summary,
[data-testid="stSidebar"] summary p,
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#e6e8f8 !important;}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.12);}
.logo{display:flex;align-items:center;gap:12px;padding:6px 4px 18px 4px;}
.logo-ic{width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,#4f7cff,#7c5cf0);
         display:flex;align-items:center;justify-content:center;color:#fff;box-shadow:0 4px 14px rgba(79,124,255,.45);}
.logo-t{color:#fff;font-weight:700;font-size:19px;line-height:1.15;}
.side-label{color:#9aa0d0;font-size:11px;letter-spacing:.09em;text-transform:uppercase;font-weight:600;margin:14px 2px 6px;}

/* ---- header + steps ---- */
.hdr{display:flex;align-items:center;gap:14px;margin:2px 0 16px;}
.hdr-ic{width:44px;height:44px;border-radius:12px;background:#e6e4fb;color:var(--violet);
        display:flex;align-items:center;justify-content:center;}
.hdr-t{font-size:27px;font-weight:800;color:var(--ink);letter-spacing:-.01em;line-height:1.2;}
.hdr-s{color:var(--muted);font-size:14.5px;margin-top:2px;}
.steps{display:flex;align-items:center;gap:12px;margin-bottom:20px;}
.step{flex:1;display:flex;align-items:center;gap:14px;background:var(--panel2);border:1.5px solid transparent;
      border-radius:14px;padding:15px 18px;position:relative;}
.step .s-ic{color:var(--violet);opacity:.85;}
.step .s-t{font-weight:700;color:var(--ink);font-size:15.5px;}
.step .s-t b{color:var(--violet);margin-right:6px;}
.step .s-s{color:var(--muted);font-size:13px;margin-top:1px;}
.step.active{background:#fff;border-color:var(--violet2);box-shadow:0 4px 16px rgba(91,63,214,.12);}
.step.done{background:#fff;border-color:#bfe6cd;}
.step .tick{position:absolute;top:10px;right:12px;width:20px;height:20px;border-radius:50%;background:var(--green);
            color:#fff;display:flex;align-items:center;justify-content:center;}
.chev{color:#a5abd6;display:flex;}

/* ---- section header strip ---- */
.sec{display:flex;align-items:center;gap:12px;background:linear-gradient(90deg,#e9ebfc,#f1effc);
     border:1px solid var(--line);border-radius:14px;padding:13px 18px;margin:6px 0 14px;}
.sec-ic{width:34px;height:34px;border-radius:10px;background:#fff;color:var(--violet);display:flex;
        align-items:center;justify-content:center;box-shadow:0 1px 4px rgba(60,70,150,.12);}
.sec-t{font-size:20px;font-weight:800;color:var(--ink);}
.sec-s{color:var(--muted);font-size:13.5px;margin-left:6px;}

/* ---- panels ---- */
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 18px;height:100%;}
.panel-t{display:flex;align-items:center;gap:8px;font-weight:700;color:var(--ink);font-size:15.5px;margin-bottom:12px;}
.panel-t svg{color:var(--violet);}
.pill{margin-left:auto;font-size:11px;font-weight:700;padding:3px 9px;border-radius:999px;background:#fff;
      color:var(--violet);border:1px solid #d9d4fa;letter-spacing:.02em;}
.pq-row{display:flex;align-items:flex-start;gap:9px;padding:5px 0;color:var(--ink);font-size:14.5px;}
.pq-row .ck{flex:none;width:18px;height:18px;border-radius:50%;background:#e1f5e8;color:var(--green);
            display:flex;align-items:center;justify-content:center;margin-top:1px;}
.pq-row .ck.off{background:#ececf5;color:#a3a8c8;}
.pq-row b{font-weight:600;}
.muted{color:var(--muted);}

.tiles{display:flex;gap:12px;}
.tile{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px 8px;text-align:center;}
.tile .n{font-size:31px;font-weight:800;line-height:1.1;letter-spacing:-.01em;}
.tile .l{color:var(--ink);font-size:13.5px;margin-top:4px;}
.tile .sub{color:var(--muted);font-size:11.5px;margin-top:1px;}
.t1 .n{color:#1d2759;} .t2 .n{color:#4c2fc4;} .t3 .n{color:#6d2fc9;}

table.dist{width:100%;border-collapse:collapse;font-size:14px;background:#fff;border-radius:10px;overflow:hidden;
           border:1px solid var(--line);}
table.dist th{background:#fff;text-align:left;padding:8px 12px;font-weight:700;color:var(--ink);
              border-bottom:1.5px solid var(--line);}
table.dist th:last-child, table.dist td:last-child{text-align:right;}
table.dist td{padding:7px 12px;color:var(--ink);border-bottom:1px solid #f0f1f8;}
table.dist tr:last-child td{border-bottom:none;}

.chips{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0 10px;}
.chip{background:#e8e5fc;color:#4a33bd;font-weight:600;font-size:13px;padding:5px 12px;border-radius:999px;}
.chip.none{background:#eef0fb;color:var(--muted);font-weight:500;}

/* ---- risk result ---- */
.risk{background:linear-gradient(180deg,#f4f2ff,#eceafc);border:1px solid var(--line);border-radius:14px;
      padding:20px 14px;text-align:center;}
.risk .rt{color:var(--ink);font-weight:600;font-size:14.5px;}
.risk .rp{font-size:56px;font-weight:800;color:#5a24b8;line-height:1.1;margin:8px 0 12px;letter-spacing:-.02em;}
.badge{display:inline-block;font-weight:700;font-size:14px;padding:6px 18px;border-radius:999px;}
.badge.low{background:#d9f3e2;color:#157a3b;} .badge.mod{background:#fdecc8;color:#a15207;}
.badge.high{background:#fde0e0;color:#b4232a;}
.risk .rd{color:var(--muted);font-size:12.5px;margin-top:14px;line-height:1.55;}
.risk.empty .rp{color:#b9b4e6;}

/* ---- native widgets ---- */
div[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border:1px solid var(--line) !important;
     border-radius:14px !important;box-shadow:0 1px 3px rgba(40,50,120,.06);}
[data-testid="stForm"]{border:none !important;padding:0 !important;background:transparent;}
button[kind="primary"], button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"]{
     background:linear-gradient(135deg,#6a4de0,#4f35c9) !important;border:none !important;border-radius:10px !important;
     font-weight:600 !important;color:#fff !important;box-shadow:0 4px 12px rgba(91,63,214,.30);}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primaryFormSubmit"]:hover{filter:brightness(1.08);}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"]{border-radius:10px !important;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"]{display:none;}
[data-testid="stTabs"] button[role="tab"]{border-radius:9px;padding:6px 16px;margin-right:6px;background:#f1f2fb;}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{background:var(--violet);}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p{color:#fff !important;}
[data-testid="stTabs"] [role="tablist"]{gap:0;border-bottom:none;}
</style>
"""


# ============================================================================
# HTML snippets
# ============================================================================
def logo_html() -> str:
    return (f'<div class="logo"><div class="logo-ic">{icon("activity", 24, "#fff")}</div>'
            f'<div class="logo-t">AI Clinical<br>Analytics</div></div>')


def side_label(text: str) -> str:
    return f'<div class="side-label">{e(text)}</div>'


def header_html() -> str:
    return (f'<div class="hdr"><div class="hdr-ic">{icon("sparkle", 24)}</div><div>'
            f'<div class="hdr-t">AI-Powered Clinical Data Explorer</div>'
            f'<div class="hdr-s">Explore patient cohorts, generate visualizations, and predict outcomes '
            f'using natural language.</div></div></div>')


def steps_html(done: tuple[bool, bool, bool]) -> str:
    items = [("database", "Upload Data", "CSV / clinical dataset"),
             ("chart", "Explore &amp; Visualize", "Ask questions in natural language"),
             ("activity", "Predict &amp; Explain", "Patient risk prediction with SHAP")]
    first_todo = next((i for i, d in enumerate(done) if not d), None)
    parts = []
    for i, (ic, title, sub) in enumerate(items):
        cls = "done" if done[i] else ("active" if i == first_todo else "")
        tick = f'<span class="tick">{icon("check", 12, "#fff")}</span>' if done[i] else ""
        parts.append(f'<div class="step {cls}">{tick}<span class="s-ic">{icon(ic, 30)}</span>'
                     f'<div><div class="s-t"><b>{i + 1}</b>{title}</div><div class="s-s">{sub}</div></div></div>')
        if i < 2:
            parts.append(f'<div class="chev">{icon("chevron", 26)}</div>')
    return f'<div class="steps">{"".join(parts)}</div>'


def section_header(title: str, subtitle: str, ic: str) -> str:
    return (f'<div class="sec"><div class="sec-ic">{icon(ic, 20)}</div>'
            f'<span class="sec-t">{e(title)}</span><span class="sec-s">{e(subtitle)}</span></div>')


# ---------------------------------------------------------------- Parsed Query
def pretty_col(name: str) -> str:
    return str(name).replace("_", " ").strip().capitalize() if str(name).islower() else str(name).replace("_", " ")


def pretty_filter(f: dict[str, Any]) -> str:
    v = f["value"]
    vs = ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
    return f'{pretty_col(f["column"])} {OP_PRETTY.get(f["op"], f["op"])} {vs}'


def chips_html(filters: Iterable[dict[str, Any]]) -> str:
    chips = [f'<span class="chip">{e(pretty_filter(f))}</span>' for f in filters]
    inner = "".join(chips) or '<span class="chip none">No filters</span>'
    return f'<div class="chips">{inner}</div>'


def _pq_row(label: str, value: str, on: bool = True) -> str:
    cls = "ck" if on else "ck off"
    return (f'<div class="pq-row"><span class="{cls}">{icon("check", 11, "currentColor")}</span>'
            f'<div><b>{e(label)}:</b> {e(value) if on else "<span class=muted>none</span>"}</div></div>')


def parsed_query_html(query: dict[str, Any], source: str) -> str:
    rows = []
    filters = query.get("filters") or []
    if filters:
        rows += [_pq_row("Filter", pretty_filter(f)) for f in filters]
    else:
        rows.append(_pq_row("Filter", "", on=False))
    outcome = query.get("outcome")
    rows.append(_pq_row("Outcome", OUTCOME_LABELS.get(outcome, str(outcome)), on=bool(outcome)))
    gb = query.get("group_by")
    rows.append(_pq_row("Group by", pretty_col(gb) if gb else "", on=bool(gb)))
    rows.append(_pq_row("Plot", PLOT_LABELS.get(query.get("plot_type"), str(query.get("plot_type")))))
    pill = f'<span class="pill">{e(SOURCE_LABELS.get(source, source))}</span>'
    return f'<div class="panel"><div class="panel-t">{icon("code", 18)}Parsed Query{pill}</div>{"".join(rows)}</div>'


def parsed_query_placeholder() -> str:
    return (f'<div class="panel"><div class="panel-t">{icon("code", 18)}Parsed Query</div>'
            f'<div class="muted" style="font-size:14px;line-height:1.5">Ask a question above and click '
            f'<b>Generate</b> to see how it is interpreted.</div></div>')


# ---------------------------------------------------------------- Cohort summary
def _is_female_col(s: pd.Series) -> bool:
    vals = {str(v).strip().lower() for v in s.dropna().unique()}
    return bool(vals & {"female", "f", "woman", "women"}) and len(vals) <= 4


def cohort_tiles(data: pd.DataFrame, total: int, event_col: str | None = None) -> list[dict[str, str]]:
    """Up to 3 tiles: patient count, median age, female share (without a sex column, falls back to event count / column count)."""
    tiles = [{"n": f"{len(data):,}", "l": "Patients", "sub": f"of {total:,}" if len(data) != total else ""}]
    age_col = next((c for c in data.columns if c.lower() == "age" and pd.api.types.is_numeric_dtype(data[c])), None)
    if age_col and data[age_col].notna().any():
        tiles.append({"n": f"{data[age_col].median():.1f}", "l": "Median age", "sub": ""})
    sex_col = next((c for c in data.columns if c.lower() in ("sex", "gender") and _is_female_col(data[c])), None)
    if sex_col:
        s = data[sex_col].dropna().astype(str).str.strip().str.lower()
        tiles.append({"n": f"{(s.isin(['female', 'f', 'woman', 'women'])).mean() * 100:.0f}%", "l": "Female", "sub": ""})
    elif event_col and event_col in data.columns:
        tiles.append({"n": f"{data[event_col].mean() * 100:.0f}%", "l": "Event rate", "sub": ""})
    else:
        tiles.append({"n": f"{data.shape[1]}", "l": "Columns", "sub": ""})
    return tiles[:3]


def tiles_html(tiles: list[dict[str, str]], title: str = "Cohort Summary", ic: str = "users") -> str:
    cells = "".join(
        f'<div class="tile t{i + 1}"><div class="n">{e(t["n"])}</div><div class="l">{e(t["l"])}</div>'
        f'{"<div class=sub>" + e(t["sub"]) + "</div>" if t["sub"] else ""}</div>'
        for i, t in enumerate(tiles))
    return (f'<div class="panel-t" style="margin-bottom:10px">{icon(ic, 18)}{e(title)}</div>'
            f'<div class="tiles">{cells}</div>')


def distribution_rows(groups: dict[str, pd.DataFrame]) -> list[tuple[str, int, float]]:
    total = sum(len(g) for g in groups.values()) or 1
    return [(str(k), len(g), len(g) / total * 100) for k, g in groups.items()]


def distribution_html(group_label: str, rows: list[tuple[str, int, float]]) -> str:
    body = "".join(f"<tr><td>{e(n)}</td><td>{c:,} ({p:.1f}%)</td></tr>" for n, c, p in rows)
    return (f'<div class="panel"><div class="panel-t">{icon("table", 18)}Cohort Distribution (n, %)</div>'
            f'<table class="dist"><tr><th>{e(group_label)}</th><th>N (%)</th></tr>{body}</table></div>')


def distribution_placeholder() -> str:
    return (f'<div class="panel"><div class="panel-t">{icon("table", 18)}Cohort Distribution (n, %)</div>'
            f'<div class="muted" style="font-size:14px">Group sizes appear here after you run a query.</div></div>')


# ---------------------------------------------------------------- Risk result
RISK_LEVELS = {"Low": ("low", "Low Risk"), "Medium": ("mod", "Moderate Risk"), "High": ("high", "High Risk")}


def risk_result_html(title: str, pct: float | None, level: str | None, details: str = "") -> str:
    if pct is None:
        return (f'<div class="risk empty"><div class="rt">{e(title)}</div><div class="rp">--%</div>'
                f'<div class="rd">Fill in the patient features and click <b>Predict Risk</b>.</div></div>')
    cls, label = RISK_LEVELS.get(level or "", ("mod", str(level)))
    return (f'<div class="risk"><div class="rt">{e(title)}</div><div class="rp">{pct:.0f}%</div>'
            f'<span class="badge {cls}">{e(label)}</span>'
            f'<div class="rd">{details}</div></div>')


def panel_title(text: str, ic: str) -> str:
    return f'<div class="panel-t" style="margin-bottom:10px">{icon(ic, 18)}{e(text)}</div>'
