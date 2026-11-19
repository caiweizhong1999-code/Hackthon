"""
app.py — AI-Powered Clinical Data Explorer (Streamlit)

Run:    streamlit run app.py
Layout: put app.py in the project root; the other modules can go in src/ (or alongside app.py).
Theme:  put streamlit_config.toml at .streamlit/config.toml (light theme + dark sidebar).

Flow:   upload a CSV (or load demo data) -> enter a question -> Parsed Query -> KM/distribution/trend plot + survival summary + Table 1
      -> -> Patient Risk Prediction: XGBoost training + single-patient risk prediction + SHAP explanation
"""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib.pyplot as plt  # noqa: E402

import ui  # noqa: E402
from model import SurvivalXGB, plot_shap_bar, plot_shap_local, plot_shap_summary  # noqa: E402
from pipeline import (  # noqa: E402
    available_outcomes, check_data, make_demo_data, run_parsed, run_question,
)
from data_loader import infer_column_types  # noqa: E402
from plots import fig_to_png_bytes  # noqa: E402
from query_engine import OUTCOME_MAP, configure_outcomes  # noqa: E402
from table1 import baseline_table, survival_table  # noqa: E402

st.set_page_config(page_title="AI Clinical Data Explorer", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

PAGES = ["Overview", "Cohort Explorer", "Risk Prediction", "About"]
EXAMPLES = [
    "Show overall survival for patients older than 65, split by treatment",
    "Overall survival for women aged 40 to 60, grouped by stage",
    "Progression-free survival for stage III or IV patients by treatment",
    "Distribution of age by sex",
    "Compare Drug A and Placebo survival time as a boxplot",
    "How does mean survival time change with age?",
]
EVENT_LABELS = {"overall_survival": "Mortality", "progression_free_survival": "Progression-or-death"}


def md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data loading: if data_loader.py already has an "upload + validation + type detection" function, change only this one
# ----------------------------------------------------------------------------
def load_dataframe(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)


def _set_df(df: pd.DataFrame, name: str) -> None:
    st.session_state["df"] = df
    st.session_state["df_name"] = name
    for k in ("out", "model", "shap", "shap_df", "query_json", "risk_out", "patient_pred",
              "sv_time", "sv_event"):   # when the data changes, clear old results and manually chosen survival columns
        st.session_state.pop(k, None)


def _use_demo() -> None:
    _set_df(make_demo_data(), "demo")


def _set_question(q: str) -> None:
    st.session_state["question"] = q


@st.cache_resource
def _make_client(api_key: str):
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def _parse_landmarks(text: str) -> tuple[float, ...]:
    try:
        vals = tuple(float(x) for x in text.replace("，", ",").split(",") if x.strip())
        return vals or (12.0, 24.0)
    except ValueError:
        return (12.0, 24.0)


def _clean_query_view(q) -> dict:
    """For panel display: tolerate missing fields / bad format in user-written JSON so the UI doesn't crash."""
    if not isinstance(q, dict):
        return {"filters": [], "outcome": None, "group_by": None, "plot_type": "?"}
    filters = [f for f in (q.get("filters") or []) if isinstance(f, dict) and {"column", "op", "value"} <= f.keys()]
    return {**q, "filters": filters}


# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    md(ui.logo_html())
    md(ui.side_label("Navigation"))
    page = st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")

    md(ui.side_label("Dataset"))
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    st.button("Load demo data", key="demo_side", on_click=_use_demo)
    if uploaded is not None:
        sig = (uploaded.name, getattr(uploaded, "size", None))
        if st.session_state.get("upload_sig") != sig:
            try:
                _set_df(load_dataframe(uploaded), uploaded.name)
                st.session_state["upload_sig"] = sig
            except Exception as e:   # any read/validation exception gets a friendly message
                st.error(f"Could not read the file: {e}")

    md(ui.side_label("Settings"))
    with st.expander("Parsing & display"):
        env_key = os.environ.get("ANTHROPIC_API_KEY", "")
        api_key = st.text_input("Anthropic API key", type="password", value=env_key)
        use_llm = st.checkbox("Use LLM parsing", value=bool(env_key))
        st.caption("Off, or on any LLM error → rule-based parser (works offline).")
        time_unit = st.text_input("Time unit", value="months")
        landmarks = _parse_landmarks(st.text_input("Landmark times for S(t)", value="12, 24"))

    # Survival columns: auto-detect first (os_time/os_event, time_months/event, etc.); if that fails let the user pick manually
    _cur = st.session_state.get("df")
    if _cur is not None:
        configure_outcomes(_cur)
        _auto_ok = bool(available_outcomes(_cur))
        with st.expander("Survival columns", expanded=not _auto_ok):
            _num_cols = [c for c in _cur.columns if pd.api.types.is_numeric_dtype(_cur[c])
                         and not pd.api.types.is_bool_dtype(_cur[c])]
            _bin_cols = [c for c in _cur.columns if set(_cur[c].dropna().unique()) and
                         set(_cur[c].dropna().unique()) <= {0, 1}]
            _t_sel = st.selectbox("Time column", ["Auto-detect", *_num_cols], key="sv_time")
            _e_sel = st.selectbox("Event column (1 = event occurred)", ["Auto-detect", *_bin_cols], key="sv_event")
            _override = (_t_sel, _e_sel) if "Auto-detect" not in (_t_sel, _e_sel) else None
            configure_outcomes(_cur, _override)
            if available_outcomes(_cur):
                st.caption("Using: " + "; ".join(f"{ui.OUTCOME_LABELS.get(k, k)} = {t} / {e}"
                                                 for k, (t, e) in OUTCOME_MAP.items()))
            else:
                st.caption("No survival columns detected. Pick a numeric time column and a 0/1 event column.")


def get_client():
    if not (use_llm and api_key):
        return None
    try:
        return _make_client(api_key)
    except Exception as e:
        st.sidebar.warning(f"LLM unavailable, using rules: {e}")
        return None


# ============================================================================
# About page (needs no data)
# ============================================================================
def render_about() -> None:
    md(ui.section_header("About", "What this tool does and what it does not do", "sparkle"))
    with st.container(border=True):
        st.markdown(
            """
**What it does**
- Turns a plain-English question into a structured, validated query (filters, outcome, grouping, plot type).
- Draws Kaplan-Meier curves with log-rank tests, distributions, and trends for the selected cohort.
- Trains an XGBoost Cox model and explains individual predictions with SHAP.

**How questions are parsed**
- With an API key and *Use LLM parsing* on, the question is sent to the Anthropic API together with the
  column names, numeric ranges, and the values of low-cardinality categorical columns (e.g. sex, stage).
  Identifiers, dates and other high-cardinality columns are never listed. **Patient-level rows are not sent.**
- Every parsed query is validated against your data (columns, operators, values). If the LLM fails or returns
  something invalid, a rule-based parser takes over, so the tool keeps working offline.
- You can always read the parsed query and edit it before re-running.

**Limitations**
- Risk models are validated on a random internal split only; they are not a substitute for external or temporal
  validation and are **not for clinical decision-making**.
- SHAP shows what the model relies on, not causal effects.
- Predicted risk levels (Low / Moderate / High) are relative to the training cohort.
"""
        )


# ============================================================================
# Page header + data readiness check
# ============================================================================
md(ui.header_html())
if page == "About":
    render_about()
    st.stop()

# Reserve a placeholder for the step cards and fill it in after this run finishes submitting/training, using the final state (otherwise the completion marks lag one step behind)
steps_slot = st.empty() if page == "Overview" else None


def update_steps() -> None:
    if steps_slot is None:
        return
    o = st.session_state.get("out")
    steps_slot.markdown(
        ui.steps_html((st.session_state.get("df") is not None,
                       o is not None and o.error is None,
                       st.session_state.get("model") is not None)),
        unsafe_allow_html=True)


df = st.session_state.get("df")

if df is None:
    md('<div class="panel" style="margin-bottom:12px"><div class="panel-t">Get started</div>'
       '<div class="muted" style="font-size:14.5px;line-height:1.6">Upload a clinical CSV in the sidebar, or try the '
       'demo dataset. Your file needs a time column and a 0/1 event column for each outcome '
       '(e.g. <code>os_time</code> / <code>os_event</code>).</div></div>')
    st.button("Load demo data", key="demo_main", on_click=_use_demo)
    update_steps()
    st.stop()

errors, data_warnings = check_data(df)
if errors:
    for err in errors:
        st.error(err)
    update_steps()
    st.stop()
st.caption(f"Dataset: **{st.session_state.get('df_name')}** · {len(df):,} patients · {df.shape[1]} columns · "
           f"outcomes: {', '.join(available_outcomes(df))}")
if data_warnings:
    with st.expander(f"{len(data_warnings)} data warning(s)"):
        for w in data_warnings:
            st.write("• " + w)


# ============================================================================
# AI Cohort Explorer
# ============================================================================
def render_summary(out, res) -> None:
    """Survival summary: headline tiles, publication-style table, pairwise tests, raw numbers."""
    rep = out.report
    label = ui.OUTCOME_LABELS.get(res.outcome, res.outcome)
    all_df = res.data
    n_ev = int(all_df[res.event_col].sum())
    overall = rep.curves.get("All") if len(rep.curves) == 1 else None
    if overall is None:
        from survival import kaplan_meier
        overall = kaplan_meier(all_df[res.time_col], all_df[res.event_col])
    med = "NR" if pd.isna(overall.median) else f"{overall.median:.1f}"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Patients", f"{len(all_df):,}")
    m2.metric("Events", f"{n_ev:,}", f"{n_ev / len(all_df):.0%}", delta_color="off")
    m3.metric(f"Median ({time_unit})", med)
    m4.metric("Log-rank p", ("<0.001" if rep.logrank.p_value < 0.001 else f"{rep.logrank.p_value:.3f}")
              if rep.logrank is not None and pd.notna(rep.logrank.p_value) else "–")

    st.markdown(f"**{label}**" + (f" by **{res.group_by}**" if res.group_by else ""))
    st_tbl = survival_table(res.groups, res.time_col, res.event_col, landmarks=landmarks, time_unit=time_unit)
    st.dataframe(st_tbl, hide_index=True)
    st.download_button("Download summary (CSV)", st_tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="survival_summary.csv", mime="text/csv", key="dl_surv")
    st.caption("NR = not reached; NE = not estimable (beyond longest follow-up). Median follow-up uses reverse "
               "Kaplan-Meier. CIs are log-log 95%. HR is the log-rank (O−E)/V estimate against the first group "
               "(close to, but not the same as, a Cox model HR).")
    if rep.logrank is not None:
        st.markdown(f"**Log-rank test:** {rep.logrank}")
    if rep.pairwise is not None:
        st.markdown("**Pairwise log-rank (Bonferroni-adjusted)**")
        st.dataframe(rep.pairwise.round(4), hide_index=True)
    with st.expander("Raw numeric summary"):
        raw = rep.summary.copy()
        st.dataframe(raw.round(3).astype(object).where(raw.notna(), "NR"), hide_index=True)


def render_table1(df: pd.DataFrame, res) -> None:
    """Table 1 for the selected cohort, split by the query's group_by variable (if any)."""
    outcome_cols = tuple(c for pair in OUTCOME_MAP.values() for c in pair)
    t1 = baseline_table(res.data, res.group_by, types=infer_column_types(df), exclude=outcome_cols)
    st.markdown("**Table 1 · Baseline characteristics**" + (f" by **{res.group_by}**" if res.group_by else ""))
    st.dataframe(t1, hide_index=True)
    st.download_button("Download Table 1 (CSV)", t1.to_csv(index=False).encode("utf-8-sig"),
                       file_name="table1.csv", mime="text/csv", key="dl_t1")
    st.caption("Numeric: median [IQR] and mean ± SD (Mann-Whitney U / Kruskal-Wallis). Categorical: n (%) "
               "(Fisher exact for 2×2, else chi-square). p-values are descriptive and not adjusted for multiple "
               "testing.")


def render_explorer(df: pd.DataFrame) -> None:
    md(ui.section_header("AI Cohort Explorer", "Ask about your cohort in plain English.", "chart"))

    with st.form("ask"):
        c_in, c_btn = st.columns([8, 1.4])
        with c_in:
            question = st.text_input("Question", key="question", label_visibility="collapsed",
                                     placeholder="e.g. Show overall survival by treatment group for patients older than 65")
        with c_btn:
            submitted = st.form_submit_button("Generate", type="primary")
    if submitted and question.strip():
        with st.spinner("Parsing and analyzing…"):
            new_out = run_question(question, df, client=get_client(), use_llm=use_llm,
                                   time_unit=time_unit, landmarks=landmarks)
        st.session_state["out"] = new_out
        if new_out.parsed is not None:
            st.session_state["query_json"] = json.dumps(new_out.parsed.query, ensure_ascii=False, indent=2)

    with st.expander("Example questions"):
        for ex in EXAMPLES:
            st.button(ex, key=f"ex_{ex}", on_click=_set_question, args=(ex,))

    out = st.session_state.get("out")
    if out is not None and out.error:
        st.error(out.error)
    res = out.result if out is not None else None

    # ---- Three info panels:Parsed Query | Cohort Summary | Cohort Distribution ----
    p_left, p_mid, p_right = st.columns([1.1, 1.35, 1.5])
    with p_left:
        if out is not None and out.parsed is not None:
            md(ui.parsed_query_html(_clean_query_view(out.parsed.query), out.parsed.source))
        else:
            md(ui.parsed_query_placeholder())
    with p_mid:
        data = res.data if res is not None else df
        event_col = res.event_col if res is not None else None
        md(ui.tiles_html(ui.cohort_tiles(data, len(df), event_col)))
    with p_right:
        if res is not None:
            label = ui.pretty_col(res.group_by) if res.group_by else "Cohort"
            md(ui.distribution_html(label, ui.distribution_rows(res.groups) if res.group_by
                                    else [("All selected patients", res.n_selected, 100.0)]))
        else:
            md(ui.distribution_placeholder())

    # ---- Plot / summary / data   |   query editor ----
    c_plot, c_edit = st.columns([1.9, 1])
    with c_plot:
        with st.container(border=True):
            tab_plot, tab_table, tab_t1, tab_data = st.tabs(["Plot", "Summary", "Table 1", "Data"])
            with tab_plot:
                if out is None:
                    st.info("Run a question to see the plot.")
                elif out.error:
                    st.info("No plot to show. Check the message above or edit the query on the right.")
                else:
                    st.caption(f"{res.n_selected:,} of {res.n_total:,} patients selected"
                               + (f" · grouped by **{res.group_by}**" if res.group_by else ""))
                    for w in out.warnings:
                        st.warning(w)
                    st.pyplot(out.fig)
                    st.download_button("Download PNG", data=fig_to_png_bytes(out.fig),
                                       file_name="figure.png", mime="image/png")
            with tab_table:
                if out is None or out.error or out.report is None:
                    st.info("Survival statistics appear here for queries with an outcome.")
                else:
                    render_summary(out, res)
            with tab_t1:
                if out is None or out.error or res is None:
                    st.info("Run a question to see baseline characteristics of the selected cohort.")
                else:
                    render_table1(df, res)
            with tab_data:
                st.dataframe(df.head(200))
                st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                                           "unique": df.nunique()}))

    with c_edit:
        with st.container(border=True):
            md(ui.panel_title("Query editor", "code"))
            if out is None or out.parsed is None:
                st.caption("After you generate a query you can review and edit it here.")
            else:
                md(ui.chips_html(_clean_query_view(out.parsed.query)["filters"]))
                st.text_area("Edit the query (JSON) and re-run", key="query_json", height=270)
                if st.button("Re-run edited query"):
                    try:
                        edited = json.loads(st.session_state["query_json"])
                    except json.JSONDecodeError as exc:
                        st.error(f"Invalid JSON: {exc}")
                    else:
                        if not isinstance(edited, dict):
                            st.error("The query must be a JSON object.")
                        else:
                            st.session_state["out"] = run_parsed(
                                edited, df, question=st.session_state.get("question", ""),
                                source="manual", time_unit=time_unit, landmarks=landmarks)
                            st.rerun()
                if out.parsed.notes:
                    with st.expander("Parser notes"):
                        for n in out.parsed.notes:
                            st.write("• " + n)


# ============================================================================
# Patient Risk Prediction
# ============================================================================
def _field_widgets(m: SurvivalXGB, df: pd.DataFrame) -> dict:
    """Auto-generate the patient input form from model features (numeric -> number_input, categorical -> selectbox)."""
    values: dict = {}
    cols = st.columns(2)
    names = list(m.encoder.numeric) + list(m.encoder.categorical)
    for i, col in enumerate(names):
        label = ui.pretty_col(col)
        with cols[i % 2]:
            if col in m.encoder.numeric:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) and (s == s.round()).all():
                    values[col] = st.number_input(label, min_value=int(s.min()), max_value=int(s.max()),
                                                  value=int(s.median()), step=1, key=f"pf_{col}")
                else:
                    values[col] = st.number_input(label, min_value=float(s.min()), max_value=float(s.max()),
                                                  value=float(s.median()), key=f"pf_{col}")
            else:
                options = list(m.encoder.categorical[col])
                mode = df[col].astype(str).mode()
                idx = options.index(mode.iloc[0]) if len(mode) and mode.iloc[0] in options else 0
                values[col] = st.selectbox(label, options, index=idx, key=f"pf_{col}")
    return values


def _predict_patient(m: SurvivalXGB, df: pd.DataFrame, values: dict, horizon: float) -> dict:
    patient = pd.DataFrame([values])
    surv_h = float(m.predict_survival(patient, [horizon]).iloc[0, 0])
    lm = m.predict_survival(patient, landmarks).iloc[0]
    unit = html.escape(time_unit)     # this is rendered as HTML, so user input must be escaped
    parts = " · ".join(f"{t:g} {unit}: {v * 100:.0f}%" for t, v in zip(landmarks, lm.to_numpy()))
    details = f"Survival probability — {parts}<br>Risk level is relative to the training cohort."
    if horizon > float(df[m.time_col].max()):
        details += "<br><b>Horizon exceeds the longest follow-up in the data; interpret with caution.</b>"
    label = EVENT_LABELS.get(m.outcome, "Event")
    return {
        "pct": (1 - surv_h) * 100,
        "level": str(m.risk_group(patient).iloc[0]),
        "title": f"{label} risk within {horizon:g} {time_unit}",
        "details": details,
        "shap": m.shap_values(patient),
    }


def render_risk(df: pd.DataFrame) -> None:
    md(ui.section_header("Patient Risk Prediction",
                         "Predict individual risk (e.g., 2-year mortality) and explain key drivers using SHAP.", "target"))

    with st.container(border=True):
        c_sel, c_btn, c_met = st.columns([1.4, 0.9, 3])
        with c_sel:
            md('<div class="muted" style="font-size:13px;margin-bottom:4px">Outcome to model</div>')
            sel = st.selectbox("Outcome to model", available_outcomes(df), label_visibility="collapsed")
        with c_btn:
            md('<div style="height:26px"></div>')
            train = st.button("Train model", type="primary")
        if train:
            with st.spinner("Training XGBoost model…"):
                try:
                    m_new = SurvivalXGB(sel).fit(df)
                    sample = df.sample(min(len(df), 2000), random_state=0)
                    st.session_state["model"] = m_new
                    st.session_state["shap_df"] = sample
                    st.session_state["shap"] = m_new.shap_values(sample)
                    for k in ("risk_out", "patient_pred"):
                        st.session_state.pop(k, None)
                except (ImportError, ValueError) as exc:
                    st.error(str(exc))
        m = st.session_state.get("model")
        with c_met:
            if m is not None:
                tiles = [
                    {"n": f"{int(m.metrics_['n']):,}", "l": "Patients", "sub": f"{int(m.metrics_['events']):,} events"},
                    {"n": f"{m.metrics_.get('c_index_val', float('nan')):.3f}", "l": "C-index (validation)", "sub": ""},
                    {"n": f"{m.metrics_['c_index_train']:.3f}", "l": "C-index (train)", "sub": ""},
                ]
                md(ui.tiles_html(tiles, title=f"Model: {ui.OUTCOME_LABELS.get(m.outcome, m.outcome)}", ic="target"))

    m = st.session_state.get("model")
    if m is None:
        st.info("Train a model to unlock individual risk predictions and SHAP explanations.")
        return
    st.caption("Internal random-split validation only; not a substitute for external or temporal validation, and "
               "not for clinical decision-making. SHAP shows what the model uses, not causal effects.")

    pred = st.session_state.get("patient_pred")
    c_form, c_res, c_shap = st.columns([1.35, 1, 1.35])

    with c_form:
        with st.container(border=True):
            md(ui.panel_title("Input Patient Features", "user"))
            with st.form("patient"):
                values = _field_widgets(m, df)
                horizon = st.number_input(f"Prediction horizon ({time_unit})", min_value=1.0,
                                          value=max(1.0, float(landmarks[-1])) if landmarks else 24.0, key="horizon")
                go = st.form_submit_button("Predict Risk", type="primary")
            if go:
                st.session_state["patient_pred"] = pred = _predict_patient(m, df, values, float(horizon))

    with c_res:
        md(ui.panel_title("Prediction Result", "chart"))
        if pred is None:
            md(ui.risk_result_html("Predicted risk", None, None))
        else:
            md(ui.risk_result_html(pred["title"], pred["pct"], pred["level"], pred["details"]))

    with c_shap:
        with st.container(border=True):
            md(ui.panel_title("SHAP Explanation (Top Features)", "sparkle"))
            if pred is None:
                st.caption("Appears after you click Predict Risk.")
            else:
                fig = plot_shap_local(pred["shap"], 0, max_display=6)
                st.pyplot(fig)
                plt.close(fig)

    with st.expander("Model performance & global drivers"):
        sr = st.session_state["shap"]
        g1, g2 = st.columns(2)
        with g1:
            fig = plot_shap_summary(sr)
            st.pyplot(fig)
            plt.close(fig)
        with g2:
            fig = plot_shap_bar(sr)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("**Risk groups → Kaplan-Meier**")
        if st.button("Plot KM by predicted risk group"):
            df_r = df.copy()
            df_r["risk_group"] = m.risk_group(df_r)
            q = {"filters": [], "outcome": m.outcome, "group_by": "risk_group", "plot_type": "km_curve"}
            st.session_state["risk_out"] = run_parsed(q, df_r, source="manual", time_unit=time_unit,
                                                      landmarks=landmarks)
        ro = st.session_state.get("risk_out")
        if ro is not None:
            if ro.error:
                st.error(ro.error)
            else:
                st.pyplot(ro.fig)
                st.caption("Risk groups are tertiles of predicted risk, computed on the same data the model was "
                           "trained on, so separation is optimistic.")


if page in ("Overview", "Cohort Explorer"):
    render_explorer(df)
if page in ("Overview", "Risk Prediction"):
    render_risk(df)
update_steps()
