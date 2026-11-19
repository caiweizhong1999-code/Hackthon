"""
app.py — Survival Explorer (Streamlit)

运行:  streamlit run app.py
目录:  app.py 放项目根目录;其余模块放 src/(或和 app.py 同级)均可。

流程: 上传 CSV(或加载演示数据) -> 输入问题 -> Parsed Query 面板 -> KM/分布/趋势图 + 汇总表
      -> Risk model 页:XGBoost 训练 + SHAP + 个体生存曲线
"""
from __future__ import annotations

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

from model import SurvivalXGB, plot_shap_bar, plot_shap_local, plot_shap_summary  # noqa: E402
from pipeline import (  # noqa: E402
    available_outcomes, check_data, make_demo_data, run_parsed, run_question,
)
from plots import fig_to_png_bytes  # noqa: E402

st.set_page_config(page_title="Survival Explorer", layout="wide")

EXAMPLES = [
    "Show overall survival for patients older than 65, split by treatment",
    "65岁以上女性患者按治疗方案分组的总生存曲线",
    "Progression-free survival for stage III or IV patients by sex",
    "Distribution of age by sex",
    "Compare Drug A and Placebo survival time as a boxplot",
    "How does mean survival time change with age?",
]


# ----------------------------------------------------------------------------
# 数据加载:如果 data_loader.py 已有「上传 + 校验 + 类型识别」函数,只改这一个函数
# ----------------------------------------------------------------------------
def load_dataframe(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)


def _set_df(df: pd.DataFrame, name: str) -> None:
    st.session_state["df"] = df
    st.session_state["df_name"] = name
    for k in ("out", "model", "shap", "shap_df", "query_json", "risk_out"):   # 换数据就清掉旧结果
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


# ============================================================================
# 侧边栏
# ============================================================================
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.button("Load demo data", on_click=_use_demo)
    if uploaded is not None:
        sig = (uploaded.name, getattr(uploaded, "size", None))
        if st.session_state.get("upload_sig") != sig:
            try:
                _set_df(load_dataframe(uploaded), uploaded.name)
                st.session_state["upload_sig"] = sig
            except Exception as e:   # 任何读取/校验异常都友好提示
                st.error(f"Could not read the file: {e}")

    st.header("Parsing")
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = st.text_input("Anthropic API key", type="password", value=env_key)
    use_llm = st.checkbox("Use LLM parsing", value=bool(env_key))
    st.caption("Off, or on any LLM error → rule-based parser (works offline).")

    st.header("Display")
    time_unit = st.text_input("Time unit", value="months")
    landmarks = _parse_landmarks(st.text_input("Landmark times (for S(t))", value="12, 24"))

    st.header("Examples")
    for ex in EXAMPLES:
        st.button(ex, key=f"ex_{ex}", on_click=_set_question, args=(ex,))


# ============================================================================
# 数据就绪检查
# ============================================================================
st.title("Survival Explorer")
df = st.session_state.get("df")
if df is None:
    st.info("Upload a CSV or click **Load demo data** in the sidebar to start.")
    st.stop()

errors, data_warnings = check_data(df)
st.caption(f"Dataset: **{st.session_state.get('df_name')}** — {len(df)} rows × {df.shape[1]} columns · "
           f"outcomes: {', '.join(available_outcomes(df)) or 'none'}")
if errors:
    for e in errors:
        st.error(e)
    st.stop()
if data_warnings:
    with st.expander(f"{len(data_warnings)} data warning(s)"):
        for w in data_warnings:
            st.write("• " + w)


def get_client():
    if not (use_llm and api_key):
        return None
    try:
        return _make_client(api_key)
    except Exception as e:
        st.sidebar.warning(f"LLM unavailable, using rules: {e}")
        return None


# ============================================================================
# 提问
# ============================================================================
question = st.text_input("Ask a question about the cohort", key="question",
                         placeholder="e.g. overall survival for patients older than 65, by treatment")
if st.button("Analyze", type="primary") and question.strip():
    with st.spinner("Parsing and analyzing…"):
        out = run_question(question, df, client=get_client(), use_llm=use_llm,
                           time_unit=time_unit, landmarks=landmarks)
    st.session_state["out"] = out
    if out.parsed is not None:
        st.session_state["query_json"] = json.dumps(out.parsed.query, ensure_ascii=False, indent=2)

out = st.session_state.get("out")
left, right = st.columns([1, 2])

# ----------------------------------------------------------------------------
# 左:Parsed Query 面板(可编辑后重跑)
# ----------------------------------------------------------------------------
with left:
    st.subheader("Parsed Query")
    if out is None or out.parsed is None:
        st.caption("Ask a question to see how it is interpreted.")
        if out is not None and out.error:
            st.error(out.error)
    else:
        label = {"llm": "LLM", "rules": "Rule-based fallback", "manual": "Manually edited"}[out.parsed.source]
        st.caption(f"Parsed by: **{label}**")
        if out.parsed.notes:
            with st.expander("Parser notes"):
                for n in out.parsed.notes:
                    st.write("• " + n)
        st.text_area("Edit and re-run", key="query_json", height=280)
        if st.button("Re-run edited query"):
            try:
                edited = json.loads(st.session_state["query_json"])
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
            else:
                st.session_state["out"] = run_parsed(edited, df, question=question, source="manual",
                                                     time_unit=time_unit, landmarks=landmarks)
                st.rerun()

# ----------------------------------------------------------------------------
# 右:结果
# ----------------------------------------------------------------------------
with right:
    tab_plot, tab_table, tab_model, tab_data = st.tabs(["Plot", "Summary", "Risk model (XGBoost + SHAP)", "Data"])

    with tab_plot:
        if out is None:
            st.info("Run a question to see the plot.")
        elif out.error:
            st.error(out.error)
        else:
            res = out.result
            st.caption(f"{res.n_selected} of {res.n_total} patients selected"
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
            rep = out.report
            if rep.logrank is not None:
                st.markdown(f"**Log-rank test:** {rep.logrank}")
            summary = rep.summary.copy()
            st.dataframe(summary.round(3).astype(object).where(summary.notna(), "NR"))
            st.caption("NR = not reached. Median CI and S(t) use log-log 95% confidence intervals.")
            if rep.pairwise is not None:
                st.markdown("**Pairwise log-rank (Bonferroni-adjusted)**")
                st.dataframe(rep.pairwise.round(4))

    # ------------------------------------------------------------------ 风险模型
    with tab_model:
        outcomes = available_outcomes(df)
        sel = st.selectbox("Outcome to model", outcomes)
        if st.button("Train XGBoost model"):
            with st.spinner("Training…"):
                try:
                    m = SurvivalXGB(sel).fit(df)
                    sample = df.sample(min(len(df), 2000), random_state=0)
                    st.session_state["model"] = m
                    st.session_state["shap_df"] = sample
                    st.session_state["shap"] = m.shap_values(sample)
                    st.session_state.pop("risk_out", None)
                except (ImportError, ValueError) as e:
                    st.error(str(e))

        m = st.session_state.get("model")
        if m is None:
            st.info("Train a model to see risk predictions and SHAP explanations.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Patients / events", f"{int(m.metrics_['n'])} / {int(m.metrics_['events'])}")
            c2.metric("C-index (validation)", f"{m.metrics_.get('c_index_val', float('nan')):.3f}")
            c3.metric("C-index (train)", f"{m.metrics_['c_index_train']:.3f}")
            st.caption("Internal random-split validation only — not a substitute for external or temporal "
                       "validation. SHAP shows what the model uses, not causal effects.")

            sr, sample = st.session_state["shap"], st.session_state["shap_df"]
            g1, g2 = st.columns(2)
            with g1:
                fig = plot_shap_summary(sr); st.pyplot(fig); plt.close(fig)
            with g2:
                fig = plot_shap_bar(sr); st.pyplot(fig); plt.close(fig)

            st.markdown("**Single patient**")
            i = int(st.number_input("Row in the SHAP sample", min_value=0, max_value=len(sample) - 1, value=0, step=1))
            fig = plot_shap_local(sr, i); st.pyplot(fig); plt.close(fig)
            st.dataframe(m.predict_survival(sample.iloc[[i]], times=landmarks).round(3))

            st.markdown("**Risk groups → Kaplan-Meier**")
            if st.button("Plot KM by predicted risk group"):
                df_r = df.copy()
                df_r["risk_group"] = m.risk_group(df_r)
                q = {"filters": [], "outcome": m.outcome, "group_by": "risk_group", "plot_type": "km_curve"}
                st.session_state["risk_out"] = run_parsed(q, df_r, source="manual",
                                                          time_unit=time_unit, landmarks=landmarks)
            ro = st.session_state.get("risk_out")
            if ro is not None:
                if ro.error:
                    st.error(ro.error)
                else:
                    st.pyplot(ro.fig)
                    st.caption("Risk groups are tertiles of predicted risk, computed on the same data the model "
                               "was trained on, so separation is optimistic.")

    with tab_data:
        st.dataframe(df.head(200))
        st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                                   "unique": df.nunique()}))
