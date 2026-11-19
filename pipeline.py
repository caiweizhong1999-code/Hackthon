"""
pipeline.py
把 自然语言解析 -> 查询执行 -> 生存分析 -> 画图 串成一个函数,与 UI 无关(方便单独测试)。
app.py 只负责展示。

    out = run_question("65岁以上患者按治疗方案分组的总生存曲线", df, client=None)
    if out.error: ...          # 友好的错误信息,不抛异常
    out.parsed.query           # Parsed Query(dict)
    out.parsed.source          # "llm" | "rules" | "manual"
    out.report                 # survival.SurvivalReport(有 outcome 时)
    out.fig                    # matplotlib Figure
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from nl_parser_v2 import ParseOutput, parse_query
from plots import render
from query_engine import OUTCOME_MAP, QueryError, QueryResult, run_query
from survival import SurvivalReport, analyze

DEFAULT_LANDMARKS = (12, 24)


@dataclass
class PipelineOutput:
    parsed: ParseOutput | None = None
    result: QueryResult | None = None
    report: SurvivalReport | None = None
    fig: Any = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# 数据检查(上传后立刻做,比等到分析时才报错友好)
# ----------------------------------------------------------------------------
def available_outcomes(df: pd.DataFrame) -> list[str]:
    return [name for name, (t, e) in OUTCOME_MAP.items() if t in df.columns and e in df.columns]


def check_data(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """返回 (errors, warnings)。errors 非空时无法做任何分析。"""
    errors: list[str] = []
    warns: list[str] = []
    if df.empty:
        return ["The dataset is empty"], warns

    outcomes = available_outcomes(df)
    if not outcomes:
        return [("Could not find a survival time column and a 0/1 event column. Expected pairs such as "
                 "os_time/os_event or pfs_time/pfs_event, or columns like time_months/event. "
                 f"Found columns: {list(df.columns)}. Choose them under Settings → Survival columns in the sidebar.")], warns

    for name in outcomes:
        t_col, e_col = OUTCOME_MAP[name]
        if not pd.api.types.is_numeric_dtype(df[t_col]):
            errors.append(f"Time column '{t_col}' is not numeric (it may contain text); please clean it first")
        vals = set(df[e_col].dropna().unique())
        if not vals <= {0, 1}:
            errors.append(f"Event column '{e_col}' must be 0/1 (1 = event occurred); found values: {sorted(map(str, vals))[:6]}")
        elif pd.api.types.is_numeric_dtype(df[t_col]):
            if (df[t_col].dropna() < 0).any():
                errors.append(f"Time column '{t_col}' contains negative values")
            n_bad = int(df[[t_col, e_col]].isna().any(axis=1).sum())
            if n_bad:
                warns.append(f"{name}: {n_bad} rows with missing {t_col}/{e_col} will be dropped")
            if df[e_col].sum() < 10:
                warns.append(f"{name}: only {int(df[e_col].sum())} events; results will be unstable")
    if len(df) < 30:
        warns.append(f"Only {len(df)} patients in the dataset")
    return errors, warns


# ----------------------------------------------------------------------------
# 从问题里挑要画的列(Parsed Query schema 里没有这个字段,这里用简单启发式)
# ----------------------------------------------------------------------------
_COLUMN_HINTS = {"age": ["年龄"]}   # 列名 -> 中文说法


def _mentioned_numeric_cols(result: QueryResult, question: str) -> list[str]:
    q = question.lower()
    skip = {c for pair in OUTCOME_MAP.values() for c in pair}
    found = []
    for c in result.data.select_dtypes("number").columns:
        if c in skip:
            continue
        words = [c.lower(), *_COLUMN_HINTS.get(c.lower(), [])]
        if any(w in q for w in words):
            found.append(c)
    return found


def pick_columns(result: QueryResult, question: str) -> tuple[str | None, str | None]:
    """返回 (value_col, x_col);拿不准就返回 None,交给 plots.render 的默认逻辑。"""
    mentioned = _mentioned_numeric_cols(result, question)
    q = question.lower()
    wants_time = result.time_col and re_search(r"time|时间", q)
    if result.plot_type in ("trend", "scatter"):
        value = result.time_col if (result.time_col and (wants_time or not mentioned)) else (mentioned[0] if mentioned else None)
        x = next((c for c in mentioned if c != value), None)
        return value, x
    if mentioned and not wants_time:
        return mentioned[0], None
    return None, None


def re_search(pat: str, text: str) -> bool:
    import re
    return re.search(pat, text) is not None


# ----------------------------------------------------------------------------
# 执行
# ----------------------------------------------------------------------------
def run_parsed(query: dict[str, Any], df: pd.DataFrame, question: str = "", source: str = "manual",
               notes: list[str] | None = None, time_unit: str = "months",
               landmarks: Iterable[float] = DEFAULT_LANDMARKS) -> PipelineOutput:
    """执行一个已有的 Parsed Query(手动编辑 JSON 后重跑也走这里)。"""
    import matplotlib.pyplot as plt

    out = PipelineOutput(parsed=ParseOutput(query=query, source=source, notes=list(notes or [])))
    try:
        res = run_query(df, query)
    except (QueryError, ValueError) as e:
        out.error = str(e)
        return out
    out.result = res
    if any("fell back to the default query" in n for n in out.parsed.notes):
        out.warnings.append("Your question could not be fully interpreted, so the default survival curve is shown. "
                            "Check the Parser notes and edit the query on the right if needed.")
    out.warnings += res.warnings

    if res.outcome:
        try:
            out.report = analyze(res, landmarks=landmarks)
            out.warnings += out.report.warnings
        except ValueError as e:
            if res.plot_type == "km_curve":
                out.error = f"Survival analysis failed: {e}"
                return out
            out.warnings.append(f"Survival summary could not be generated: {e}")

    value_col, x_col = pick_columns(res, question)
    try:
        out.fig = render(res, value_col=value_col, x_col=x_col, time_unit=time_unit)
        plt.close(out.fig)          # 从 pyplot 注册表摘掉,避免 Streamlit 里反复运行后内存堆积
    except (QueryError, ValueError, KeyError) as e:
        out.error = f"Plot failed: {e}"
    return out


def run_question(question: str, df: pd.DataFrame, client: Any | None = None, use_llm: bool = True,
                 time_unit: str = "months", landmarks: Iterable[float] = DEFAULT_LANDMARKS) -> PipelineOutput:
    try:
        parsed = parse_query(question, df, client=client, use_llm=use_llm)
    except QueryError as e:
        return PipelineOutput(error=str(e))
    return run_parsed(parsed.query, df, question=question, source=parsed.source, notes=parsed.notes,
                      time_unit=time_unit, landmarks=landmarks)


# ----------------------------------------------------------------------------
# 演示数据:上台时不依赖现场上传
# ----------------------------------------------------------------------------
def make_demo_data(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 90, n)
    sex = rng.choice(["Male", "Female"], n)
    trt = rng.choice(["Drug A", "Drug B", "Placebo"], n)
    stage = rng.choice(["I", "II", "III", "IV"], n, p=[0.2, 0.3, 0.3, 0.2])
    lin = 0.025 * (age - 60) + 0.45 * np.isin(stage, ["III", "IV"]) - 0.45 * (trt == "Drug A") - 0.2 * (trt == "Drug B")
    t_os = rng.exponential(32 / np.exp(lin))
    t_pfs = t_os * rng.uniform(0.4, 0.9, n)
    cens = rng.uniform(5, 60, n)
    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "age": age, "sex": sex, "treatment": trt, "stage": stage,
        "os_time": np.minimum(t_os, cens).round(1).clip(min=0.1), "os_event": (t_os <= cens).astype(int),
        "pfs_time": np.minimum(t_pfs, cens).round(1).clip(min=0.1), "pfs_event": (t_pfs <= cens).astype(int),
    })


# ----------------------------------------------------------------------------
# 自测:python pipeline.py(不需要网络和 API key)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    df = make_demo_data()
    errs, warns = check_data(df)
    print("check_data:", errs, warns)
    assert not errs

    questions = [   # (问题, 期望的图类型)
        ("show overall survival for patients older than 65 by treatment", "km_curve"),
        ("65岁以上女性患者按治疗方案分组的总生存曲线", "km_curve"),
        ("progression-free survival, stage III or IV, split by sex", "km_curve"),
        ("distribution of age by sex", "histogram"),
        ("compare Drug A and Placebo survival as a boxplot", "boxplot"),
        ("How does mean survival time change with age?", "trend"),
        ("patients younger than 50", "histogram"),
    ]
    for q, expected in questions:
        out = run_question(q, df)
        assert out.error is None and out.fig is not None, (q, out.error)
        assert out.result.plot_type == expected, (
            f"图类型不符: {q!r} 期望 {expected},实际 {out.result.plot_type}。"
            f"请检查实际导入的 nl_parser_v2: python -c \"import nl_parser_v2; print(nl_parser_v2.__file__)\"")
        lr = out.report.logrank if out.report else None
        print(f"{q[:55]:<57} {out.parsed.source:<6} {out.result.plot_type:<10} "
              f"n={out.result.n_selected:<4} {lr if lr else ''}")

    # 筛不出样本 -> 友好错误,不抛异常
    out = run_question("patients older than 200", df)
    print("空结果:", out.error)
    assert out.error and out.fig is None

    # 手动编辑的 JSON 里有非法列 / 非法 op -> 友好错误
    for bad in ({"filters": [{"column": "agee", "op": ">", "value": 1}], "outcome": None, "group_by": None, "plot_type": "histogram"},
                {"filters": [{"column": "age", "op": "like", "value": 1}], "outcome": None, "group_by": None, "plot_type": "histogram"}):
        out = run_parsed(bad, df)
        print("手动编辑非法查询:", out.error)
        assert out.error

    # 数据检查:事件列不是 0/1 / 缺列
    broken = df.copy(); broken["os_event"] = broken["os_event"].map({0: "Alive", 1: "Dead"})
    print("坏数据:", check_data(broken)[0])
    assert check_data(broken)[0]
    print("缺列:", check_data(df.drop(columns=["os_time", "pfs_time"]))[0])
    print("\n全部通过")
