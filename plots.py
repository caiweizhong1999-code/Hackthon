"""
plots.py
KM 曲线、分布图、趋势图。全部返回 matplotlib Figure,
可直接用于 Streamlit (st.pyplot(fig)) / Flask (fig_to_png_bytes) / Jupyter。

依赖: numpy, pandas, matplotlib;统计部分见 survival.py
KM 估计和 log-rank 检验在 survival.py 中(纯统计,不依赖 lifelines)。

主入口: render(result)  # 根据 QueryResult.plot_type 自动选图
"""
from __future__ import annotations

import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from survival import KMCurve, kaplan_meier, logrank_test

PALETTE = ["#2563eb", "#7c3aed", "#16a34a", "#ea580c", "#0891b2", "#ca8a04", "#db2777", "#64748b"]


# ============================================================================
# 通用小工具
# ============================================================================
def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _default_value_col(result: Any) -> str:
    """没指定要画哪一列时的默认选择:有 outcome 画时间列,否则取第一个数值列。"""
    if getattr(result, "time_col", None):
        return result.time_col
    num = result.data.select_dtypes("number").columns
    if len(num) == 0:
        raise ValueError("No numeric column available to plot")
    return num[0]


def fig_to_png_bytes(fig: Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


# ============================================================================
# 1. KM 曲线(带 CI、删失标记、风险人数表、log-rank p、中位生存)
# ============================================================================
def plot_km(
    groups: dict[str, pd.DataFrame],
    time_col: str,
    event_col: str,
    title: str = "Kaplan-Meier Survival Curve",
    time_unit: str = "months",
    show_ci: bool = True,
    show_censor: bool = True,
    show_risk_table: bool = True,
) -> Figure:
    curves: dict[str, KMCurve] = {}
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, g in groups.items():
        t, e = g[time_col].to_numpy(), g[event_col].to_numpy()
        curves[name] = kaplan_meier(t, e)
        raw[name] = (t, e)

    if show_risk_table:
        fig, (ax, ax_tbl) = plt.subplots(
            2, 1, figsize=(8, 6.2), gridspec_kw={"height_ratios": [4, 1], "hspace": 0.35}, sharex=True
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax_tbl = None

    x_max = max(float(np.max(raw[n][0])) for n in raw)
    for i, (name, c) in enumerate(curves.items()):
        col = _color(i)
        label = f"{name} (n={c.n}, events={c.events})"
        ax.step(c.times, c.surv, where="post", color=col, lw=2, label=label)
        if show_ci and c.n > 1:
            ax.fill_between(c.times, c.lower, c.upper, step="post", color=col, alpha=0.15, lw=0)
        if show_censor and len(c.censor_times):
            ax.plot(c.censor_times, c.censor_surv, "|", color=col, ms=8, mew=1.2)
        # 末尾延伸到最大随访时间
        ax.hlines(c.surv[-1], c.times[-1], x_max, color=col, lw=2)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, x_max * 1.02)
    ax.set_ylabel("Survival probability")
    ax.set_title(title, fontweight="bold")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    # 注释:log-rank + 各组中位生存
    notes = []
    if len(raw) >= 2:
        lr = logrank_test(raw)
        if lr.p_value == lr.p_value:   # 不是 nan
            notes.append(f"Log-rank {_fmt_p(lr.p_value)}")
    for name, c in curves.items():
        med = "NR" if np.isnan(c.median) else f"{c.median:.1f}"
        notes.append(f"Median {name}: {med} {time_unit}")
    ax.text(0.02, 0.04, "\n".join(notes), transform=ax.transAxes, fontsize=8.5,
            va="bottom", bbox=dict(boxstyle="round", fc="white", ec="#cbd5e1", alpha=0.9))

    if ax_tbl is not None:
        raw_step = x_max / 5
        mag = 10 ** np.floor(np.log10(raw_step))
        step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw_step)
        ticks = np.arange(0, x_max + 1e-9, step)
        ax.set_xticks(ticks)
        ax_tbl.set_xlim(ax.get_xlim())
        ax_tbl.set_ylim(-0.5, len(curves) - 0.5)
        ax_tbl.set_yticks(range(len(curves)))
        ax_tbl.set_yticklabels(list(curves)[::-1], fontsize=8)
        ax_tbl.tick_params(axis="y", pad=14)
        for i, (name, (t, _)) in enumerate(raw.items()):
            y = len(curves) - 1 - i
            for tk in ticks:
                ax_tbl.text(tk, y, str(int(np.sum(t >= tk))), ha="center", va="center",
                            fontsize=8, color=_color(i))
        ax_tbl.set_title("Number at risk", fontsize=9, loc="left")
        ax_tbl.set_xlabel(f"Time ({time_unit})")
        for s in ("top", "right", "left"):
            ax_tbl.spines[s].set_visible(False)
        ax_tbl.tick_params(left=False)
    else:
        ax.set_xlabel(f"Time ({time_unit})")

    return fig


# ============================================================================
# 2. 分布图:直方图 / 箱线图 / 柱状图
# ============================================================================
def plot_histogram(groups: dict[str, pd.DataFrame], column: str, bins: int = 20,
                   title: str | None = None) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    all_vals = pd.concat([g[column] for g in groups.values()]).dropna()
    edges = np.histogram_bin_edges(all_vals, bins=bins)   # 各组共用分箱,便于比较
    multi = len(groups) > 1
    for i, (name, g) in enumerate(groups.items()):
        ax.hist(g[column].dropna(), bins=edges, alpha=0.55 if multi else 0.8,
                color=_color(i), label=f"{name} (n={g[column].notna().sum()})", edgecolor="white")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_title(title or f"Distribution of {column}", fontweight="bold")
    ax.grid(alpha=0.25)
    if multi:
        ax.legend(frameon=False)
    return fig


def plot_boxplot(groups: dict[str, pd.DataFrame], column: str, title: str | None = None) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(groups)
    data = [groups[n][column].dropna().to_numpy() for n in names]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=names, widths=0.55,
                    medianprops=dict(color="black", lw=1.5))
    for i, patch in enumerate(bp["boxes"]):
        patch.set(facecolor=_color(i), alpha=0.6)
    # 叠加抖动散点
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        ax.scatter(rng.normal(i, 0.05, len(d)), d, s=8, color="black", alpha=0.25, zorder=3)
    ax.set_ylabel(column)
    ax.set_title(title or f"{column} by group", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_bar(groups: dict[str, pd.DataFrame], column: str, title: str | None = None) -> Figure:
    """分类变量的计数柱状图(分组时为并列柱)。"""
    counts = pd.DataFrame({n: g[column].value_counts() for n, g in groups.items()}).fillna(0)
    counts = counts.sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(counts))
    w = 0.8 / counts.shape[1]
    for i, name in enumerate(counts.columns):
        ax.bar(x + i * w - 0.4 + w / 2, counts[name], width=w, color=_color(i), label=str(name))
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index.astype(str))
    ax.set_ylabel("Count")
    ax.set_xlabel(column)
    ax.set_title(title or f"Counts of {column}", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    if counts.shape[1] > 1:
        ax.legend(frameon=False)
    return fig


# ============================================================================
# 3. 趋势图 / 散点图
# ============================================================================
def plot_trend(groups: dict[str, pd.DataFrame], x: str, y: str, bins: int = 6,
               title: str | None = None) -> Figure:
    """x 分箱后画 y 的均值 ± 95% CI(例如 年龄段 vs 平均生存时间 / 事件率)。"""
    all_x = pd.concat([g[x] for g in groups.values()]).dropna()
    edges = np.unique(np.quantile(all_x, np.linspace(0, 1, bins + 1)))  # 等频分箱
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, g) in enumerate(groups.items()):
        d = g[[x, y]].dropna()
        cut = pd.cut(d[x], bins=edges, include_lowest=True)
        agg = d.groupby(cut, observed=True).agg(
            xm=(x, "mean"), ym=(y, "mean"), sd=(y, "std"), n=(y, "size"))
        ci = 1.96 * agg["sd"].fillna(0) / np.sqrt(agg["n"])
        ax.errorbar(agg["xm"], agg["ym"], yerr=ci, marker="o", capsize=3, lw=2,
                    color=_color(i), label=f"{name} (n={len(d)})")
    ax.set_xlabel(x)
    ax.set_ylabel(f"Mean {y}")
    ax.set_title(title or f"{y} trend across {x}", fontweight="bold")
    ax.grid(alpha=0.25)
    if len(groups) > 1:
        ax.legend(frameon=False)
    return fig


def plot_scatter(groups: dict[str, pd.DataFrame], x: str, y: str, title: str | None = None) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, g) in enumerate(groups.items()):
        ax.scatter(g[x], g[y], s=18, alpha=0.6, color=_color(i), label=f"{name} (n={len(g)})")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} vs {x}", fontweight="bold")
    ax.grid(alpha=0.25)
    if len(groups) > 1:
        ax.legend(frameon=False)
    return fig


# ============================================================================
# 总入口:按 QueryResult.plot_type 分发
# ============================================================================
def render(result: Any, value_col: str | None = None, x_col: str | None = None,
           time_unit: str = "months") -> Figure:
    """
    result: query_engine.QueryResult
    value_col: 直方图/箱线图/柱状图/趋势图 y 轴要画的列(默认: outcome 时间列或第一个数值列)
    x_col:     趋势图/散点图的 x 轴列(默认: 第一个数值 filter 列,没有则 'age')
    """
    pt = result.plot_type
    groups = result.groups
    title_suffix = f" — by {result.group_by}" if result.group_by else ""

    if pt == "km_curve":
        if not result.time_col:
            raise ValueError("km_curve requires an outcome")
        return plot_km(groups, result.time_col, result.event_col,
                       title=f"{result.outcome.replace('_', ' ').capitalize()}{title_suffix}", time_unit=time_unit)

    col = value_col or _default_value_col(result)
    if pt == "histogram":
        return plot_histogram(groups, col)
    if pt == "boxplot":
        return plot_boxplot(groups, col)
    if pt == "bar":
        return plot_bar(groups, value_col or result.group_by or col)
    if pt in ("scatter", "trend"):
        x = x_col or ("age" if "age" in result.data.columns else result.data.select_dtypes("number").columns[0])
        return plot_trend(groups, x, col) if pt == "trend" else plot_scatter(groups, x, col)
    raise ValueError(f"Unknown plot_type: {pt}")


# ----------------------------------------------------------------------------
# 自测:python plots.py  -> 在当前目录输出 demo_*.png
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    from query_engine import run_query

    rng = np.random.default_rng(1)
    n = 400
    trt = rng.choice(["Drug A", "Drug B", "Placebo"], n)
    scale = pd.Series(trt).map({"Drug A": 36, "Drug B": 28, "Placebo": 18}).to_numpy()
    t = rng.exponential(scale)
    cens = rng.uniform(0, 60, n)
    demo = pd.DataFrame({
        "age": rng.integers(30, 90, n),
        "sex": rng.choice(["Male", "Female"], n),
        "treatment": trt,
        "stage": rng.choice(["I", "II", "III", "IV"], n),
        "os_time": np.minimum(t, cens).round(1),
        "os_event": (t <= cens).astype(int),
    })

    base = {"filters": [{"column": "age", "op": ">", "value": 50}], "outcome": "overall_survival",
            "group_by": "treatment"}
    cases = {
        "demo_km": dict(base, plot_type="km_curve"),
        "demo_hist": dict(base, plot_type="histogram"),
        "demo_box": dict(base, plot_type="boxplot"),
        "demo_trend": dict(base, plot_type="trend"),
        "demo_bar": dict(base, plot_type="bar", group_by="sex"),
    }
    for name, q in cases.items():
        res = run_query(demo, q)
        fig = render(res, value_col="stage" if q["plot_type"] == "bar" else None)
        fig.savefig(f"{name}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("saved", name)
