"""
query_engine.py
结构化查询 (Parsed Query) -> Pandas 筛选 / 分组

输入的 query 可以是 dict 或 Pydantic 模型(会自动 model_dump),格式:
{
  "filters": [{"column": "age", "op": ">", "value": 65}],
  "outcome": "overall_survival",
  "group_by": "treatment",
  "plot_type": "km_curve"
}

只依赖 pandas / numpy。校验失败抛出 QueryError,方便上层捕获后重试或走规则兜底。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 配置:按你们真实数据集改这里
# ----------------------------------------------------------------------------
ALLOWED_OPS = {">", "<", ">=", "<=", "==", "!=", "in"}
ALLOWED_PLOTS = {"km_curve", "boxplot", "histogram", "bar", "scatter"}

# outcome 名称 -> (时间列, 事件列);事件列 1=发生事件, 0=删失
OUTCOME_MAP: dict[str, tuple[str, str]] = {
    "overall_survival": ("os_time", "os_event"),
    "progression_free_survival": ("pfs_time", "pfs_event"),
}

# 数值列做分组时,最多允许的唯一值个数(超过则要求先分箱)
MAX_NUMERIC_GROUPS = 8


class QueryError(ValueError):
    """查询不合法(列不存在、op 不在白名单、值类型不符等)。"""


# ----------------------------------------------------------------------------
# 结果容器
# ----------------------------------------------------------------------------
@dataclass
class QueryResult:
    data: pd.DataFrame                       # 筛选后的数据
    groups: dict[str, pd.DataFrame]          # 分组名 -> 子数据框
    outcome: str | None
    time_col: str | None
    event_col: str | None
    plot_type: str
    group_by: str | None
    n_total: int                             # 筛选前行数
    n_selected: int                          # 筛选后行数
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """给前端展示 / 日志用的简要信息。"""
        return {
            "n_total": self.n_total,
            "n_selected": self.n_selected,
            "groups": {k: len(v) for k, v in self.groups.items()},
            "outcome": self.outcome,
            "plot_type": self.plot_type,
            "warnings": self.warnings,
        }


# ----------------------------------------------------------------------------
# 校验
# ----------------------------------------------------------------------------
def _to_dict(query: Any) -> dict[str, Any]:
    if isinstance(query, dict):
        return query
    if hasattr(query, "model_dump"):      # pydantic v2
        return query.model_dump()
    if hasattr(query, "dict"):            # pydantic v1
        return query.dict()
    raise QueryError(f"不支持的 query 类型: {type(query).__name__}")


def _match_category(series: pd.Series, value: Any, column: str) -> Any:
    """分类变量:大小写不敏感地匹配到数据里真实存在的取值。"""
    lookup = {str(v).lower(): v for v in series.dropna().unique()}
    key = str(value).lower()
    if key not in lookup:
        raise QueryError(
            f"列 '{column}' 中没有取值 '{value}',可选值: {sorted(map(str, lookup.values()))}"
        )
    return lookup[key]


def _coerce_value(df: pd.DataFrame, column: str, op: str, value: Any) -> Any:
    series = df[column]
    is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

    if op == "in":
        if not isinstance(value, (list, tuple, set)):
            raise QueryError(f"op 'in' 需要列表作为 value,收到: {value!r}")
        values = list(value)
    else:
        if isinstance(value, (list, tuple, set)):
            raise QueryError(f"op '{op}' 需要单个值,收到列表: {value!r}")
        values = [value]

    coerced = []
    for v in values:
        if is_numeric:
            if isinstance(v, bool) or not isinstance(v, (int, float, np.number, str)):
                raise QueryError(f"数值列 '{column}' 的比较值必须是数字,收到: {v!r}")
            try:
                coerced.append(float(v))
            except ValueError:
                raise QueryError(f"数值列 '{column}' 的比较值必须是数字,收到: {v!r}")
        else:
            if op in {">", "<", ">=", "<="}:
                raise QueryError(f"分类列 '{column}' 不支持比较运算 '{op}'")
            coerced.append(_match_category(series, v, column))

    return coerced if op == "in" else coerced[0]


def validate_query(df: pd.DataFrame, query: Any) -> dict[str, Any]:
    """校验并规范化 query,返回清洗后的 dict。不合格抛 QueryError。"""
    q = _to_dict(query)
    cols = set(df.columns)
    clean: dict[str, Any] = {"filters": []}

    # filters
    for i, f in enumerate(q.get("filters") or []):
        column, op, value = f.get("column"), f.get("op"), f.get("value")
        if column not in cols:
            raise QueryError(f"filters[{i}]: 列 '{column}' 不存在,可用列: {sorted(cols)}")
        if op not in ALLOWED_OPS:
            raise QueryError(f"filters[{i}]: op '{op}' 不在白名单 {sorted(ALLOWED_OPS)}")
        clean["filters"].append(
            {"column": column, "op": op, "value": _coerce_value(df, column, op, value)}
        )

    # outcome
    outcome = q.get("outcome")
    if outcome is not None:
        if outcome not in OUTCOME_MAP:
            raise QueryError(f"未知 outcome '{outcome}',可选: {sorted(OUTCOME_MAP)}")
        t_col, e_col = OUTCOME_MAP[outcome]
        missing = [c for c in (t_col, e_col) if c not in cols]
        if missing:
            raise QueryError(f"outcome '{outcome}' 需要的列在数据中缺失: {missing}")
    clean["outcome"] = outcome

    # group_by
    group_by = q.get("group_by")
    if group_by is not None and group_by not in cols:
        raise QueryError(f"group_by 列 '{group_by}' 不存在,可用列: {sorted(cols)}")
    clean["group_by"] = group_by

    # plot_type
    plot_type = q.get("plot_type") or ("km_curve" if outcome else "histogram")
    if plot_type not in ALLOWED_PLOTS:
        raise QueryError(f"未知 plot_type '{plot_type}',可选: {sorted(ALLOWED_PLOTS)}")
    if plot_type == "km_curve" and outcome is None:
        raise QueryError("km_curve 需要指定 outcome")
    clean["plot_type"] = plot_type

    return clean


# ----------------------------------------------------------------------------
# 筛选 / 分组
# ----------------------------------------------------------------------------
def build_mask(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.Series:
    """多个 filter 之间是 AND 关系。缺失值一律不满足条件。"""
    mask = pd.Series(True, index=df.index)
    for f in filters:
        s, op, v = df[f["column"]], f["op"], f["value"]
        if op == ">":
            m = s > v
        elif op == "<":
            m = s < v
        elif op == ">=":
            m = s >= v
        elif op == "<=":
            m = s <= v
        elif op == "==":
            m = s == v
        elif op == "!=":
            m = s.notna() & (s != v)   # NaN != x 不应被当成"满足"
        elif op == "in":
            m = s.isin(v)
        else:  # 理论上 validate 已拦截
            raise QueryError(f"不支持的 op: {op}")
        mask &= m.fillna(False)
    return mask


def split_groups(df: pd.DataFrame, group_by: str | None) -> dict[str, pd.DataFrame]:
    if group_by is None:
        return {"All": df}

    s = df[group_by]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > MAX_NUMERIC_GROUPS:
        raise QueryError(
            f"数值列 '{group_by}' 有 {s.nunique()} 个不同取值,不适合直接分组(请先分箱,如年龄段)"
        )
    return {str(k): g for k, g in df.dropna(subset=[group_by]).groupby(group_by, sort=True)}


def run_query(df: pd.DataFrame, query: Any) -> QueryResult:
    """主入口:校验 -> 筛选 -> (outcome 列处理) -> 分组。"""
    q = validate_query(df, query)
    warnings: list[str] = []

    selected = df[build_mask(df, q["filters"])]
    if selected.empty:
        raise QueryError("筛选后没有任何样本,请放宽条件")

    time_col = event_col = None
    if q["outcome"]:
        time_col, event_col = OUTCOME_MAP[q["outcome"]]
        before = len(selected)
        selected = selected.dropna(subset=[time_col, event_col])
        dropped = before - len(selected)
        if dropped:
            warnings.append(f"{dropped} 行因 {time_col}/{event_col} 缺失被剔除")
        if selected.empty:
            raise QueryError(f"所选样本的 {time_col}/{event_col} 全部缺失")

    groups = split_groups(selected, q["group_by"])

    # 小样本组提醒(KM 曲线样本太少意义不大)
    for name, g in groups.items():
        if len(g) < 5:
            warnings.append(f"分组 '{name}' 仅 {len(g)} 例,结果可能不稳定")

    return QueryResult(
        data=selected,
        groups=groups,
        outcome=q["outcome"],
        time_col=time_col,
        event_col=event_col,
        plot_type=q["plot_type"],
        group_by=q["group_by"],
        n_total=len(df),
        n_selected=len(selected),
        warnings=warnings,
    )


# ----------------------------------------------------------------------------
# 自测:python query_engine.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    demo = pd.DataFrame(
        {
            "age": rng.integers(30, 90, n),
            "sex": rng.choice(["Male", "Female"], n),
            "treatment": rng.choice(["Drug A", "Drug B", "Placebo"], n),
            "stage": rng.choice(["I", "II", "III", "IV"], n),
            "os_time": rng.exponential(24, n).round(1),
            "os_event": rng.integers(0, 2, n),
        }
    )
    demo.loc[:5, "os_time"] = np.nan

    res = run_query(
        demo,
        {
            "filters": [
                {"column": "age", "op": ">", "value": 65},
                {"column": "sex", "op": "==", "value": "female"},  # 大小写自动纠正
            ],
            "outcome": "overall_survival",
            "group_by": "treatment",
            "plot_type": "km_curve",
        },
    )
    print(res.summary())

    for bad in (
        {"filters": [{"column": "agee", "op": ">", "value": 1}]},
        {"filters": [{"column": "age", "op": "like", "value": 1}]},
        {"filters": [{"column": "sex", "op": ">", "value": "Male"}]},
        {"filters": [{"column": "treatment", "op": "==", "value": "Drug Z"}]},
        {"group_by": "age"},
    ):
        try:
            run_query(demo, bad)
        except QueryError as e:
            print("OK 拦截:", e)
