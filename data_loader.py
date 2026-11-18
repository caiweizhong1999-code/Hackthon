"""
data_loader.py
CSV 读取、变量类型识别、数据校验、列映射建议。
不依赖 Streamlit,方便单独测试;app 里只需调用这里的函数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO, Union



import pandas as pd

CATEGORICAL_MAX_UNIQUE = 10  # 唯一值不超过该数,且为非连续变量时视为分类

# 用于自动猜测列映射的关键词(小写匹配)
TIME_HINTS = ["time_months", "time", "os_time", "duration", "followup", "follow_up", "months"]
EVENT_HINTS = ["event", "status", "death", "os_event", "censor"]
ID_HINTS = ["patient_id", "id", "subject", "usubjid"]


# ------------------------------------------------------------
# 数据结构
# ------------------------------------------------------------
@dataclass
class ColumnMapping:
    time_col: str | None = None
    event_col: str | None = None
    id_col: str | None = None
    group_col: str | None = None  # 默认分组变量(如 treatment)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)    # 必须修复,否则不能继续
    warnings: list[str] = field(default_factory=list)  # 可以继续,但需要提醒
    missing_rate: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


# ------------------------------------------------------------
# 1. 读取
# ------------------------------------------------------------
def load_csv(source: Union[str, IO]) -> pd.DataFrame:
    """读取 CSV(路径或 st.file_uploader 返回的文件对象)。"""
    df = pd.read_csv(source, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    # 尝试把像日期的列转成 datetime
    for col in df.columns:
        if (not pd.api.types.is_numeric_dtype(df[col])
                and not pd.api.types.is_datetime64_any_dtype(df[col])
                and any(k in col.lower() for k in ["date", "dt"])):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.9:
                df[col] = parsed
    return df

if __name__ == "__main__":
    df = load_csv(
        r"C:\Users\caiwei\Desktop\hackthon\synthetic_clinical.csv"
    )

    print(df.head())
    print(df.shape)

# ------------------------------------------------------------
# 2. 变量类型识别
# ------------------------------------------------------------
def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """返回 {列名: 'numeric' | 'categorical' | 'datetime' | 'id' | 'text'}。"""
    types: dict[str, str] = {}
    n = len(df)
    for col in df.columns:
        s = df[col]
        nunique = s.nunique(dropna=True)

        if pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            # 数值但取值很少(如 ECOG 0/1/2、event 0/1)按分类处理
            types[col] = "categorical" if nunique <= CATEGORICAL_MAX_UNIQUE else "numeric"
        else:
            if nunique == n and n > CATEGORICAL_MAX_UNIQUE:
                types[col] = "id"
            elif nunique <= CATEGORICAL_MAX_UNIQUE:
                types[col] = "categorical"
            else:
                types[col] = "text"
    return types


# ------------------------------------------------------------
# 3. 列映射建议
# ------------------------------------------------------------
def _find_col(df: pd.DataFrame, hints: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for h in hints:  # 先精确匹配
        if h in lowered:
            return lowered[h]
    for h in hints:  # 再包含匹配
        for low, orig in lowered.items():
            if h in low:
                return orig
    return None


def suggest_mapping(df: pd.DataFrame, types: dict[str, str]) -> ColumnMapping:
    """自动猜测时间/事件/ID/分组列;用户可在侧边栏修改。"""
    time_col = _find_col(df, TIME_HINTS)
    event_col = _find_col(df, EVENT_HINTS)
    id_col = _find_col(df, ID_HINTS)

    group_col = _find_col(df, ["treatment", "arm", "group", "therapy"])
    if group_col is None:
        # 兜底: 第一个 2-5 个水平的分类变量(排除 event)
        for col, t in types.items():
            if t == "categorical" and col != event_col and 2 <= df[col].nunique() <= 5:
                group_col = col
                break

    return ColumnMapping(time_col, event_col, id_col, group_col)


# ------------------------------------------------------------
# 4. 数据校验
# ------------------------------------------------------------
def validate(df: pd.DataFrame, mapping: ColumnMapping,
             types: dict[str, str] | None = None) -> ValidationReport:
    rep = ValidationReport()
    types = types or infer_column_types(df)

    if len(df) == 0:
        rep.errors.append("数据集为空。")
        return rep

    # 缺失率
    miss = df.isna().mean()
    rep.missing_rate = {c: round(float(r), 4) for c, r in miss.items() if r > 0}
    for col, r in rep.missing_rate.items():
        if r > 0.3:
            rep.warnings.append(f"列 `{col}` 缺失率较高 ({r:.1%}),分析时可能被排除。")

    # 重复 ID
    if mapping.id_col and mapping.id_col in df:
        dup = int(df[mapping.id_col].duplicated().sum())
        if dup:
            rep.warnings.append(f"`{mapping.id_col}` 有 {dup} 个重复值。")

    # 生存变量
    if mapping.time_col is None or mapping.time_col not in df:
        rep.errors.append("未指定有效的生存时间列。")
    else:
        t = pd.to_numeric(df[mapping.time_col], errors="coerce")
        if t.isna().any():
            rep.errors.append(f"时间列 `{mapping.time_col}` 含有非数值或缺失值。")
        if (t < 0).any():
            rep.errors.append(f"时间列 `{mapping.time_col}` 存在负值 ({int((t < 0).sum())} 行)。")
        if (t == 0).any():
            rep.warnings.append(f"时间列 `{mapping.time_col}` 有 {int((t == 0).sum())} 行为 0。")

    if mapping.event_col is None or mapping.event_col not in df:
        rep.errors.append("未指定有效的事件列。")
    else:
        vals = set(df[mapping.event_col].dropna().unique())
        if not vals.issubset({0, 1}):
            rep.errors.append(
                f"事件列 `{mapping.event_col}` 必须是 0/1,当前取值: {sorted(map(str, vals))[:6]}"
            )
        elif len(vals) == 1:
            rep.warnings.append(f"事件列 `{mapping.event_col}` 只有一个取值,无法做生存比较。")
        else:
            rate = float(df[mapping.event_col].mean())
            if rate < 0.05:
                rep.warnings.append(f"事件率仅 {rate:.1%},统计效力可能不足。")

    # 常见临床变量的合理范围
    for col in df.columns:
        low = col.lower()
        if types.get(col) == "numeric" and "age" in low:
            bad = df[(df[col] < 0) | (df[col] > 120)]
            if len(bad):
                rep.errors.append(f"`{col}` 有 {len(bad)} 行超出合理范围 (0–120)。")

    # 分组变量的样本量
    if mapping.group_col and mapping.group_col in df:
        counts = df[mapping.group_col].value_counts()
        small = counts[counts < 10]
        if len(small):
            rep.warnings.append(
                f"分组 `{mapping.group_col}` 中样本量过小的组: {', '.join(map(str, small.index))}。"
            )
    return rep


# ------------------------------------------------------------
# 5. 提供给 NL 解析器的数据 schema(LLM prompt 用)
# ------------------------------------------------------------
def build_schema(df: pd.DataFrame, types: dict[str, str], mapping: ColumnMapping,
                 max_levels: int = 10) -> dict:
    """生成 LLM 可读的列描述: 类型、取值范围或数值范围。"""
    cols = {}
    for col in df.columns:
        if types[col] in ("id", "text"):
            continue  # 不暴露 ID 和自由文本
        info: dict = {"type": types[col]}
        if types[col] == "categorical":
            info["values"] = [str(v) if not isinstance(v, (int, float)) else v
                              for v in sorted(df[col].dropna().unique().tolist(), key=str)][:max_levels]
        elif types[col] == "numeric":
            info["min"] = float(df[col].min())
            info["max"] = float(df[col].max())
        cols[col] = info
    return {
        "columns": cols,
        "time_col": mapping.time_col,
        "event_col": mapping.event_col,
        "default_group": mapping.group_col,
    }


# ------------------------------------------------------------
# 简单自测: python data_loader.py data/synthetic_clinical.csv
# ------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys

    path = r"C:\Users\caiwei\Desktop\hackthon\synthetic_clinical.csv"
    data = load_csv(path)
    col_types = infer_column_types(data)
    mp = suggest_mapping(data, col_types)
    report = validate(data, mp, col_types)

    print("列类型:", col_types)
    print("列映射:", mp)
    print("校验通过:", report.ok)
    print("错误:", report.errors)
    print("警告:", report.warnings)
    print("缺失率:", report.missing_rate)
    print(json.dumps(build_schema(data, col_types, mp), ensure_ascii=False, indent=2, default=str))
