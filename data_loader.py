"""
data_loader.py
CSV loading, column type inference, data validation, and column mapping suggestions.
Does not depend on Streamlit, so it is easy to test on its own; the app only needs to call these functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO, Union



import pandas as pd

CATEGORICAL_MAX_UNIQUE = 10  # Columns with at most this many unique values (and not continuous) are treated as categorical

# Keywords for auto-guessing the column mapping (lowercase matching)
TIME_HINTS = ["time_months", "time", "os_time", "duration", "followup", "follow_up", "months"]
EVENT_HINTS = ["event", "status", "death", "os_event", "censor"]
ID_HINTS = ["patient_id", "id", "subject", "usubjid"]


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------
@dataclass
class ColumnMapping:
    time_col: str | None = None
    event_col: str | None = None
    id_col: str | None = None
    group_col: str | None = None  # Default grouping variable (e.g. treatment)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)    # Must be fixed before continuing
    warnings: list[str] = field(default_factory=list)  # Can continue, but the user should be warned
    missing_rate: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


# ------------------------------------------------------------
# 1. Loading
# ------------------------------------------------------------
def load_csv(source: Union[str, IO]) -> pd.DataFrame:
    """Read a CSV (a path or a file object returned by st.file_uploader)."""
    df = pd.read_csv(source, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    # Try to convert date-like columns to datetime
    for col in df.columns:
        if (not pd.api.types.is_numeric_dtype(df[col])
                and not pd.api.types.is_datetime64_any_dtype(df[col])
                and any(k in col.lower() for k in ["date", "dt"])):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.9:
                df[col] = parsed
    return df

# ------------------------------------------------------------
# 2. Column type inference
# ------------------------------------------------------------
def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Return {column name: 'numeric' | 'categorical' | 'datetime' | 'id' | 'text'}."""
    types: dict[str, str] = {}
    n = len(df)
    for col in df.columns:
        s = df[col]
        nunique = s.nunique(dropna=True)

        if pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            # Numeric but with very few distinct values (e.g. ECOG 0/1/2, event 0/1): treat as categorical
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
# 3. Column mapping suggestions
# ------------------------------------------------------------
def _find_col(df: pd.DataFrame, hints: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for h in hints:  # exact match first
        if h in lowered:
            return lowered[h]
    for h in hints:  # then substring match
        for low, orig in lowered.items():
            if h in low:
                return orig
    return None


def suggest_mapping(df: pd.DataFrame, types: dict[str, str]) -> ColumnMapping:
    """Auto-guess the time/event/ID/group columns; the user can change them in the sidebar."""
    time_col = _find_col(df, TIME_HINTS)
    event_col = _find_col(df, EVENT_HINTS)
    id_col = _find_col(df, ID_HINTS)

    group_col = _find_col(df, ["treatment", "arm", "group", "therapy"])
    if group_col is None:
        # Fallback: the first categorical variable with 2-5 levels (excluding event)
        for col, t in types.items():
            if t == "categorical" and col != event_col and 2 <= df[col].nunique() <= 5:
                group_col = col
                break

    return ColumnMapping(time_col, event_col, id_col, group_col)


# ------------------------------------------------------------
# 4. Data validation
# ------------------------------------------------------------
def validate(df: pd.DataFrame, mapping: ColumnMapping,
             types: dict[str, str] | None = None) -> ValidationReport:
    rep = ValidationReport()
    types = types or infer_column_types(df)

    if len(df) == 0:
        rep.errors.append("The dataset is empty.")
        return rep

    # Missing rate
    miss = df.isna().mean()
    rep.missing_rate = {c: round(float(r), 4) for c, r in miss.items() if r > 0}
    for col, r in rep.missing_rate.items():
        if r > 0.3:
            rep.warnings.append(f"Column `{col}` has a high missing rate ({r:.1%}); it may be excluded from analysis.")

    # Duplicate IDs
    if mapping.id_col and mapping.id_col in df:
        dup = int(df[mapping.id_col].duplicated().sum())
        if dup:
            rep.warnings.append(f"`{mapping.id_col}` has {dup} duplicate values.")

    # Survival variables
    if mapping.time_col is None or mapping.time_col not in df:
        rep.errors.append("No valid survival time column specified.")
    else:
        t = pd.to_numeric(df[mapping.time_col], errors="coerce")
        if t.isna().any():
            rep.errors.append(f"Time column `{mapping.time_col}` contains non-numeric or missing values.")
        if (t < 0).any():
            rep.errors.append(f"Time column `{mapping.time_col}` has negative values ({int((t < 0).sum())} rows).")
        if (t == 0).any():
            rep.warnings.append(f"Time column `{mapping.time_col}` has {int((t == 0).sum())} rows equal to 0.")

    if mapping.event_col is None or mapping.event_col not in df:
        rep.errors.append("No valid event column specified.")
    else:
        vals = set(df[mapping.event_col].dropna().unique())
        if not vals.issubset({0, 1}):
            rep.errors.append(
                f"Event column `{mapping.event_col}` must be 0/1; current values: {sorted(map(str, vals))[:6]}"
            )
        elif len(vals) == 1:
            rep.warnings.append(f"Event column `{mapping.event_col}` has only one value; survival comparison is not possible.")
        else:
            rate = float(df[mapping.event_col].mean())
            if rate < 0.05:
                rep.warnings.append(f"Event rate is only {rate:.1%}; statistical power may be insufficient.")

    # Plausible ranges for common clinical variables
    for col in df.columns:
        low = col.lower()
        if types.get(col) == "numeric" and "age" in low:
            bad = df[(df[col] < 0) | (df[col] > 120)]
            if len(bad):
                rep.errors.append(f"`{col}` has {len(bad)} rows outside the plausible range (0-120).")

    # Sample size of the grouping variable
    if mapping.group_col and mapping.group_col in df:
        counts = df[mapping.group_col].value_counts()
        small = counts[counts < 10]
        if len(small):
            rep.warnings.append(
                f"Groups in `{mapping.group_col}` with very small sample size: {', '.join(map(str, small.index))}."
            )
    return rep


# ------------------------------------------------------------
# 5. Data schema for the NL parser (used in the LLM prompt)
# ------------------------------------------------------------
def build_schema(df: pd.DataFrame, types: dict[str, str], mapping: ColumnMapping,
                 max_levels: int = 10) -> dict:
    """Build an LLM-readable column description: type, allowed values or numeric range."""
    cols = {}
    for col in df.columns:
        if types[col] in ("id", "text"):
            continue  # Do not expose IDs or free text
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
# Simple self-test: python data_loader.py data/synthetic_clinical.csv
# ------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "synthetic_clinical.csv"
    data = load_csv(path)
    print(data.head())
    print(data.shape)
    col_types = infer_column_types(data)
    mp = suggest_mapping(data, col_types)
    report = validate(data, mp, col_types)

    print("Column types:", col_types)
    print("Column mapping:", mp)
    print("Validation passed:", report.ok)
    print("Errors:", report.errors)
    print("Warnings:", report.warnings)
    print("Missing rates:", report.missing_rate)
    print(json.dumps(build_schema(data, col_types, mp), ensure_ascii=False, indent=2, default=str))
