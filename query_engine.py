"""
query_engine.py
Structured query (Parsed Query) -> Pandas filtering / grouping

The input query can be a dict or a Pydantic model (auto model_dump), in the format:
{
  "filters": [{"column": "age", "op": ">", "value": 65}],
  "outcome": "overall_survival",
  "group_by": "treatment",
  "plot_type": "km_curve"
}

Depends only on pandas / numpy. Validation failures raise QueryError so callers can catch it and retry or fall back to rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config: edit here for your real dataset
# ----------------------------------------------------------------------------
ALLOWED_OPS = {">", "<", ">=", "<=", "==", "!=", "in"}
ALLOWED_PLOTS = {"km_curve", "boxplot", "histogram", "bar", "scatter", "trend"}

# outcome name -> (time column, event column); event column 1 = event occurred, 0 = censored
OUTCOME_MAP: dict[str, tuple[str, str]] = {
    "overall_survival": ("os_time", "os_event"),
    "progression_free_survival": ("pfs_time", "pfs_event"),
}

# Max unique values allowed when grouping by a numeric column (above this, binning is required first)
MAX_NUMERIC_GROUPS = 8

# ----------------------------------------------------------------------------
# Survival column auto-detection: when the data has no standard names like os_time/os_event, guess from common naming (don't guess if unsure)
# ----------------------------------------------------------------------------
_DEFAULT_OUTCOME_MAP = dict(OUTCOME_MAP)   # snapshot of the standard mapping; re-derive from it when switching datasets

_TIME_NAMES = ["os_time", "os_months", "time_months", "survival_time", "time_to_event", "duration",
               "time", "months", "followup_months", "follow_up_months"]
_EVENT_NAMES = ["os_event", "event", "status", "death", "dead", "os_status"]
_TIME_TOKENS = {"time", "months", "month", "days", "day", "duration"}
_EVENT_TOKENS = {"event", "status", "death", "dead"}     # deliberately excludes censor*: such columns usually have 1 = censored, the opposite meaning


def _is_binary(s: pd.Series) -> bool:
    vals = set(s.dropna().unique())
    return bool(vals) and vals <= {0, 1}


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def detect_outcome_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    """Return (time column, event column); None if not uniquely determined. The time column must be numeric and the event column must hold only 0/1."""
    lower = {c.lower(): c for c in df.columns}
    t_col = next((lower[n] for n in _TIME_NAMES if n in lower and _is_numeric(df[lower[n]])), None)
    e_col = next((lower[n] for n in _EVENT_NAMES if n in lower and _is_binary(df[lower[n]])), None)

    def by_token(tokens: set[str], ok) -> str | None:
        import re
        found = [c for c in df.columns if tokens & set(re.split(r"[^a-z0-9]+", c.lower())) and ok(df[c])]
        return found[0] if len(found) == 1 else None      # don't guess when there are several candidates

    t_col = t_col or by_token(_TIME_TOKENS, _is_numeric)
    e_col = e_col or by_token(_EVENT_TOKENS, _is_binary)
    return (t_col, e_col) if t_col and e_col and t_col != e_col else None


def configure_outcomes(df: pd.DataFrame, override: tuple[str, str] | None = None) -> dict[str, tuple[str, str]]:
    """
    Set OUTCOME_MAP for the dataset (modified in place, so every module that imported it sees the change immediately).
    Priority: manual override > standard column names (os_*/pfs_*) > auto-detection. If none apply, restore the standard mapping.
    Both manual and auto-detected results are named "overall_survival".
    """
    new = {k: v for k, v in _DEFAULT_OUTCOME_MAP.items() if v[0] in df.columns and v[1] in df.columns}
    if override and override[0] in df.columns and override[1] in df.columns:
        new["overall_survival"] = tuple(override)
    elif "overall_survival" not in new:   # check OS specifically: a dataset with only pfs_* must still auto-detect OS
        det = detect_outcome_columns(df)
        if det:
            new["overall_survival"] = det
    OUTCOME_MAP.clear()
    OUTCOME_MAP.update(new or _DEFAULT_OUTCOME_MAP)
    return dict(OUTCOME_MAP)


class QueryError(ValueError):
    """Invalid query (column missing, op not in allowlist, value type mismatch, etc.)."""


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class QueryResult:
    data: pd.DataFrame                       # filtered data
    groups: dict[str, pd.DataFrame]          # group name -> sub-DataFrame
    outcome: str | None
    time_col: str | None
    event_col: str | None
    plot_type: str
    group_by: str | None
    n_total: int                             # row count before filtering
    n_selected: int                          # row count after filtering
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Brief info for frontend display / logging."""
        return {
            "n_total": self.n_total,
            "n_selected": self.n_selected,
            "groups": {k: len(v) for k, v in self.groups.items()},
            "outcome": self.outcome,
            "plot_type": self.plot_type,
            "warnings": self.warnings,
        }


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def _to_dict(query: Any) -> dict[str, Any]:
    if isinstance(query, dict):
        return query
    if hasattr(query, "model_dump"):      # pydantic v2
        return query.model_dump()
    if hasattr(query, "dict"):            # pydantic v1
        return query.dict()
    raise QueryError(f"Unsupported query type: {type(query).__name__}")


def _match_category(series: pd.Series, value: Any, column: str) -> Any:
    """Categorical variable: match case-insensitively to values that actually exist in the data."""
    lookup = {str(v).lower(): v for v in series.dropna().unique()}
    key = str(value).lower()
    if key not in lookup:
        raise QueryError(
            f"Column '{column}' has no value '{value}'. Available values: {sorted(map(str, lookup.values()))}"
        )
    return lookup[key]


def _coerce_value(df: pd.DataFrame, column: str, op: str, value: Any) -> Any:
    series = df[column]
    is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

    if op == "in":
        if not isinstance(value, (list, tuple, set)):
            raise QueryError(f"Operator 'in' needs a list as value, got: {value!r}")
        values = list(value)
    else:
        if isinstance(value, (list, tuple, set)):
            raise QueryError(f"Operator '{op}' needs a single value, got a list: {value!r}")
        values = [value]

    coerced = []
    for v in values:
        if is_numeric:
            if isinstance(v, bool) or not isinstance(v, (int, float, np.number, str)):
                raise QueryError(f"Numeric column '{column}' must be compared with a number, got: {v!r}")
            try:
                coerced.append(float(v))
            except ValueError:
                raise QueryError(f"Numeric column '{column}' must be compared with a number, got: {v!r}")
        else:
            if op in {">", "<", ">=", "<="}:
                raise QueryError(f"Categorical column '{column}' does not support operator '{op}'")
            coerced.append(_match_category(series, v, column))

    return coerced if op == "in" else coerced[0]


def validate_query(df: pd.DataFrame, query: Any) -> dict[str, Any]:
    """Validate and normalize the query, returning a cleaned dict. Raise QueryError if invalid."""
    q = _to_dict(query)
    cols = set(df.columns)
    clean: dict[str, Any] = {"filters": []}

    # filters
    for i, f in enumerate(q.get("filters") or []):
        column, op, value = f.get("column"), f.get("op"), f.get("value")
        if column not in cols:
            raise QueryError(f"filters[{i}]: column '{column}' does not exist. Available columns: {sorted(cols)}")
        if op not in ALLOWED_OPS:
            raise QueryError(f"filters[{i}]: operator '{op}' is not allowed. Allowed: {sorted(ALLOWED_OPS)}")
        clean["filters"].append(
            {"column": column, "op": op, "value": _coerce_value(df, column, op, value)}
        )

    # outcome
    outcome = q.get("outcome")
    if outcome is not None:
        if outcome not in OUTCOME_MAP:
            raise QueryError(f"Unknown outcome '{outcome}'. Available: {sorted(OUTCOME_MAP)}")
        t_col, e_col = OUTCOME_MAP[outcome]
        missing = [c for c in (t_col, e_col) if c not in cols]
        if missing:
            raise QueryError(f"Outcome '{outcome}' needs columns missing from the data: {missing}")
    clean["outcome"] = outcome

    # group_by
    group_by = q.get("group_by")
    if group_by is not None and group_by not in cols:
        raise QueryError(f"group_by column '{group_by}' does not exist. Available columns: {sorted(cols)}")
    clean["group_by"] = group_by

    # plot_type
    plot_type = q.get("plot_type") or ("km_curve" if outcome else "histogram")
    if plot_type not in ALLOWED_PLOTS:
        raise QueryError(f"Unknown plot_type '{plot_type}'. Available: {sorted(ALLOWED_PLOTS)}")
    if plot_type == "km_curve" and outcome is None:
        raise QueryError("km_curve requires an outcome")
    clean["plot_type"] = plot_type

    return clean


# ----------------------------------------------------------------------------
# Filtering / grouping
# ----------------------------------------------------------------------------
def build_mask(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.Series:
    """Multiple filters are combined with AND. Missing values never satisfy a condition."""
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
            m = s.notna() & (s != v)   # NaN != x must not count as "satisfied"
        elif op == "in":
            m = s.isin(v)
        else:  # in theory validate already rejects this
            raise QueryError(f"Unsupported operator: {op}")
        mask &= m.fillna(False)
    return mask


def split_groups(df: pd.DataFrame, group_by: str | None) -> dict[str, pd.DataFrame]:
    if group_by is None:
        return {"All": df}

    s = df[group_by]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > MAX_NUMERIC_GROUPS:
        raise QueryError(
            f"Numeric column '{group_by}' has {s.nunique()} distinct values; too many to group by directly (bin it first, e.g. into age bands)"
        )
    return {str(k): g for k, g in df.dropna(subset=[group_by]).groupby(group_by, sort=True)}


def run_query(df: pd.DataFrame, query: Any) -> QueryResult:
    """Main entry: validate -> filter -> (outcome column handling) -> group."""
    q = validate_query(df, query)
    warnings: list[str] = []

    selected = df[build_mask(df, q["filters"])]
    if selected.empty:
        raise QueryError("No patients match these filters. Try relaxing them.")

    time_col = event_col = None
    if q["outcome"]:
        time_col, event_col = OUTCOME_MAP[q["outcome"]]
        before = len(selected)
        selected = selected.dropna(subset=[time_col, event_col])
        dropped = before - len(selected)
        if dropped:
            warnings.append(f"{dropped} rows dropped because {time_col}/{event_col} is missing")
        if selected.empty:
            raise QueryError(f"{time_col}/{event_col} is missing for all selected patients")

    groups = split_groups(selected, q["group_by"])

    # Small-group warning (a KM curve with too few samples is of little value)
    for name, g in groups.items():
        if len(g) < 5:
            warnings.append(f"Group '{name}' has only {len(g)} patients; results may be unstable")

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
# Self-test: python query_engine.py
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
                {"column": "sex", "op": "==", "value": "female"},  # case is auto-corrected
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
            print("OK, rejected:", e)
