"""
table1.py
Baseline characteristics (Table 1) and a publication-style survival summary table.
Pure numpy / pandas / scipy; no Streamlit, so it is easy to test on its own.

    baseline_table(df, group_col, types=..., exclude=...)   -> DataFrame   Table 1
    survival_table(groups, time_col, event_col, ...)        -> DataFrame   survival summary
    logrank_hr(t_a, e_a, t_b, e_b)                          -> (HR, lo, hi)

Self-test: python table1.py
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:          # p-values are left blank without scipy
    stats = None

from data_loader import infer_column_types
from survival import kaplan_meier, logrank_test, survival_at


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------
def _p(p: float | None) -> str:
    if p is None or np.isnan(p):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _num(v: float | None, d: int = 1) -> str:
    return "NR" if v is None or np.isnan(v) else f"{v:.{d}f}"


def _ci(x: float, lo: float, hi: float, d: int = 1) -> str:
    return f"{_num(x, d)} ({_num(lo, d)}–{_num(hi, d)})"


def _pct_ci(x: float, lo: float, hi: float) -> str:
    if np.isnan(x):
        return "NE"          # not estimable: landmark is beyond the longest follow-up
    return f"{x:.1%} ({lo:.1%}–{hi:.1%})"


def _singular(unit: str) -> str:
    return unit[:-1] if unit.endswith("s") else unit


# ----------------------------------------------------------------------------
# Table 1
# ----------------------------------------------------------------------------
def baseline_table(df: pd.DataFrame, group_col: str | None = None,
                   types: dict[str, str] | None = None,
                   exclude: Iterable[str] = (), max_vars: int = 30) -> pd.DataFrame:
    """
    Rows = variables, columns = Overall + one column per group + p-value.
      numeric:     median [IQR] and mean ± SD   (Mann-Whitney U / Kruskal-Wallis)
      categorical: n (%)                        (Fisher exact for 2x2, otherwise chi-square)
    A 'Missing' row is added for variables that have missing values.
    `types` should come from the FULL dataset (infer_column_types(df)) so that a filtered cohort
    does not change how a column is classified.
    """
    types = types or infer_column_types(df)
    skip = set(exclude) | ({group_col} if group_col else set())
    cols = [c for c in df.columns if c not in skip and types.get(c) in ("numeric", "categorical")][:max_vars]

    if group_col and group_col in df.columns:
        d = df.dropna(subset=[group_col])
        levels = sorted(d[group_col].unique().tolist(), key=str)
        parts = {str(lv): d[d[group_col] == lv] for lv in levels}
    else:
        d, parts = df, {}
    heads = [f"Overall (N={len(d)})"] + [f"{k} (N={len(v)})" for k, v in parts.items()]
    rows: list[list[str]] = []

    for c in cols:
        series = [d[c]] + [v[c] for v in parts.values()]
        pval = np.nan
        if types[c] == "numeric":
            if parts and stats is not None:
                samples = [s.dropna() for s in series[1:] if s.notna().any()]
                try:
                    if len(samples) == 2:
                        pval = stats.mannwhitneyu(*samples).pvalue
                    elif len(samples) > 2:
                        pval = stats.kruskal(*samples).pvalue
                except ValueError:       # e.g. all values identical
                    pass
            rows.append([f"{c}, median [IQR]",
                         *[f"{s.median():.1f} [{s.quantile(.25):.1f}–{s.quantile(.75):.1f}]" if s.notna().any() else ""
                           for s in series], _p(pval)])
            rows.append([f"{c}, mean ± SD",
                         *[f"{s.mean():.1f} ± {s.std():.1f}" if s.notna().sum() > 1 else "" for s in series], ""])
        else:
            levels_c = sorted(d[c].dropna().unique().tolist(), key=str)
            if parts and stats is not None:
                tab = pd.crosstab(d[c], d[group_col])
                if tab.shape[0] > 1 and tab.shape[1] > 1:
                    try:
                        pval = (stats.fisher_exact(tab.values)[1] if tab.shape == (2, 2)
                                else stats.chi2_contingency(tab.values)[1])
                    except ValueError:
                        pass
            rows.append([f"{c}, n (%)", *[""] * len(series), _p(pval)])
            for lv in levels_c:
                rows.append([f"    {lv}", *[
                    f"{int((s == lv).sum())} ({(s == lv).sum() / max(int(s.notna().sum()), 1):.1%})" for s in series
                ], ""])
        miss = [int(s.isna().sum()) for s in series]
        if any(miss):
            rows.append(["    Missing", *[str(m) for m in miss], ""])

    return pd.DataFrame(rows, columns=["Characteristic", *heads, "p-value"])


# ----------------------------------------------------------------------------
# Hazard ratio from the log-rank statistic
# ----------------------------------------------------------------------------
def logrank_hr(t_a, e_a, t_b, e_b) -> tuple[float, float, float]:
    """
    HR of group a vs group b from the log-rank (O-E)/V estimate, with a Wald 95% CI.
    Close to, but not identical with, a Cox model HR (exact when the true HR is near 1).
    """
    t = np.concatenate([t_a, t_b]).astype(float)
    e = np.concatenate([e_a, e_b]).astype(int)
    g = np.concatenate([np.ones(len(t_a)), np.zeros(len(t_b))])
    O = E = V = 0.0
    for tt in np.unique(t[e == 1]):
        risk = t >= tt
        n = risk.sum()
        na = (risk & (g == 1)).sum()
        at = (t == tt) & (e == 1)
        d, da = at.sum(), (at & (g == 1)).sum()
        O += da
        E += d * na / n
        if n > 1:
            V += d * (n - d) / (n - 1) * (na / n) * (1 - na / n)
    if V <= 0:
        return float("nan"), float("nan"), float("nan")
    ln, se = (O - E) / V, 1 / np.sqrt(V)
    return float(np.exp(ln)), float(np.exp(ln - 1.96 * se)), float(np.exp(ln + 1.96 * se))


# ----------------------------------------------------------------------------
# Survival summary table
# ----------------------------------------------------------------------------
def survival_table(groups: dict[str, pd.DataFrame], time_col: str, event_col: str,
                   landmarks: Iterable[float] = (12, 24), time_unit: str = "months",
                   ref: str | None = None, include_overall: bool = True) -> pd.DataFrame:
    """
    One row per group (+ an 'Overall' row when there are several groups):
      N, events, event rate, median follow-up (reverse Kaplan-Meier),
      median survival (95% CI), survival at each landmark (95% CI),
      HR vs the reference group (95% CI), and the overall log-rank p.
    """
    names = list(groups)
    ref = ref if ref in groups else names[0]
    items = dict(groups)
    if include_overall and len(groups) > 1:
        items = {"Overall": pd.concat(groups.values()), **groups}

    lr_p = ""
    if len(groups) >= 2:
        raw = {n: (g[time_col].to_numpy(), g[event_col].to_numpy()) for n, g in groups.items()}
        lr_p = _p(logrank_test(raw).p_value)

    unit1 = _singular(time_unit)
    rows = []
    for name, g in items.items():
        t, e = g[time_col].to_numpy(), g[event_col].to_numpy()
        c = kaplan_meier(t, e)
        fu = kaplan_meier(t, 1 - e).median       # reverse KM = median follow-up
        row = {
            "Group": name, "N": c.n, "Events": c.events,
            "Event rate": f"{c.events / c.n:.1%}",
            f"Median follow-up ({time_unit})": _num(fu),
            f"Median survival ({time_unit}, 95% CI)": _ci(c.median, c.median_lower, c.median_upper),
        }
        for lm in landmarks:
            row[f"{lm:g}-{unit1} survival (95% CI)"] = _pct_ci(*survival_at(c, lm))
        if len(groups) >= 2 and name in groups:
            if name == ref:
                row["HR vs ref (95% CI)"] = "Reference"
            else:
                r = groups[ref]
                row["HR vs ref (95% CI)"] = _ci(*logrank_hr(t, e, r[time_col].to_numpy(), r[event_col].to_numpy()), d=2)
        rows.append(row)

    out = pd.DataFrame(rows).fillna("")     # the Overall row has no HR
    if lr_p:
        out["Log-rank p"] = [lr_p] + [""] * (len(out) - 1)
    return out


# ----------------------------------------------------------------------------
# Self-test: python table1.py
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    path = Path(__file__).resolve().parent / "synthetic_clinical.csv"
    data = pd.read_csv(path)
    ty = infer_column_types(data)
    skip = ("os_time", "os_event", "pfs_time", "pfs_event")

    t1 = baseline_table(data, "treatment", types=ty, exclude=skip)
    print(t1.to_string(index=False))
    assert t1.shape[1] == 6 and "Overall (N=1248)" in t1.columns
    assert baseline_table(data, None, types=ty, exclude=skip).shape[1] == 3   # Characteristic, Overall, p-value

    for t_col, e_col in (("os_time", "os_event"), ("pfs_time", "pfs_event")):
        groups = {k: g for k, g in data.groupby("treatment")}
        st_ = survival_table(groups, t_col, e_col, landmarks=(12, 24))
        print("\n", t_col)
        print(st_.to_string(index=False))
        assert st_.iloc[0]["Group"] == "Overall" and st_.iloc[1]["HR vs ref (95% CI)"] == "Reference"

    # HR sanity: a true 2x hazard difference should give HR ~ 2
    rng = np.random.default_rng(0)
    ta, tb = rng.exponential(10, 2000), rng.exponential(20, 2000)
    hr, lo, hi = logrank_hr(ta, np.ones(2000), tb, np.ones(2000))
    print(f"\nHR (true 2.0) = {hr:.2f} ({lo:.2f}-{hi:.2f})")
    assert lo < 2.0 < hi
    print("\nAll tests passed")
