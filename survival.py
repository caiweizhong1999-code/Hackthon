"""
survival.py
Kaplan-Meier estimation + log-rank test (pure statistics, no matplotlib / lifelines).

Dependencies: numpy, pandas; scipy is used only for the chi-square p-value (p is nan if not installed)

Main interface
    kaplan_meier(time, event)            -> KMCurve        single-group KM (log-log 95% CI, median survival and its CI)
    survival_at(curve, t)                -> (S, lo, hi)    survival probability at a given time
    logrank_test(groups)                 -> LogRankResult  two-group or multi-group log-rank
    pairwise_logrank(groups)             -> DataFrame      pairwise comparisons (Bonferroni correction)
    summarize_groups(groups, t_col, e_col, landmarks=...) -> DataFrame  summary table
    analyze(result)                      -> SurvivalReport takes a query_engine.QueryResult directly

Convention: event = 1 means the event occurred, 0 means censored; time must be >= 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2
except ImportError:
    chi2 = None

Z95 = 1.959963984540054


# ============================================================================
# Data cleaning
# ============================================================================
def _clean(time: Iterable[float], event: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(list(time) if not hasattr(time, "__array__") else time, dtype=float)
    e = np.asarray(list(event) if not hasattr(event, "__array__") else event, dtype=float)
    if t.shape != e.shape:
        raise ValueError("time and event must have the same length")
    ok = ~(np.isnan(t) | np.isnan(e))
    t, e = t[ok], e[ok]
    if len(t) == 0:
        raise ValueError("No usable samples (time/event all missing)")
    if (t < 0).any():
        raise ValueError("time cannot be negative")
    if not np.isin(e, (0, 1)).all():
        raise ValueError("event must be 0/1 (1 = event occurred, 0 = censored)")
    return t, e.astype(int)


# ============================================================================
# Kaplan-Meier
# ============================================================================
@dataclass
class KMCurve:
    times: np.ndarray          # event time points (including t=0)
    surv: np.ndarray           # S(t)
    lower: np.ndarray          # 95% CI lower bound (log-log transform, guaranteed to lie in [0,1])
    upper: np.ndarray          # 95% CI upper bound
    n_at_risk: np.ndarray      # number at risk just before each time point (total n at t=0)
    n_events: np.ndarray       # number of events at each time point
    censor_times: np.ndarray
    censor_surv: np.ndarray    # S(t) at the censoring points, used to draw censor marks
    n: int
    events: int
    median: float              # median survival time; nan if S(t) never drops to 0.5 or below
    median_lower: float        # median survival 95% CI; nan if unavailable
    median_upper: float
    max_time: float            # maximum follow-up time

    def at_risk(self, t: float, raw_time: np.ndarray | None = None) -> int:
        """Number at risk at time t; needs the raw time array (KMCurve does not store it)."""
        if raw_time is None:
            raise ValueError("The raw time array is required")
        return int(np.sum(np.asarray(raw_time) >= t))


def _first_time_le(times: np.ndarray, values: np.ndarray, thr: float = 0.5) -> float:
    idx = np.where(values <= thr)[0]
    return float(times[idx[0]]) if len(idx) else float("nan")


def kaplan_meier(time: Iterable[float], event: Iterable[int]) -> KMCurve:
    t, e = _clean(time, event)
    order = np.argsort(t, kind="stable")
    t, e = t[order], e[order]

    ev_times = np.unique(t[e == 1])
    surv = [1.0]
    green = [0.0]                      # Greenwood cumulative term sum d/(n(n-d))
    n_risk = [len(t)]
    n_ev = [0]
    s, g = 1.0, 0.0
    for tt in ev_times:
        n_i = int(np.sum(t >= tt))
        d_i = int(np.sum((t == tt) & (e == 1)))
        s *= 1 - d_i / n_i
        if n_i > d_i:
            g += d_i / (n_i * (n_i - d_i))
        surv.append(s)
        green.append(g)
        n_risk.append(n_i)
        n_ev.append(d_i)

    times = np.concatenate([[0.0], ev_times])
    S = np.array(surv)
    G = np.array(green)

    # log-log confidence interval: S^exp(±z*se), where se = sqrt(G)/|ln S|; the interval degenerates to a point when S is 0 or 1
    lower, upper = S.copy(), S.copy()
    mask = (S > 0) & (S < 1) & (G > 0)
    se_ll = np.zeros_like(S)
    se_ll[mask] = np.sqrt(G[mask]) / np.abs(np.log(S[mask]))
    lower[mask] = S[mask] ** np.exp(Z95 * se_ll[mask])
    upper[mask] = S[mask] ** np.exp(-Z95 * se_ll[mask])

    # S(t) at the censoring points
    cens_t = t[e == 0]
    idx = np.clip(np.searchsorted(times, cens_t, side="right") - 1, 0, len(S) - 1)
    cens_s = S[idx]

    # Median survival and its CI: the first time with S(t)<=0.5; the CI uses the same rule on the upper/lower bound curves
    median = _first_time_le(times, S)
    med_lo = _first_time_le(times, lower)     # the lower bound curve drops to 0.5 earlier -> lower limit of the median
    med_hi = _first_time_le(times, upper)     # the upper bound curve drops to 0.5 later -> upper limit; nan (not reached) if it never drops

    return KMCurve(
        times=times, surv=S, lower=lower, upper=upper,
        n_at_risk=np.array(n_risk), n_events=np.array(n_ev),
        censor_times=cens_t, censor_surv=cens_s,
        n=len(t), events=int(e.sum()),
        median=median, median_lower=med_lo, median_upper=med_hi,
        max_time=float(t.max()),
    )


def survival_at(curve: KMCurve, t: float) -> tuple[float, float, float]:
    """Return (S(t), lower, upper). Returns (nan, nan, nan) when t exceeds the maximum follow-up time."""
    if t > curve.max_time:
        return float("nan"), float("nan"), float("nan")
    i = int(np.searchsorted(curve.times, t, side="right") - 1)
    i = max(i, 0)
    return float(curve.surv[i]), float(curve.lower[i]), float(curve.upper[i])


# ============================================================================
# log-rank
# ============================================================================
@dataclass
class LogRankResult:
    statistic: float
    df: int
    p_value: float
    observed: dict[str, int]
    expected: dict[str, float]

    def __str__(self) -> str:
        p = "p < 0.001" if self.p_value < 0.001 else f"p = {self.p_value:.3f}"
        return f"log-rank chi2({self.df}) = {self.statistic:.2f}, {p}"


def logrank_test(groups: dict[str, tuple[Iterable[float], Iterable[int]]]) -> LogRankResult:
    """
    Multi-group log-rank test (Mantel-Haenszel form).
    groups: {group name: (time, event)}. Raises ValueError if fewer than 2 groups.
    """
    if len(groups) < 2:
        raise ValueError("Log-rank needs at least 2 groups")
    names = list(groups)
    k = len(names)
    cleaned = [_clean(*groups[g]) for g in names]
    all_t = np.concatenate([c[0] for c in cleaned])
    all_e = np.concatenate([c[1] for c in cleaned])
    all_g = np.concatenate([np.full(len(c[0]), i) for i, c in enumerate(cleaned)])

    obs = np.zeros(k)
    exp = np.zeros(k)
    V = np.zeros((k, k))
    for tt in np.unique(all_t[all_e == 1]):
        at_risk = all_t >= tt
        n = at_risk.sum()
        at_t = all_t == tt
        d = (at_t & (all_e == 1)).sum()
        n_k = np.array([(at_risk & (all_g == i)).sum() for i in range(k)])
        d_k = np.array([(at_t & (all_e == 1) & (all_g == i)).sum() for i in range(k)])
        obs += d_k
        exp += d * n_k / n
        if n > 1:
            f = d * (n - d) / (n - 1) / n**2
            V += f * (n * np.diag(n_k) - np.outer(n_k, n_k))

    U = (obs - exp)[:-1]            # drop the last group so the covariance matrix is invertible
    Vr = V[:-1, :-1]
    try:
        stat = float(U @ np.linalg.solve(Vr, U))
    except np.linalg.LinAlgError:
        stat = float("nan")
    p = float(chi2.sf(stat, k - 1)) if (chi2 is not None and not np.isnan(stat)) else float("nan")
    return LogRankResult(
        statistic=stat, df=k - 1, p_value=p,
        observed={n: int(o) for n, o in zip(names, obs)},
        expected={n: float(x) for n, x in zip(names, exp)},
    )


def pairwise_logrank(groups: dict[str, tuple[Iterable[float], Iterable[int]]],
                     correction: str = "bonferroni") -> pd.DataFrame:
    """Pairwise log-rank. correction: 'bonferroni' or 'none'."""
    pairs = list(combinations(groups, 2))
    rows = []
    for a, b in pairs:
        r = logrank_test({a: groups[a], b: groups[b]})
        rows.append({"group_a": a, "group_b": b, "chi2": r.statistic, "p_value": r.p_value})
    df = pd.DataFrame(rows)
    if correction == "bonferroni" and len(df):
        df["p_adjusted"] = (df["p_value"] * len(pairs)).clip(upper=1.0)
    return df


# ============================================================================
# Summary
# ============================================================================
def _fmt_median(x: float) -> float:
    return x   # keep nan to mean not reached (NR); the display layer converts it to "NR"


def summarize_groups(groups: dict[str, pd.DataFrame], time_col: str, event_col: str,
                     landmarks: Iterable[float] = ()) -> pd.DataFrame:
    """One row per group: n, events, median survival and 95% CI, survival rate at each landmark time."""
    rows = []
    for name, g in groups.items():
        c = kaplan_meier(g[time_col], g[event_col])
        row: dict[str, Any] = {
            "group": name, "n": c.n, "events": c.events,
            "median": _fmt_median(c.median),
            "median_ci_lower": c.median_lower, "median_ci_upper": c.median_upper,
        }
        for lm in landmarks:
            s, lo, hi = survival_at(c, lm)
            row[f"S({lm:g})"] = s
            row[f"S({lm:g})_lo"] = lo
            row[f"S({lm:g})_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class SurvivalReport:
    curves: dict[str, KMCurve]
    summary: pd.DataFrame
    logrank: LogRankResult | None = None
    pairwise: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)


def analyze(result: Any, landmarks: Iterable[float] = ()) -> SurvivalReport:
    """
    Takes a query_engine.QueryResult directly (needs groups / time_col / event_col).
    One group -> KM and summary only; >=2 groups -> add log-rank; >=3 groups -> add pairwise comparisons.
    """
    if not getattr(result, "time_col", None):
        raise ValueError("This query has no outcome, so survival analysis is not possible")
    t_col, e_col = result.time_col, result.event_col
    groups = result.groups

    curves = {n: kaplan_meier(g[t_col], g[e_col]) for n, g in groups.items()}
    raw = {n: (g[t_col].to_numpy(), g[e_col].to_numpy()) for n, g in groups.items()}
    warnings: list[str] = []

    lr = pw = None
    if len(groups) >= 2:
        lr = logrank_test(raw)
        if np.isnan(lr.p_value):
            warnings.append("Log-rank p-value could not be computed (scipy missing or singular covariance matrix)")
    if len(groups) >= 3:
        pw = pairwise_logrank(raw)

    for n, c in curves.items():
        if c.events == 0:
            warnings.append(f"Group '{n}' has no events; its KM curve is a flat line")

    return SurvivalReport(curves=curves, summary=summarize_groups(groups, t_col, e_col, landmarks),
                          logrank=lr, pairwise=pw, warnings=warnings)


# ----------------------------------------------------------------------------
# Self-test: python survival.py
# Check against known results using the classic Freireich 6-MP leukemia data (Gehan 1965):
#   R survdiff: log-rank chi2 = 16.8 (1 df), p = 4e-05;median survival 6-MP = 23 weeks, placebo = 8 weeks
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    mp6_t = [6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32, 34, 35]
    mp6_e = [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    pl_t = [1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23]
    pl_e = [1] * 21

    c6 = kaplan_meier(mp6_t, mp6_e)
    cp = kaplan_meier(pl_t, pl_e)
    print(f"6-MP   n={c6.n} events={c6.events} median={c6.median} "
          f"CI=({c6.median_lower}, {c6.median_upper})")
    print(f"Placebo n={cp.n} events={cp.events} median={cp.median} "
          f"CI=({cp.median_lower}, {cp.median_upper})")

    lr = logrank_test({"6-MP": (mp6_t, mp6_e), "Placebo": (pl_t, pl_e)})
    print(lr, "| observed:", lr.observed, "| expected:", {k: round(v, 2) for k, v in lr.expected.items()})

    assert c6.median == 23 and cp.median == 8, "median survival does not match the known result"
    assert c6.events == 9 and cp.events == 21
    # R survfit (log-log CI): 6-MP median 23 (13, NA); Placebo median 8 (4, 11)
    assert c6.median_lower == 13 and np.isnan(c6.median_upper)
    assert (cp.median_lower, cp.median_upper) == (4, 11)
    assert abs(lr.statistic - 16.8) < 0.2, f"log-rank statistic does not match R: {lr.statistic}"
    assert lr.p_value < 0.001
    s, lo, hi = survival_at(c6, 20)
    print(f"S(20) 6-MP = {s:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    assert 0 <= lo <= s <= hi <= 1

    # Smoke test for three groups + pairwise comparisons
    rng = np.random.default_rng(0)
    gs = {}
    for name, scale in {"A": 36, "B": 28, "C": 18}.items():
        tt = rng.exponential(scale, 80)
        cc = rng.uniform(0, 60, 80)
        gs[name] = (np.minimum(tt, cc), (tt <= cc).astype(int))
    print(logrank_test(gs))
    print(pairwise_logrank(gs).round(4).to_string(index=False))

    # Input validation
    for bad in [([1, 2], [1, 2]), ([-1, 2], [1, 0]), ([1, 2], [1])]:
        try:
            kaplan_meier(*bad)
        except ValueError as ex:
            print("OK, rejected:", ex)
    print("\nAll tests passed")
