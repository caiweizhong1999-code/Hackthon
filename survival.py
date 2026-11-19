"""
survival.py
Kaplan-Meier 估计 + log-rank 检验(纯统计,不依赖 matplotlib / lifelines)。

依赖: numpy, pandas;scipy 仅用于卡方分布的 p 值(没装则 p 值为 nan)

主要接口
    kaplan_meier(time, event)            -> KMCurve        单组 KM(log-log 95% CI、中位生存及其 CI)
    survival_at(curve, t)                -> (S, lo, hi)    某时间点的生存率
    logrank_test(groups)                 -> LogRankResult  两组或多组 log-rank
    pairwise_logrank(groups)             -> DataFrame      两两比较(Bonferroni 校正)
    summarize_groups(groups, t_col, e_col, landmarks=...) -> DataFrame  汇总表
    analyze(result)                      -> SurvivalReport 直接吃 query_engine.QueryResult

约定: event = 1 表示发生事件,0 表示删失;time 必须 >= 0。
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
# 数据清洗
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
    times: np.ndarray          # 事件时间点(含 t=0)
    surv: np.ndarray           # S(t)
    lower: np.ndarray          # 95% CI 下界(log-log 变换,保证落在 [0,1])
    upper: np.ndarray          # 95% CI 上界
    n_at_risk: np.ndarray      # 每个时间点之前的风险人数(t=0 处为总人数)
    n_events: np.ndarray       # 每个时间点发生的事件数
    censor_times: np.ndarray
    censor_surv: np.ndarray    # 删失点处的 S(t),画删失标记用
    n: int
    events: int
    median: float              # 中位生存时间,没降到 0.5 以下则 nan
    median_lower: float        # 中位生存 95% CI;取不到则 nan
    median_upper: float
    max_time: float            # 最大随访时间

    def at_risk(self, t: float, raw_time: np.ndarray | None = None) -> int:
        """t 时刻的风险人数;需要原始 time 数组(KMCurve 本身不保存)。"""
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
    green = [0.0]                      # Greenwood 累积项 sum d/(n(n-d))
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

    # log-log 置信区间:S^exp(±z*se),其中 se = sqrt(G)/|ln S|;S 为 0 或 1 时区间退化为点
    lower, upper = S.copy(), S.copy()
    mask = (S > 0) & (S < 1) & (G > 0)
    se_ll = np.zeros_like(S)
    se_ll[mask] = np.sqrt(G[mask]) / np.abs(np.log(S[mask]))
    lower[mask] = S[mask] ** np.exp(Z95 * se_ll[mask])
    upper[mask] = S[mask] ** np.exp(-Z95 * se_ll[mask])

    # 删失点对应的 S(t)
    cens_t = t[e == 0]
    idx = np.clip(np.searchsorted(times, cens_t, side="right") - 1, 0, len(S) - 1)
    cens_s = S[idx]

    # 中位生存及其 CI:S(t)<=0.5 的第一个时间;CI 用上/下界曲线同样的规则
    median = _first_time_le(times, S)
    med_lo = _first_time_le(times, lower)     # 下界曲线更早降到 0.5 -> 中位生存的下限
    med_hi = _first_time_le(times, upper)     # 上界曲线更晚才降到 0.5 -> 上限;一直没降到则 nan(未达到)

    return KMCurve(
        times=times, surv=S, lower=lower, upper=upper,
        n_at_risk=np.array(n_risk), n_events=np.array(n_ev),
        censor_times=cens_t, censor_surv=cens_s,
        n=len(t), events=int(e.sum()),
        median=median, median_lower=med_lo, median_upper=med_hi,
        max_time=float(t.max()),
    )


def survival_at(curve: KMCurve, t: float) -> tuple[float, float, float]:
    """返回 (S(t), 下界, 上界)。t 超过最大随访时间时返回 (nan, nan, nan)。"""
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
    多组 log-rank 检验(Mantel-Haenszel 形式)。
    groups: {组名: (time, event)}。少于 2 组抛 ValueError。
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

    U = (obs - exp)[:-1]            # 去掉最后一组,协方差阵才可逆
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
    """两两 log-rank。correction: 'bonferroni' 或 'none'。"""
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
# 汇总
# ============================================================================
def _fmt_median(x: float) -> float:
    return x   # 保留 nan 表示未达到(NR),展示层再转成 "NR"


def summarize_groups(groups: dict[str, pd.DataFrame], time_col: str, event_col: str,
                     landmarks: Iterable[float] = ()) -> pd.DataFrame:
    """每组一行:n、事件数、中位生存及 95% CI、各 landmark 时间点的生存率。"""
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
    直接吃 query_engine.QueryResult(需要 groups / time_col / event_col)。
    一组 -> 只有 KM 和汇总;>=2 组 -> 加 log-rank;>=3 组 -> 加两两比较。
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
# 自测:python survival.py
# 用经典的 Freireich 6-MP 白血病数据(Gehan 1965)对照已知结果:
#   R survdiff: log-rank chi2 = 16.8 (1 df), p = 4e-05;中位生存 6-MP=23 周,安慰剂=8 周
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

    assert c6.median == 23 and cp.median == 8, "中位生存与已知结果不符"
    assert c6.events == 9 and cp.events == 21
    # R survfit (log-log CI): 6-MP 中位 23 (13, NA);Placebo 中位 8 (4, 11)
    assert c6.median_lower == 13 and np.isnan(c6.median_upper)
    assert (cp.median_lower, cp.median_upper) == (4, 11)
    assert abs(lr.statistic - 16.8) < 0.2, f"log-rank 统计量与 R 结果不符: {lr.statistic}"
    assert lr.p_value < 0.001
    s, lo, hi = survival_at(c6, 20)
    print(f"S(20) 6-MP = {s:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    assert 0 <= lo <= s <= hi <= 1

    # 三组 + 两两比较的冒烟测试
    rng = np.random.default_rng(0)
    gs = {}
    for name, scale in {"A": 36, "B": 28, "C": 18}.items():
        tt = rng.exponential(scale, 80)
        cc = rng.uniform(0, 60, 80)
        gs[name] = (np.minimum(tt, cc), (tt <= cc).astype(int))
    print(logrank_test(gs))
    print(pairwise_logrank(gs).round(4).to_string(index=False))

    # 输入校验
    for bad in [([1, 2], [1, 2]), ([-1, 2], [1, 0]), ([1, 2], [1])]:
        try:
            kaplan_meier(*bad)
        except ValueError as ex:
            print("OK 拦截:", ex)
    print("\n全部通过")
