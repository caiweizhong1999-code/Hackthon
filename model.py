"""
model.py
XGBoost 生存风险模型:训练 / 预测 / SHAP 解释。

模型:XGBoost 的 Cox 比例风险目标 (objective="survival:cox")
  - 标签约定:发生事件 -> +time;删失 -> -time(XGBoost 的要求)
  - predict_risk   -> 相对风险(hazard ratio 尺度,越大越危险)
  - predict_survival -> 个体生存曲线 S(t|x) = S0(t) ** risk,S0 用 Breslow 估计
  - 评估:Harrell C-index(手写,不依赖 lifelines)

SHAP:直接用 XGBoost 自带的 TreeSHAP(pred_contribs=True),不依赖 shap 包。
  值的单位是 log-hazard(margin)空间:正值 = 推高风险,负值 = 降低风险。
  满足可加性:sum(各特征 SHAP) + base_value = 模型 margin。

依赖: xgboost, numpy, pandas, scikit-learn(仅用于分层划分验证集), matplotlib(仅画图时)

典型用法
    from model import SurvivalXGB
    m = SurvivalXGB(outcome="overall_survival").fit(df)
    risk = m.predict_risk(df)
    surv = m.predict_survival(df.head(5), times=[12, 24, 36])
    sr = m.shap_values(df)
    fig = plot_shap_summary(sr)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from query_engine import OUTCOME_MAP

try:
    import xgboost as xgb
except ImportError:   # 允许在没装 xgboost 的环境里导入本模块(例如只用 C-index)
    xgb = None

MIN_TIME = 1e-6                 # Cox 标签用符号区分事件/删失,time 必须 > 0
MAX_CATEGORICAL_UNIQUE = 30     # 取值太多的文本列(如 patient_id)默认不当作特征

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "survival:cox",
    "eval_metric": "cox-nloglik",
    "tree_method": "hist",
    "eta": 0.05,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


def _looks_like_date(s: pd.Series) -> bool:
    """非数值列里 90% 以上能解析成日期,就当作日期列。"""
    vals = s.dropna().astype(str).head(50)
    if len(vals) == 0:
        return False
    try:
        parsed = pd.to_datetime(vals, errors="coerce", format="mixed")
    except (TypeError, ValueError):      # 老版本 pandas 没有 format="mixed"
        return False
    return float(parsed.notna().mean()) >= 0.9


def _require_xgb() -> None:
    if xgb is None:
        raise ImportError("xgboost is required: pip install xgboost")


# ============================================================================
# 评估:Harrell C-index
# ============================================================================
def concordance_index(time: Iterable[float], event: Iterable[int], risk: Iterable[float]) -> float:
    """
    Harrell's C:在所有"可比较"的样本对中,风险更高者确实更早发生事件的比例。
    可比较 = 较早的那个人发生了事件(i 事件且 t_i < t_j)。风险相同记 0.5。
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    r = np.asarray(risk, dtype=float)
    ok = ~(np.isnan(t) | np.isnan(e) | np.isnan(r))
    t, e, r = t[ok], e[ok], r[ok]

    conc = comparable = 0.0
    for i in np.where(e == 1)[0]:
        later = t > t[i]
        n = int(later.sum())
        if n == 0:
            continue
        comparable += n
        conc += float((r[i] > r[later]).sum()) + 0.5 * float((r[i] == r[later]).sum())
    return conc / comparable if comparable else float("nan")


# ============================================================================
# 特征编码:数值列原样;分类列 one-hot(二分类只保留一列);未见过的类别 -> 全 0
# ============================================================================
def _safe_name(name: str) -> str:
    return re.sub(r"[\[\]<>]", "_", name)   # XGBoost 特征名不允许 [ ] <


@dataclass
class Encoder:
    numeric: list[str]
    categorical: dict[str, list[str]]       # 列名 -> 训练时见过的类别(有序)
    columns: list[str]                      # 编码后的列名,顺序固定

    @staticmethod
    def _levels(col: str, cats: list[str]) -> list[str]:
        return cats[1:] if len(cats) == 2 else cats      # 二分类只留第二个水平

    @classmethod
    def fit(cls, df: pd.DataFrame, features: list[str]) -> "Encoder":
        numeric, categorical = [], {}
        for c in features:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                numeric.append(c)
            else:
                categorical[c] = sorted(map(str, s.dropna().unique()))
        cols = list(numeric)
        for c, cats in categorical.items():
            cols += [_safe_name(f"{c}={lv}") for lv in cls._levels(c, cats)]
        return cls(numeric, categorical, cols)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in [*self.numeric, *self.categorical] if c not in df.columns]
        if missing:
            raise ValueError(f"Input data is missing feature columns: {missing}")
        out: dict[str, np.ndarray] = {}
        for c in self.numeric:
            out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        for c, cats in self.categorical.items():
            s = df[c].astype(object)
            isna = s.isna().to_numpy()
            sv = s.astype(str)
            for lv in self._levels(c, cats):
                arr = (sv == lv).to_numpy(float)
                arr[isna] = np.nan             # 缺失值交给 XGBoost 原生处理
                out[_safe_name(f"{c}={lv}")] = arr
        return pd.DataFrame(out, index=df.index)[self.columns]

    def to_dict(self) -> dict[str, Any]:
        return {"numeric": self.numeric, "categorical": self.categorical, "columns": self.columns}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Encoder":
        return cls(d["numeric"], {k: list(v) for k, v in d["categorical"].items()}, d["columns"])


# ============================================================================
# SHAP 结果容器
# ============================================================================
@dataclass
class ShapResult:
    values: pd.DataFrame        # n × p,每个特征的 SHAP(log-hazard 空间)
    base: np.ndarray            # 长度 n 的基准值(各行相同)
    X: pd.DataFrame             # 编码后的特征值,与 values 对齐

    @property
    def base_value(self) -> float:
        return float(np.mean(self.base))

    def margin(self) -> pd.Series:
        """模型 margin = base + sum(SHAP);exp(margin) 即相对风险。"""
        return self.values.sum(axis=1) + self.base

    def mean_abs(self) -> pd.Series:
        """全局重要性:各特征 |SHAP| 的均值(降序)。"""
        return self.values.abs().mean().sort_values(ascending=False)

    def top_contributors(self, row: int, k: int = 5) -> pd.DataFrame:
        """某个病人贡献最大的 k 个特征。row 是行的位置下标(第几行),不是 index 标签。"""
        sv, xv = self.values.iloc[row], self.X.iloc[row]
        df = pd.DataFrame({"feature": sv.index, "value": xv.values, "shap": sv.values})
        return df.reindex(df["shap"].abs().sort_values(ascending=False).index).head(k).reset_index(drop=True)


# ============================================================================
# 模型
# ============================================================================
class SurvivalXGB:
    def __init__(self, outcome: str = "overall_survival", features: list[str] | None = None,
                 params: dict[str, Any] | None = None, seed: int = 42):
        if outcome not in OUTCOME_MAP:
            raise ValueError(f"Unknown outcome '{outcome}'. Available: {sorted(OUTCOME_MAP)}")
        self.outcome = outcome
        self.time_col, self.event_col = OUTCOME_MAP[outcome]
        self.features = features
        self.seed = seed
        self.params = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}

        self.booster = None
        self.encoder: Encoder | None = None
        self.best_iteration_: int | None = None      # 实际使用的树的棵数
        self.baseline_times_: np.ndarray | None = None
        self.baseline_cumhaz_: np.ndarray | None = None
        self.risk_cutpoints_: list[float] | None = None
        self.metrics_: dict[str, float] = {}

    # ------------------------------------------------------------------ 训练
    def _default_features(self, df: pd.DataFrame) -> list[str]:
        # 排除所有 outcome 的时间/事件列(包括别的 outcome,避免信息泄漏)
        banned = {c for pair in OUTCOME_MAP.values() for c in pair}
        feats = []
        for c in df.columns:
            if c in banned:
                continue
            s = df[c]
            numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            if not numeric and s.nunique() > MAX_CATEGORICAL_UNIQUE:
                continue       # 疑似 ID / 自由文本
            if not numeric and _looks_like_date(s):
                continue       # 入组日期等:当成分类特征既无意义,又容易泄漏时间信息
            feats.append(c)
        return feats

    def fit(self, df: pd.DataFrame, num_boost_round: int = 500, early_stopping_rounds: int = 30,
            val_size: float = 0.2, verbose: bool = False) -> "SurvivalXGB":
        _require_xgb()
        data = df.dropna(subset=[self.time_col, self.event_col]).copy()
        e = data[self.event_col].to_numpy(int)
        if len(data) < 30 or e.sum() < 10:
            raise ValueError(f"Too few samples: n={len(data)}, events={int(e.sum())} (need at least 30 patients and 10 events)")

        feats = self.features or self._default_features(data)
        missing = [c for c in feats if c not in data.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        self.features = feats
        self.encoder = Encoder.fit(data, feats)
        X = self.encoder.transform(data)

        t = np.maximum(data[self.time_col].to_numpy(float), MIN_TIME)
        y = np.where(e == 1, t, -t)                      # Cox 标签:事件 +t,删失 -t

        idx = np.arange(len(data))
        if val_size and 0 < val_size < 1:
            from sklearn.model_selection import train_test_split
            tr, va = train_test_split(idx, test_size=val_size, stratify=e, random_state=self.seed)
        else:
            tr, va = idx, np.array([], dtype=int)

        dtrain = xgb.DMatrix(X.iloc[tr], label=y[tr])
        evals, es = [(dtrain, "train")], None
        if len(va):
            dval = xgb.DMatrix(X.iloc[va], label=y[va])
            evals.append((dval, "val"))
            es = early_stopping_rounds

        self.booster = xgb.train(self.params, dtrain, num_boost_round=num_boost_round, evals=evals,
                                 early_stopping_rounds=es, verbose_eval=verbose)

        best = None
        if es is not None:
            try:
                best = int(self.booster.best_iteration) + 1
            except AttributeError:
                best = None
        self.best_iteration_ = best or num_boost_round

        # 指标 + Breslow 基线风险 + 风险分层切点
        risk_all = self._risk_from_X(X)
        self.metrics_ = {
            "n": float(len(data)), "events": float(e.sum()), "n_trees": float(self.best_iteration_),
            "c_index_train": concordance_index(t[tr], e[tr], risk_all[tr]),
        }
        if len(va):
            self.metrics_["c_index_val"] = concordance_index(t[va], e[va], risk_all[va])
        self._fit_baseline(t, e, risk_all)
        self.risk_cutpoints_ = [float(q) for q in np.quantile(risk_all, [1 / 3, 2 / 3])]
        return self

    # --------------------------------------------------------------- 基线风险
    def _fit_baseline(self, t: np.ndarray, e: np.ndarray, risk: np.ndarray) -> None:
        """Breslow 估计累积基线风险 H0(t) = sum_{事件时间 ti<=t} d_i / sum_{j: t_j>=ti} risk_j"""
        times = np.unique(t[e == 1])
        inc = np.empty(len(times))
        for k, ti in enumerate(times):
            d = np.sum((t == ti) & (e == 1))
            inc[k] = d / np.sum(risk[t >= ti])
        self.baseline_times_, self.baseline_cumhaz_ = times, np.cumsum(inc)

    # --------------------------------------------------------------- 预测
    def _check_fitted(self) -> None:
        if self.booster is None or self.encoder is None:
            raise RuntimeError("The model is not trained yet; call fit() or load() first")

    def _iter_range(self) -> tuple[int, int]:
        return (0, int(self.best_iteration_ or 0))

    def _margin_from_X(self, X: pd.DataFrame) -> np.ndarray:
        kw = {"iteration_range": self._iter_range()} if self.best_iteration_ else {}
        return np.asarray(self.booster.predict(xgb.DMatrix(X), output_margin=True, **kw), dtype=float)

    def _risk_from_X(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(np.clip(self._margin_from_X(X), -30, 30))

    def predict_margin(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._margin_from_X(self.encoder.transform(df))

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """相对风险(越大越危险);数值本身只在样本之间比较有意义。"""
        self._check_fitted()
        return self._risk_from_X(self.encoder.transform(df))

    def predict_survival(self, df: pd.DataFrame, times: Iterable[float]) -> pd.DataFrame:
        """个体生存概率 S(t|x) = exp(-H0(t) * risk)。返回 行=病人、列=时间点 的 DataFrame。"""
        self._check_fitted()
        times = np.asarray(list(times), dtype=float)
        risk = self.predict_risk(df)
        pos = np.searchsorted(self.baseline_times_, times, side="right") - 1
        h0 = np.where(pos >= 0, self.baseline_cumhaz_[np.clip(pos, 0, None)], 0.0)
        surv = np.exp(-np.outer(risk, h0))
        return pd.DataFrame(surv, index=df.index, columns=[f"S({t:g})" for t in times])

    def risk_group(self, df: pd.DataFrame, labels: tuple[str, str, str] = ("Low", "Medium", "High")) -> pd.Series:
        """按训练集风险三分位切点分层,可直接作为 group_by 列喂给 KM 曲线。"""
        self._check_fitted()
        risk = self.predict_risk(df)
        g = np.digitize(risk, self.risk_cutpoints_)
        return pd.Series(np.array(labels, dtype=object)[g], index=df.index, name="risk_group")

    # --------------------------------------------------------------- SHAP
    def shap_values(self, df: pd.DataFrame) -> ShapResult:
        """TreeSHAP(XGBoost 内置 pred_contribs),log-hazard 空间。"""
        self._check_fitted()
        X = self.encoder.transform(df)
        kw = {"iteration_range": self._iter_range()} if self.best_iteration_ else {}
        contribs = np.asarray(self.booster.predict(xgb.DMatrix(X), pred_contribs=True, **kw), dtype=float)
        values = pd.DataFrame(contribs[:, :-1], index=X.index, columns=X.columns)
        return ShapResult(values=values, base=contribs[:, -1], X=X)

    def check_additivity(self, df: pd.DataFrame) -> float:
        """返回 |sum(SHAP)+base - margin| 的最大值,应接近 0(~1e-5)。"""
        sr = self.shap_values(df)
        return float(np.max(np.abs(sr.margin().to_numpy() - self.predict_margin(df))))

    # --------------------------------------------------------------- 存取
    def save(self, directory: str | Path) -> None:
        self._check_fitted()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(d / "model.json"))
        meta = {
            "outcome": self.outcome, "features": self.features, "params": self.params, "seed": self.seed,
            "encoder": self.encoder.to_dict(), "best_iteration": self.best_iteration_,
            "baseline_times": self.baseline_times_.tolist(), "baseline_cumhaz": self.baseline_cumhaz_.tolist(),
            "risk_cutpoints": self.risk_cutpoints_, "metrics": self.metrics_,
        }
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "SurvivalXGB":
        _require_xgb()
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        m = cls(outcome=meta["outcome"], features=meta["features"], params=meta["params"], seed=meta["seed"])
        m.params = meta["params"]
        m.booster = xgb.Booster()
        m.booster.load_model(str(d / "model.json"))
        m.encoder = Encoder.from_dict(meta["encoder"])
        m.best_iteration_ = meta["best_iteration"]
        m.baseline_times_ = np.array(meta["baseline_times"])
        m.baseline_cumhaz_ = np.array(meta["baseline_cumhaz"])
        m.risk_cutpoints_ = meta["risk_cutpoints"]
        m.metrics_ = meta["metrics"]
        return m


# ============================================================================
# SHAP 图(matplotlib 手画,不依赖 shap 包)
# ============================================================================
def _scale_for_color(x: np.ndarray) -> np.ndarray:
    """把特征值缩放到 [0,1] 用于着色:5%-95% 分位截断;分位数相同则退回 min-max。"""
    ok = ~np.isnan(x)
    out = np.full(len(x), np.nan)
    if not ok.any():
        return out
    lo, hi = np.percentile(x[ok], [5, 95])
    if hi == lo:
        lo, hi = x[ok].min(), x[ok].max()
    if hi == lo:
        out[ok] = 0.5
    else:
        out[ok] = np.clip((x[ok] - lo) / (hi - lo), 0, 1)
    return out


def plot_shap_summary(sr: ShapResult, max_display: int = 10, seed: int = 0):
    """蜂群图:每个点是一个病人;横轴是 SHAP,颜色是特征值高低(红=高,蓝=低,灰=缺失)。"""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    order = list(sr.mean_abs().index[:max_display])[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(order) + 1.8))
    rng = np.random.default_rng(seed)
    for yi, f in enumerate(order):
        sv = sr.values[f].to_numpy()
        xv = sr.X[f].to_numpy(float)
        c = _scale_for_color(xv)
        jit = yi + rng.uniform(-0.28, 0.28, len(sv))
        nan = np.isnan(c)
        ax.scatter(sv[~nan], jit[~nan], c=c[~nan], cmap="coolwarm", vmin=0, vmax=1, s=11, alpha=0.75, linewidths=0)
        if nan.any():
            ax.scatter(sv[nan], jit[nan], c="#9ca3af", s=11, alpha=0.75, linewidths=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("SHAP value (impact on log hazard)")
    ax.set_title("Feature impact on predicted risk", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="coolwarm"), ax=ax, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"])
    cb.set_label("Feature value")
    return fig


def plot_shap_bar(sr: ShapResult, max_display: int = 10):
    """全局重要性:mean(|SHAP|)。"""
    import matplotlib.pyplot as plt

    imp = sr.mean_abs().iloc[:max_display][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(imp) + 1.6))
    ax.barh(imp.index, imp.values, color="#2563eb")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global feature importance", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    return fig


def plot_shap_local(sr: ShapResult, row: int = 0, max_display: int = 8):
    """单个病人的解释:各特征把风险推高(红)或拉低(蓝)了多少。"""
    import matplotlib.pyplot as plt

    top = sr.top_contributors(row, k=max_display).iloc[::-1]
    labels = [f"{f} = {v:.3g}" if not np.isnan(v) else f"{f} = NA" for f, v in zip(top["feature"], top["value"])]
    colors = ["#ef5b73" if s > 0 else "#3b82f6" for s in top["shap"]]
    margin = float(sr.margin().iloc[row])
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(top) + 1.8))
    ax.barh(labels, top["shap"], color=colors)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("SHAP value (impact on log hazard)")
    ax.set_title(f"Patient explanation — relative risk vs. baseline: {np.exp(margin - sr.base_value):.2f}x",
                 fontweight="bold", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    return fig


# ----------------------------------------------------------------------------
# 自测:python model.py(需要 xgboost)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import tempfile

    import matplotlib
    matplotlib.use("Agg")

    if xgb is None:
        sys.exit("自测需要 xgboost:pip install xgboost")

    rng = np.random.default_rng(0)
    n = 1500
    age = rng.integers(30, 90, n)
    stage = rng.choice(["I", "II", "III", "IV"], n)
    trt = rng.choice(["Drug A", "Drug B", "Placebo"], n)
    sex = rng.choice(["Male", "Female"], n)
    # 真实风险:年龄越大、分期越晚越危险,Drug A 保护
    lin = 0.04 * (age - 60) + 0.5 * np.isin(stage, ["III", "IV"]) - 0.5 * (trt == "Drug A")
    t = rng.exponential(30 / np.exp(lin))
    c = rng.uniform(0, 80, n)
    df = pd.DataFrame({
        "age": age, "sex": sex, "treatment": trt, "stage": stage,
        "os_time": np.minimum(t, c).round(2), "os_event": (t <= c).astype(int),
        "pfs_time": np.minimum(t, c).round(2), "pfs_event": (t <= c).astype(int),
    })
    df.loc[rng.choice(n, 40, replace=False), "age"] = np.nan   # 缺失值

    m = SurvivalXGB("overall_survival").fit(df)
    print("指标:", {k: round(v, 3) for k, v in m.metrics_.items()})
    assert m.metrics_["c_index_val"] > 0.6, "验证集 C-index 偏低,模型没学到信号"

    sr = m.shap_values(df)
    print("全局重要性(mean|SHAP|):\n", sr.mean_abs().round(3).head(6).to_string())
    err = m.check_additivity(df)
    print("SHAP 可加性最大误差:", err)
    assert err < 1e-3

    surv = m.predict_survival(df.head(3), times=[12, 24, 36])
    print(surv.round(3).to_string())
    assert (np.diff(surv.to_numpy(), axis=1) <= 1e-12).all(), "生存概率应随时间单调不增"

    with tempfile.TemporaryDirectory() as d:
        m.save(d)
        m2 = SurvivalXGB.load(d)
        assert np.allclose(m.predict_risk(df), m2.predict_risk(df)), "保存/加载后预测不一致"
    print("风险分层人数:", m.risk_group(df).value_counts().to_dict())

    for name, fig in {"demo_shap_summary": plot_shap_summary(sr), "demo_shap_bar": plot_shap_bar(sr),
                      "demo_shap_local": plot_shap_local(sr, 0)}.items():
        fig.savefig(f"{name}.png", dpi=110, bbox_inches="tight")
    print("全部通过")
