"""
model.py
XGBoost survival risk model: training / prediction / SHAP explanation.

Model: XGBoost Cox proportional hazards objective (objective="survival:cox")
  - Label convention: event -> +time; censored -> -time (an XGBoost requirement)
  - predict_risk   -> relative risk (hazard ratio scale; larger = higher risk)
  - predict_survival -> individual survival curve S(t|x) = S0(t) ** risk, with S0 estimated by Breslow
  - Evaluation: Harrell C-index (hand-written, no lifelines)

SHAP: uses XGBoost's built-in TreeSHAP (pred_contribs=True), no shap package needed.
  Values are in log-hazard (margin) space: positive = raises risk, negative = lowers risk.
  Additive: sum(SHAP over features) + base_value = model margin.

Dependencies: xgboost, numpy, pandas, scikit-learn (only for the stratified validation split), matplotlib (only for plotting)

Typical usage
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
except ImportError:   # allow importing this module without xgboost installed (e.g. when only using the C-index)
    xgb = None

MIN_TIME = 1e-6                 # Cox labels use the sign to distinguish event/censored, so time must be > 0
MAX_CATEGORICAL_UNIQUE = 30     # text columns with too many unique values (e.g. patient_id) are not treated as features by default

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
    """If over 90% of a non-numeric column parses as dates, treat it as a date column."""
    vals = s.dropna().astype(str).head(50)
    if len(vals) == 0:
        return False
    try:
        parsed = pd.to_datetime(vals, errors="coerce", format="mixed")
    except (TypeError, ValueError):      # older pandas versions lack format="mixed"
        return False
    return float(parsed.notna().mean()) >= 0.9


def _require_xgb() -> None:
    if xgb is None:
        raise ImportError("xgboost is required: pip install xgboost")


# ============================================================================
# Evaluation: Harrell C-index
# ============================================================================
def concordance_index(time: Iterable[float], event: Iterable[int], risk: Iterable[float]) -> float:
    """
    Harrell's C: among all "comparable" pairs, the fraction in which the higher-risk subject indeed has the event earlier.
    Comparable = the earlier subject had the event (i is an event and t_i < t_j). Tied risks count as 0.5.
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
# Feature encoding: numeric columns as-is; categorical columns one-hot (binary keeps one column); unseen categories -> all 0
# ============================================================================
def _safe_name(name: str) -> str:
    return re.sub(r"[\[\]<>]", "_", name)   # XGBoost feature names must not contain [ ] <


@dataclass
class Encoder:
    numeric: list[str]
    categorical: dict[str, list[str]]       # column name -> categories seen in training (ordered)
    columns: list[str]                      # encoded column names, fixed order

    @staticmethod
    def _levels(col: str, cats: list[str]) -> list[str]:
        return cats[1:] if len(cats) == 2 else cats      # binary: keep only the second level

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
                arr[isna] = np.nan             # missing values are handled natively by XGBoost
                out[_safe_name(f"{c}={lv}")] = arr
        return pd.DataFrame(out, index=df.index)[self.columns]

    def to_dict(self) -> dict[str, Any]:
        return {"numeric": self.numeric, "categorical": self.categorical, "columns": self.columns}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Encoder":
        return cls(d["numeric"], {k: list(v) for k, v in d["categorical"].items()}, d["columns"])


# ============================================================================
# SHAP result container
# ============================================================================
@dataclass
class ShapResult:
    values: pd.DataFrame        # n × p, SHAP of each feature (log-hazard space)
    base: np.ndarray            # length-n base value (same for every row)
    X: pd.DataFrame             # encoded feature values, aligned with values

    @property
    def base_value(self) -> float:
        return float(np.mean(self.base))

    def margin(self) -> pd.Series:
        """Model margin = base + sum(SHAP); exp(margin) is the relative risk."""
        return self.values.sum(axis=1) + self.base

    def mean_abs(self) -> pd.Series:
        """Global importance: mean |SHAP| per feature (descending)."""
        return self.values.abs().mean().sort_values(ascending=False)

    def top_contributors(self, row: int, k: int = 5) -> pd.DataFrame:
        """The top k features contributing most for a given patient. row is the positional row index (which row), not an index label."""
        sv, xv = self.values.iloc[row], self.X.iloc[row]
        df = pd.DataFrame({"feature": sv.index, "value": xv.values, "shap": sv.values})
        return df.reindex(df["shap"].abs().sort_values(ascending=False).index).head(k).reset_index(drop=True)


# ============================================================================
# Model
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
        self.best_iteration_: int | None = None      # number of trees actually used
        self.baseline_times_: np.ndarray | None = None
        self.baseline_cumhaz_: np.ndarray | None = None
        self.risk_cutpoints_: list[float] | None = None
        self.metrics_: dict[str, float] = {}

    # ------------------------------------------------------------------ Training
    def _default_features(self, df: pd.DataFrame) -> list[str]:
        # exclude the time/event columns of all outcomes (including other outcomes, to avoid information leakage)
        banned = {c for pair in OUTCOME_MAP.values() for c in pair}
        feats = []
        for c in df.columns:
            if c in banned:
                continue
            s = df[c]
            numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            if not numeric and s.nunique() > MAX_CATEGORICAL_UNIQUE:
                continue       # likely an ID / free text
            if not numeric and _looks_like_date(s):
                continue       # enrollment dates etc.: meaningless as categorical features and prone to leaking time information
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
        y = np.where(e == 1, t, -t)                      # Cox label: event +t, censored -t

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

        # metrics + Breslow baseline hazard + risk stratification cutoffs
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

    # --------------------------------------------------------------- Baseline hazard
    def _fit_baseline(self, t: np.ndarray, e: np.ndarray, risk: np.ndarray) -> None:
        """Breslow estimate of the cumulative baseline hazard H0(t) = sum_{event times ti<=t} d_i / sum_{j: t_j>=ti} risk_j"""
        times = np.unique(t[e == 1])
        inc = np.empty(len(times))
        for k, ti in enumerate(times):
            d = np.sum((t == ti) & (e == 1))
            inc[k] = d / np.sum(risk[t >= ti])
        self.baseline_times_, self.baseline_cumhaz_ = times, np.cumsum(inc)

    # --------------------------------------------------------------- Prediction
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
        """Relative risk (larger = higher risk); the values are only meaningful when compared across samples."""
        self._check_fitted()
        return self._risk_from_X(self.encoder.transform(df))

    def predict_survival(self, df: pd.DataFrame, times: Iterable[float]) -> pd.DataFrame:
        """Individual survival probability S(t|x) = exp(-H0(t) * risk). Returns a DataFrame with rows = patients, columns = time points."""
        self._check_fitted()
        times = np.asarray(list(times), dtype=float)
        risk = self.predict_risk(df)
        pos = np.searchsorted(self.baseline_times_, times, side="right") - 1
        h0 = np.where(pos >= 0, self.baseline_cumhaz_[np.clip(pos, 0, None)], 0.0)
        surv = np.exp(-np.outer(risk, h0))
        return pd.DataFrame(surv, index=df.index, columns=[f"S({t:g})" for t in times])

    def risk_group(self, df: pd.DataFrame, labels: tuple[str, str, str] = ("Low", "Medium", "High")) -> pd.Series:
        """Stratify by the training-set risk tertile cutoffs; can be passed directly as the group_by column for KM curves."""
        self._check_fitted()
        risk = self.predict_risk(df)
        g = np.digitize(risk, self.risk_cutpoints_)
        return pd.Series(np.array(labels, dtype=object)[g], index=df.index, name="risk_group")

    # --------------------------------------------------------------- SHAP
    def shap_values(self, df: pd.DataFrame) -> ShapResult:
        """TreeSHAP (XGBoost built-in pred_contribs), log-hazard space."""
        self._check_fitted()
        X = self.encoder.transform(df)
        kw = {"iteration_range": self._iter_range()} if self.best_iteration_ else {}
        contribs = np.asarray(self.booster.predict(xgb.DMatrix(X), pred_contribs=True, **kw), dtype=float)
        values = pd.DataFrame(contribs[:, :-1], index=X.index, columns=X.columns)
        return ShapResult(values=values, base=contribs[:, -1], X=X)

    def check_additivity(self, df: pd.DataFrame) -> float:
        """Return the maximum of |sum(SHAP)+base - margin|; should be close to 0 (~1e-5)."""
        sr = self.shap_values(df)
        return float(np.max(np.abs(sr.margin().to_numpy() - self.predict_margin(df))))

    # --------------------------------------------------------------- Save / load
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
# SHAP plots (hand-drawn with matplotlib, no shap package)
# ============================================================================
def _scale_for_color(x: np.ndarray) -> np.ndarray:
    """Scale feature values to [0,1] for coloring: clip at the 5%-95% quantiles; fall back to min-max if the quantiles are equal."""
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
    """Beeswarm plot: each point is a patient; the x axis is SHAP, color is the feature value (red = high, blue = low, gray = missing)."""
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
    """Global importance: mean(|SHAP|)."""
    import matplotlib.pyplot as plt

    imp = sr.mean_abs().iloc[:max_display][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(imp) + 1.6))
    ax.barh(imp.index, imp.values, color="#2563eb")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global feature importance", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    return fig


def plot_shap_local(sr: ShapResult, row: int = 0, max_display: int = 8):
    """Single-patient explanation: how much each feature raises (red) or lowers (blue) the risk."""
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
# Self-test: python model.py (requires xgboost)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import tempfile

    import matplotlib
    matplotlib.use("Agg")

    if xgb is None:
        sys.exit("The self-test requires xgboost: pip install xgboost")

    rng = np.random.default_rng(0)
    n = 1500
    age = rng.integers(30, 90, n)
    stage = rng.choice(["I", "II", "III", "IV"], n)
    trt = rng.choice(["Drug A", "Drug B", "Placebo"], n)
    sex = rng.choice(["Male", "Female"], n)
    # True risk: older age and later stage are riskier, Drug A is protective
    lin = 0.04 * (age - 60) + 0.5 * np.isin(stage, ["III", "IV"]) - 0.5 * (trt == "Drug A")
    t = rng.exponential(30 / np.exp(lin))
    c = rng.uniform(0, 80, n)
    df = pd.DataFrame({
        "age": age, "sex": sex, "treatment": trt, "stage": stage,
        "os_time": np.minimum(t, c).round(2), "os_event": (t <= c).astype(int),
        "pfs_time": np.minimum(t, c).round(2), "pfs_event": (t <= c).astype(int),
    })
    df.loc[rng.choice(n, 40, replace=False), "age"] = np.nan   # missing values

    m = SurvivalXGB("overall_survival").fit(df)
    print("Metrics:", {k: round(v, 3) for k, v in m.metrics_.items()})
    assert m.metrics_["c_index_val"] > 0.6, "validation C-index is low; the model did not learn the signal"

    sr = m.shap_values(df)
    print("Global importance (mean|SHAP|):\n", sr.mean_abs().round(3).head(6).to_string())
    err = m.check_additivity(df)
    print("Max SHAP additivity error:", err)
    assert err < 1e-3

    surv = m.predict_survival(df.head(3), times=[12, 24, 36])
    print(surv.round(3).to_string())
    assert (np.diff(surv.to_numpy(), axis=1) <= 1e-12).all(), "survival probability should be non-increasing over time"

    with tempfile.TemporaryDirectory() as d:
        m.save(d)
        m2 = SurvivalXGB.load(d)
        assert np.allclose(m.predict_risk(df), m2.predict_risk(df)), "predictions differ after save/load"
    print("Risk stratum sizes:", m.risk_group(df).value_counts().to_dict())

    for name, fig in {"demo_shap_summary": plot_shap_summary(sr), "demo_shap_bar": plot_shap_bar(sr),
                      "demo_shap_local": plot_shap_local(sr, 0)}.items():
        fig.savefig(f"{name}.png", dpi=110, bbox_inches="tight")
    print("All tests passed")
