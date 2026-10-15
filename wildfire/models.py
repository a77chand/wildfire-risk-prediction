"""Models: a climatology baseline, a linear model, and two tree ensembles, plus calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .features import MODEL_FEATURES


class ClimatologyBaseline(BaseEstimator, ClassifierMixin):
    """P(ignition) = historical rate for the same cell and month, from training years only.

    Shrunk towards the all-cells rate for that month (``prior_weight`` pseudo-observations)
    so cells with few training weeks are not over-confident. This is the bar any weather
    model must clear: it already knows *where* and *when* fires usually happen.
    """

    def __init__(self, prior_weight: float = 10.0):
        self.prior_weight = prior_weight

    def fit(self, X: pd.DataFrame, y):
        d = self._frame(X).assign(y=np.asarray(y))
        self.global_rate_ = float(d["y"].mean())
        month = d.groupby("month")["y"].agg(["sum", "count"])
        self.month_rate_ = ((month["sum"] + self.prior_weight * self.global_rate_)
                            / (month["count"] + self.prior_weight))
        cm = d.groupby(["cell_id", "month"])["y"].agg(["sum", "count"]).reset_index()
        prior = cm["month"].map(self.month_rate_).fillna(self.global_rate_)
        cm["rate"] = (cm["sum"] + self.prior_weight * prior) / (cm["count"] + self.prior_weight)
        self.table_ = cm.set_index(["cell_id", "month"])["rate"]
        self.classes_ = np.array([0, 1])
        return self

    @staticmethod
    def _frame(X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"cell_id": X["cell_id"].to_numpy(),
                             "month": pd.DatetimeIndex(X["date"]).month})

    def predict_proba(self, X: pd.DataFrame):
        d = self._frame(X)
        idx = pd.MultiIndex.from_arrays([d["cell_id"], d["month"]])
        p = self.table_.reindex(idx).to_numpy()
        fallback = d["month"].map(self.month_rate_).fillna(self.global_rate_).to_numpy()
        p = np.where(np.isnan(p), fallback, p)
        return np.column_stack([1 - p, p])


class FeatureModel(BaseEstimator, ClassifierMixin):
    """Wraps a sklearn-style estimator so it selects ``features`` from a DataFrame."""

    def __init__(self, estimator, features=None):
        self.estimator = estimator
        self.features = features

    def fit(self, X: pd.DataFrame, y):
        self.cols_ = list(self.features or MODEL_FEATURES)
        self.estimator.fit(X[self.cols_], y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.estimator.predict_proba(X[self.cols_])


class BoostedModel(FeatureModel):
    """Gradient boosting whose iteration count is chosen on held-out *years*.

    sklearn's built-in early stopping splits training rows at random, which would leak
    neighbouring weeks/cells into the stopping set. Here we fit once with a generous
    ``max_iter``, pick the iteration with the best validation log-loss from
    ``staged_predict_proba``, then refit with exactly that many iterations.
    """

    def __init__(self, params: dict, seed: int = 42, features=None, val=None, n_iter=None):
        super().__init__(None, features)
        self.params = params
        self.seed = seed
        self.val = val
        self.n_iter = n_iter

    def _make(self, **over):
        return HistGradientBoostingClassifier(
            early_stopping=False, random_state=self.seed, **{**self.params, **over})

    def fit(self, X: pd.DataFrame, y):
        self.cols_ = list(self.features or MODEL_FEATURES)
        n_iter = self.n_iter
        if n_iter is None:
            Xv, yv = self.val
            probe = self._make().fit(X[self.cols_], y)
            ll = [log_loss(yv, p[:, 1], labels=[0, 1])
                  for p in probe.staged_predict_proba(Xv[self.cols_])]
            n_iter = int(np.argmin(ll)) + 1
        self.best_iteration_ = n_iter
        self.estimator = self._make(max_iter=n_iter).fit(X[self.cols_], y)
        self.classes_ = np.array([0, 1])
        return self


class Calibrated:
    """Isotonic calibration fitted on held-out (validation-year) predictions."""

    def __init__(self, model):
        self.model = model
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, X_val, y_val):
        self.iso.fit(self.model.predict_proba(X_val)[:, 1], y_val)
        return self

    def predict_proba(self, X):
        p = self.iso.predict(self.model.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])


BOOSTED = "Gradient boosting"


def build_models(cfg: Config, X_val: pd.DataFrame, y_val) -> dict:
    """Unfitted model zoo; the boosted model picks its size on the validation years."""
    seed = cfg.seed
    p = cfg["model"]
    return {
        "Climatology": ClimatologyBaseline(),
        "Logistic regression": FeatureModel(make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.5))),
        "Random forest": FeatureModel(make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_jobs=-1, random_state=seed, **p["random_forest"]))),
        BOOSTED: BoostedModel(p["gradient_boosting"], seed, val=(X_val, np.asarray(y_val))),
    }


def boosted_fixed(cfg: Config, n_iter: int, features=None) -> BoostedModel:
    """Boosted model with a fixed iteration count (for CV / ablations)."""
    return BoostedModel(cfg["model"]["gradient_boosting"], cfg.seed, features=features, n_iter=n_iter)
