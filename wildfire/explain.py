"""Model-agnostic SHAP attributions (additive in log-odds) for the boosted model."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _logit_fn(model):
    est, cols = model.estimator, model.cols_

    def f(arr):
        return est.decision_function(pd.DataFrame(arr, columns=cols))

    return f


def shap_values(model, background: pd.DataFrame, explain: pd.DataFrame, seed: int = 0):
    """SHAP values (log-odds) via the permutation explainer; works for any sklearn model."""
    import shap

    cols = model.cols_
    bg = background[cols].sample(min(60, len(background)), random_state=seed).to_numpy()
    explainer = shap.PermutationExplainer(_logit_fn(model), bg, seed=seed)
    ex = explain[cols].to_numpy()
    sv = explainer(ex, max_evals=2 * len(cols) + 1, silent=True)
    sv.feature_names = list(cols)
    return sv


def single_row_contributions(model, background: pd.DataFrame, row: pd.DataFrame) -> pd.Series:
    sv = shap_values(model, background, row)
    return pd.Series(sv.values[0], index=model.cols_).sort_values(key=np.abs, ascending=False)
