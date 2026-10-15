"""Hyperparameter search with expanding-window (forward-chaining) cross-validation by year.

Each fold trains on all years before a validation block, so tuning never rewards a model for
using the future. Per fold the iteration count is picked from ``staged_predict_proba``; the
configuration is scored by the mean validation log-loss at those iterations. (Local-history
rate features use all training years leave-one-year-out, which is slightly optimistic but
identical across configurations, so it does not change the ranking.)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss

from .config import Config
from .features import LABEL, MODEL_FEATURES
from .train import add_rate_features, load_dataset

SPACE = {
    "learning_rate": [0.03, 0.05, 0.08, 0.12],
    "max_leaf_nodes": [7, 15, 31, 63],
    "min_samples_leaf": [50, 100, 200, 400],
    "l2_regularization": [0.0, 1.0, 5.0, 20.0],
    "max_features": [0.5, 0.8, 1.0],
}
FOLDS = [(2010, (2011, 2012)), (2013, (2014, 2015)), (2016, (2017, 2018))]  # (train_end, val years)


def sample_configs(base: dict, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    cfgs = [{k: base[k] for k in SPACE}]                    # always include the current default
    while len(cfgs) < n:
        c = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        if c not in cfgs:
            cfgs.append(c)
    return cfgs


def run(cfg: Config, n_configs: int = 14, max_iter: int = 600, verbose: bool = True) -> pd.DataFrame:
    df, _ = add_rate_features(load_dataset(cfg))
    df = df[df["year"] <= cfg["splits"]["val_end"]]          # never touch the test years
    base = dict(cfg["model"]["gradient_boosting"])
    rows = []
    for i, params in enumerate(sample_configs(base, n_configs, cfg.seed)):
        t0 = time.time()
        lls, prs, iters = [], [], []
        for train_end, vyears in FOLDS:
            tr, va = df[df["year"] <= train_end], df[df["year"].isin(vyears)]
            m = HistGradientBoostingClassifier(
                **{**base, **params, "max_iter": max_iter}, early_stopping=False,
                random_state=cfg.seed).fit(tr[MODEL_FEATURES], tr[LABEL])
            stages = [p[:, 1] for p in m.staged_predict_proba(va[MODEL_FEATURES])]
            ll = [log_loss(va[LABEL], p) for p in stages]
            k = int(np.argmin(ll))
            lls.append(ll[k])
            iters.append(k + 1)
            prs.append(average_precision_score(va[LABEL], stages[k]))
        rows.append({**params, "val_logloss": float(np.mean(lls)), "val_pr_auc": float(np.mean(prs)),
                     "mean_best_iter": float(np.mean(iters)), "seconds": round(time.time() - t0)})
        if verbose:
            print(f"[{i + 1}/{n_configs}] ll={rows[-1]['val_logloss']:.5f} "
                  f"pr={rows[-1]['val_pr_auc']:.4f} it={rows[-1]['mean_best_iter']:.0f} {params}")
    res = pd.DataFrame(rows).sort_values("val_logloss").reset_index(drop=True)
    res.to_json(cfg.ensure("reports_dir") / "tuning.json", orient="records", indent=2)
    return res
