"""Evaluation that matches how the model would actually be used.

Why these metrics: ignitions are rare (~5% of cell-weeks), so ROC-AUC alone flatters.
We also report PR-AUC (average precision), Brier score (calibration + sharpness), and
"capture": the share of ignition cell-weeks that fall in the top X% of predicted risk,
which is how a dispatcher would pre-position crews.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from .config import Config
from .features import FEATURE_GROUPS, MODEL_FEATURES, RATE_FEATURES
from .models import BOOSTED, ClimatologyBaseline, boosted_fixed


def expected_calibration_error(y, p, bins: int = 10) -> float:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    err, n = 0.0, len(y)
    for b in np.unique(which):
        m = which == b
        err += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(err)


def capture_at(y, p, top_frac: float) -> tuple[float, float]:
    """(share of all positives captured, precision) within the top ``top_frac`` of scores."""
    k = max(1, int(round(top_frac * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    hits = y[top].sum()
    return float(hits / max(y.sum(), 1)), float(hits / k)


def metrics(y, p) -> dict[str, float]:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    base = y.mean()
    brier = brier_score_loss(y, p)
    cap5, prec5 = capture_at(y, p, 0.05)
    cap10, _ = capture_at(y, p, 0.10)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier,
        "brier_skill": 1 - brier / (base * (1 - base)),
        "log_loss": log_loss(y, p),
        "ece": expected_calibration_error(y, p),
        "capture_top5": cap5,
        "capture_top10": cap10,
        "precision_top5": prec5,
        "lift_top5": prec5 / base,
        "prevalence": float(base),
        "n": int(len(y)),
    }


def metrics_table(y, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame({name: metrics(y, p) for name, p in preds.items()}).T


def per_year(df: pd.DataFrame, preds: dict[str, np.ndarray], metric: str = "roc_auc") -> pd.DataFrame:
    rows = {}
    for yr, idx in df.groupby("year").indices.items():
        y = df[LABEL_COL].to_numpy()[idx]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        rows[yr] = {name: metrics(y, p[idx])[metric] for name, p in preds.items()}
    return pd.DataFrame(rows).T


LABEL_COL = "label"


def bootstrap_by_date(df: pd.DataFrame, preds: dict[str, np.ndarray], reference: str,
                      n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Cluster bootstrap over prediction dates (cells on the same day are correlated).

    Returns mean and 95% interval of each model's ROC-AUC / PR-AUC and the paired
    difference to ``reference``.
    """
    rng = np.random.default_rng(seed)
    y = df[LABEL_COL].to_numpy()
    groups = list(df.groupby("date").indices.values())
    out: dict[tuple[str, str], list[float]] = {}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        yb = y[idx]
        if yb.sum() == 0:
            continue
        vals = {}
        for name, p in preds.items():
            vals[name] = (roc_auc_score(yb, p[idx]), average_precision_score(yb, p[idx]))
        for name, (roc, pr) in vals.items():
            out.setdefault((name, "roc_auc"), []).append(roc)
            out.setdefault((name, "pr_auc"), []).append(pr)
            if name != reference:
                out.setdefault((name, "roc_auc - ref"), []).append(roc - vals[reference][0])
                out.setdefault((name, "pr_auc - ref"), []).append(pr - vals[reference][1])
    rows = []
    for (name, m), v in out.items():
        v = np.asarray(v)
        rows.append({"model": name, "metric": m, "mean": v.mean(),
                     "lo": np.quantile(v, 0.025), "hi": np.quantile(v, 0.975)})
    return pd.DataFrame(rows)


def spatial_block_cv(cfg: Config, df: pd.DataFrame, n_iter: int) -> pd.DataFrame:
    """Hold out whole geographic blocks: how well does the model work in places it never saw?

    The per-cell ignition-history rates are excluded here: for a held-out block they would be
    computed from that block's own past labels, which would defeat the purpose of the test.
    """
    cols = [f for f in MODEL_FEATURES if f not in RATE_FEATURES]
    cv = cfg["model"]["spatial_cv"]
    block = cv["block_deg"]
    groups = (np.floor(df["lat"] / block).astype(int).astype(str) + "_"
              + np.floor(df["lon"] / block).astype(int).astype(str)).to_numpy()
    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(cv["n_splits"]).split(df, df[LABEL_COL], groups)):
        model = boosted_fixed(cfg, n_iter, features=cols).fit(df.iloc[tr], df[LABEL_COL].iloc[tr])
        clim = ClimatologyBaseline().fit(df.iloc[tr], df[LABEL_COL].iloc[tr])  # unseen cells -> month rate
        yte = df[LABEL_COL].iloc[te].to_numpy()
        for name, est in ((BOOSTED, model), ("Climatology (no local history)", clim)):
            m = metrics(yte, est.predict_proba(df.iloc[te])[:, 1])
            rows.append({"fold": fold, "model": name, "roc_auc": m["roc_auc"],
                         "pr_auc": m["pr_auc"], "brier_skill": m["brier_skill"]})
    return pd.DataFrame(rows)


def feature_group_ablation(cfg: Config, train: pd.DataFrame, test: pd.DataFrame,
                           n_iter: int) -> pd.DataFrame:
    """Re-train the boosted model with feature groups removed/isolated and score on the test years."""
    G = FEATURE_GROUPS
    where_when = G["Location / terrain"] + G["Seasonality"]
    history = G["Recent ignitions"] + G["Local ignition history"]
    sets = {
        "All features": MODEL_FEATURES,
        "Without FWI codes": [f for f in MODEL_FEATURES if f not in G["Fire Weather Index codes"]],
        "Without ignition history": [f for f in MODEL_FEATURES if f not in history],
        "Location + season + history only": where_when + history,
        "Location + season only": where_when,
        "Weather + FWI + season only": (G["Weather"] + G["Fire Weather Index codes"] + G["Seasonality"]),
    }
    rows = {}
    for name, cols in sets.items():
        m = boosted_fixed(cfg, n_iter, features=cols).fit(train, train[LABEL_COL])
        r = metrics(test[LABEL_COL].to_numpy(), m.predict_proba(test)[:, 1])
        rows[name] = {"n_features": len(cols), "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
                      "brier_skill": r["brier_skill"], "capture_top5": r["capture_top5"]}
    return pd.DataFrame(rows).T
