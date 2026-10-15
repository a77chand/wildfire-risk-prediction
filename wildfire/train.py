"""End-to-end training: fit models on early years, calibrate on validation years, score on test years."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import evaluate as ev
from .config import Config
from .features import LABEL, MODEL_FEATURES, LocalRateEncoder, assign_split
from .models import BOOSTED, Calibrated, build_models


def load_dataset(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.path("data_dir", "processed", "dataset.parquet"))
    df["split"] = assign_split(df["year"], cfg).to_numpy()
    return df


def add_rate_features(df: pd.DataFrame) -> tuple[pd.DataFrame, LocalRateEncoder]:
    """Fit the local-history encoder on training years and apply it to every row."""
    enc = LocalRateEncoder().fit(df[df.split == "train"])
    return enc.transform(df), enc


def run(cfg: Config, quick: bool = False, verbose: bool = True) -> dict:
    df, encoder = add_rate_features(load_dataset(cfg))
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    if verbose:
        print(f"train {len(tr):,} ({tr.year.min()}-{tr.year.max()})  "
              f"val {len(va):,} ({va.year.min()}-{va.year.max()})  "
              f"test {len(te):,} ({te.year.min()}-{te.year.max()})")

    models = build_models(cfg, va, va[LABEL])
    for name, m in models.items():
        if verbose:
            print(f"fitting {name} ...")
        m.fit(tr, tr[LABEL])
    boosted = models[BOOSTED]
    best_iter = int(boosted.best_iteration_)
    if verbose:
        print(f"{BOOSTED}: best_iteration = {best_iter}")

    # isotonic calibration on the validation years (never on test)
    calibrated = {f"{n} (calibrated)": Calibrated(models[n]).fit(va, va[LABEL])
                  for n in (BOOSTED, "Random forest", "Logistic regression")}
    all_models = {**models, **calibrated}

    pred_val = {n: m.predict_proba(va)[:, 1] for n, m in all_models.items()}
    pred_te = {n: m.predict_proba(te)[:, 1] for n, m in all_models.items()}
    tab_val = ev.metrics_table(va[LABEL].to_numpy(), pred_val)
    tab_te = ev.metrics_table(te[LABEL].to_numpy(), pred_te)

    result = {"best_iteration": best_iter, "test": tab_te, "val": tab_val}
    shown = ["Climatology", "Logistic regression", "Random forest", BOOSTED, f"{BOOSTED} (calibrated)"]
    if verbose:
        print("\nTEST metrics\n", tab_te.loc[shown, ["roc_auc", "pr_auc", "brier_skill", "ece",
                                                     "capture_top5", "lift_top5"]].round(3))

    if not quick:
        boot_models = ["Climatology", "Logistic regression", "Random forest", BOOSTED]
        result["bootstrap"] = ev.bootstrap_by_date(
            te, {n: pred_te[n] for n in boot_models}, reference="Climatology")
        result["per_year_auc"] = ev.per_year(te, {n: pred_te[n] for n in ("Climatology", BOOSTED)})
        result["per_year_pr"] = ev.per_year(te, {n: pred_te[n] for n in ("Climatology", BOOSTED)}, "pr_auc")
        trval = pd.concat([tr, va], ignore_index=True)
        result["spatial_cv"] = ev.spatial_block_cv(cfg, trval, best_iter)
        result["ablation"] = ev.feature_group_ablation(cfg, tr, te, best_iter)
        if verbose:
            print("\nSpatial-block CV (mean over folds)\n",
                  result["spatial_cv"].groupby("model")[["roc_auc", "pr_auc", "brier_skill"]].mean().round(3))
            print("\nFeature-group ablation (test)\n", result["ablation"].round(3))

    # ---- persist artefacts ---------------------------------------------------------
    models_dir = cfg.ensure("models_dir")
    joblib.dump({"model": boosted, "calibrator": calibrated[f"{BOOSTED} (calibrated)"].iso,
                 "features": MODEL_FEATURES, "encoder": encoder,
                 "best_iteration": best_iter}, models_dir / "boosted.joblib")
    for n in ("Climatology", "Logistic regression", "Random forest"):
        joblib.dump(models[n], models_dir / f"{n.lower().replace(' ', '_')}.joblib")

    preds = te[["cell_id", "date", LABEL, "n_next"]].copy()
    for n, p in pred_te.items():
        preds[n] = p.astype(np.float32)
    preds.to_parquet(cfg.ensure("data_dir", "processed") / "predictions_test.parquet")
    val_preds = va[["cell_id", "date", LABEL]].copy()
    for n, p in pred_val.items():
        val_preds[n] = p.astype(np.float32)
    val_preds.to_parquet(cfg.ensure("data_dir", "processed") / "predictions_val.parquet")

    out = {k: (v.round(5).to_dict() if isinstance(v, pd.DataFrame) else v)
           for k, v in result.items() if k not in ("bootstrap", "per_year_auc", "per_year_pr",
                                                   "spatial_cv", "ablation")}
    for k in ("bootstrap", "per_year_auc", "per_year_pr", "spatial_cv", "ablation"):
        if k in result:
            out[k] = json.loads(result[k].to_json(orient="split"))
    out["split"] = {"train_years": [int(tr.year.min()), int(tr.year.max())],
                    "val_years": [int(va.year.min()), int(va.year.max())],
                    "test_years": [int(te.year.min()), int(te.year.max())],
                    "rows": {"train": len(tr), "val": len(va), "test": len(te)}}
    with open(cfg.ensure("reports_dir") / "metrics.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    result["data"] = {"train": tr, "val": va, "test": te}
    result["preds_test"] = preds
    result["models"] = all_models
    return result
