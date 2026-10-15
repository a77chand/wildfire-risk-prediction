"""End-to-end training: fit models on early years, calibrate on validation years, score on test years."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import evaluate as ev
from .config import Config
from .features import (
    CONTEXT_FEATURES,
    LABEL,
    MODEL_FEATURES,
    TARGETS,
    LocalRateEncoder,
    assign_split,
)
from .models import (
    BOOSTED,
    BoostedModel,
    Calibrated,
    ClimatologyBaseline,
    boosted_fixed,
    build_models,
)


def load_dataset(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.path("data_dir", "processed", "dataset.parquet"))
    df["split"] = assign_split(df["year"], cfg).to_numpy()
    return df


def add_rate_features(df: pd.DataFrame, target: str = LABEL) -> tuple[pd.DataFrame, LocalRateEncoder]:
    """Fit the local-history encoder (for ``target``) on training years and apply it to every row."""
    enc = LocalRateEncoder(label_col=target).fit(df[df.split == "train"])
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

    cap_models = ["Climatology", "Logistic regression", "Random forest", BOOSTED]
    result["weekly_capture"] = ev.weekly_topk_capture(te, {n: pred_te[n] for n in cap_models})
    if verbose:
        print("\nShare of ignition cell-weeks inside the K cells watched each week\n",
              result["weekly_capture"].round(3))

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
                 "best_iteration": best_iter}, models_dir / "boosted_label.joblib")
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
                                                   "spatial_cv", "ablation", "weekly_capture")}
    for k in ("bootstrap", "per_year_auc", "per_year_pr", "spatial_cv", "ablation", "weekly_capture"):
        if k in result:
            out[k] = json.loads(result[k].to_json(orient="split"))
    out["split"] = {"train_years": [int(tr.year.min()), int(tr.year.max())],
                    "val_years": [int(va.year.min()), int(va.year.max())],
                    "test_years": [int(te.year.min()), int(te.year.max())],
                    "rows": {"train": len(tr), "val": len(va), "test": len(te)}}
    with open(cfg.ensure("reports_dir") / "metrics.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    result["targets"] = run_targets(cfg, quick=quick, verbose=verbose)
    result["data"] = {"train": tr, "val": va, "test": te}
    result["preds_test"] = preds
    result["models"] = all_models
    return result


def run_targets(cfg: Config, quick: bool = False, verbose: bool = True) -> dict:
    """Train a climatology baseline and a boosted model for every target.

    Targets: any ignition, large fire, lightning-caused, person-caused. Each gets its own
    local-history rates (a lightning-prone cell is not necessarily a person-caused-prone one),
    its own validation-year iteration count, held-out metrics, SHAP drivers and model file.
    """
    from . import explain

    base = load_dataset(cfg)
    models_dir = cfg.ensure("models_dir")
    summary, drivers, pred_cols = {}, {}, {}
    for target, desc in TARGETS.items():
        df, enc = add_rate_features(base, target)
        tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
        clim = ClimatologyBaseline().fit(tr, tr[target])
        bm = BoostedModel(cfg["model"]["gradient_boosting"], cfg.seed,
                          val=(va, va[target].to_numpy())).fit(tr, tr[target])
        p_clim, p_boost = clim.predict_proba(te)[:, 1], bm.predict_proba(te)[:, 1]
        y = te[target].to_numpy()
        tab = ev.metrics_table(y, {"Climatology": p_clim, BOOSTED: p_boost})
        entry = {"description": desc, "best_iteration": int(bm.best_iteration_),
                 "test": tab.round(5).to_dict(),
                 "weekly_capture": ev.weekly_topk_capture(
                     te, {"Climatology": p_clim, BOOSTED: p_boost}, ks=(10, 30, 50),
                     label_col=target).round(5).to_dict()}
        if not quick:
            boot = ev.bootstrap_by_date(te, {"Climatology": p_clim, BOOSTED: p_boost},
                                        reference="Climatology", n_boot=150, label_col=target)
            entry["bootstrap"] = json.loads(boot.to_json(orient="records"))
            # controlled comparison: do road density + neighbourhood/trend features help this target?
            old_cols = [f for f in MODEL_FEATURES if f not in CONTEXT_FEATURES + ["road_log"]]
            p_old = boosted_fixed(cfg, int(bm.best_iteration_), features=old_cols).fit(
                tr, tr[target]).predict_proba(te)[:, 1]
            entry["new_features"] = {
                name: {k: v for k, v in ev.metrics(y, p).items() if k in ("roc_auc", "pr_auc", "log_loss")}
                for name, p in (("without", p_old), ("with", p_boost))}
            sv = explain.shap_values(bm, tr, te.sample(min(500, len(te)), random_state=0))
            imp = pd.Series(np.abs(sv.values).mean(0), index=sv.feature_names).sort_values(ascending=False)
            drivers[target] = imp.head(12).round(5).to_dict()
        summary[target] = entry
        pred_cols[f"p_{target}"], pred_cols[f"c_{target}"] = p_boost.astype(np.float32), p_clim.astype(np.float32)
        joblib.dump({"model": bm, "encoder": enc, "features": MODEL_FEATURES,
                     "best_iteration": int(bm.best_iteration_), "target": target},
                    models_dir / f"boosted_{target}.joblib")
        if verbose:
            t = tab.loc[:, ["roc_auc", "pr_auc", "brier_skill", "capture_top5"]].round(3)
            print(f"\n[{target}] {desc}  (test prevalence {y.mean():.2%})\n{t}")

    preds = te[["cell_id", "date", "n_next", *TARGETS]].copy()
    for k, v in pred_cols.items():
        preds[k] = v
    preds.to_parquet(cfg.ensure("data_dir", "processed") / "predictions_targets_test.parquet")
    out = {"targets": summary, "drivers": drivers}
    with open(cfg.ensure("reports_dir") / "targets.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    return out
