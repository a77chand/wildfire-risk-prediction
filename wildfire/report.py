"""Build all figures, maps and the results table from saved artefacts."""
from __future__ import annotations

import json

import joblib
import pandas as pd

from . import explain, plots
from . import grid as G
from . import ignitions as I
from .config import Config
from .maps import cell_polygons, folium_risk_map
from .models import BOOSTED
from .train import add_rate_features, load_dataset

CAL = f"{BOOSTED} (calibrated)"


def _frame(js: dict) -> pd.DataFrame:
    return pd.DataFrame(js["data"], index=js["index"], columns=js["columns"])


def make(cfg: Config, verbose: bool = True) -> dict:
    figs = cfg.ensure("reports_dir", "figures")
    metrics = json.load(open(cfg.path("reports_dir") / "metrics.json"))
    preds = pd.read_parquet(cfg.path("data_dir", "processed") / "predictions_test.parquet")
    y = preds["label"].to_numpy()
    grid = G.load_grid(cfg)
    boundary = G.fetch_boundary(cfg)
    art = joblib.load(cfg.path("models_dir") / "boosted.joblib")
    out = {}

    raw = ["Climatology", "Logistic regression", "Random forest", BOOSTED]
    out["comparison"] = plots.fig_model_comparison(y, {n: preds[n].to_numpy() for n in raw},
                                                    figs / "model_comparison.png")
    out["calibration"] = plots.fig_calibration(
        y, {BOOSTED: preds[BOOSTED].to_numpy(), CAL: preds[CAL].to_numpy()}, figs / "calibration.png")
    out["per_year"] = plots.fig_per_year(_frame(metrics["per_year_auc"]), _frame(metrics["per_year_pr"]),
                                         figs / "per_year.png")
    out["ablation"] = plots.fig_ablation(_frame(metrics["ablation"]), figs / "ablation.png")
    out["seasonality"] = plots.fig_seasonality(preds, BOOSTED, figs / "seasonality.png")
    if verbose:
        print("core figures done")

    # ---- SHAP -------------------------------------------------------------------------
    df, _ = add_rate_features(load_dataset(cfg))
    train_df, test_df = df[df.split == "train"], df[df.split == "test"]
    sv = explain.shap_values(art["model"], train_df, test_df.sample(700, random_state=0))
    out["shap_bar"] = plots.fig_shap_bar(sv, figs / "shap_importance.png")
    out["shap_beeswarm"] = plots.fig_shap_beeswarm(sv, figs / "shap_beeswarm.png")
    if verbose:
        print("SHAP done")

    # ---- risk maps for the busiest test week ---------------------------------------------
    ign = I.load_ignitions(cfg)
    week = preds.groupby("date")["n_next"].sum().idxmax()
    day = preds[preds["date"] == week].set_index("cell_id")[BOOSTED]
    nxt = ign[(ign["date"] > week) & (ign["date"] <= week + pd.Timedelta(days=cfg.horizon))]
    title = (f"Predicted ignition risk, week after {week:%d %b %Y}\n"
             f"(model had no knowledge of this period; dots = ignitions that followed)")
    out["map_peak"] = plots.fig_risk_map(grid, cfg.cell_deg, boundary, day, nxt, title,
                                         figs / "risk_map_peak_week.png", vmax=0.9)
    cells = cell_polygons(grid, cfg.cell_deg)
    cells["risk"] = cells["cell_id"].map(day)
    cells["ignitions_next_7d"] = cells["cell_id"].map(
        preds[preds["date"] == week].set_index("cell_id")["n_next"]).fillna(0).astype(int)
    folium_risk_map(cells, boundary, "risk", f"Risk, week after {week:%d %b %Y}", vmax=0.9,
                    extra_cols=["ignitions_next_7d"]).save(cfg.ensure("reports_dir") / "risk_map_peak_week.html")

    # ---- summary table -------------------------------------------------------------------
    tab = pd.DataFrame(metrics["test"])
    cols = ["roc_auc", "pr_auc", "brier_skill", "ece", "capture_top5", "lift_top5"]
    names = ["Climatology", "Logistic regression", "Random forest", BOOSTED, CAL]
    t = tab.loc[names, cols].rename(columns={
        "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "brier_skill": "Brier skill", "ece": "ECE",
        "capture_top5": "Ignitions in top 5% of risk", "lift_top5": "Lift @ top 5%"})
    (cfg.path("reports_dir") / "results.md").write_text(t.round(3).to_markdown())
    out["week"] = str(week.date())
    if verbose:
        print(t.round(3).to_string())
        print("peak week:", week.date())
    return out
