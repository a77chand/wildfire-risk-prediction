"""Bundle the small set of files the Streamlit app needs, so it runs from a fresh clone."""
from __future__ import annotations

import copy
import shutil

import joblib
import pandas as pd

from .config import Config
from .models import BOOSTED
from .train import add_rate_features, load_dataset


def make(cfg: Config) -> dict:
    out = cfg.root / "artifacts"
    out.mkdir(exist_ok=True)

    art = joblib.load(cfg.path("models_dir") / "boosted.joblib")
    model = copy.deepcopy(art["model"])
    model.val = None                        # drop the validation set carried for early stopping
    joblib.dump({"model": model, "encoder": art["encoder"], "features": art["features"],
                 "best_iteration": art["best_iteration"]}, out / "model.joblib", compress=3)

    raw = cfg.path("data_dir", "raw")
    shutil.copy(raw / "grid.parquet", out / "grid.parquet")
    shutil.copy(raw / "bc_boundary.geojson", out / "bc_boundary.geojson")

    preds = pd.read_parquet(cfg.path("data_dir", "processed") / "predictions_test.parquet")
    preds[["cell_id", "date", "label", "n_next", "Climatology", BOOSTED]].to_parquet(
        out / "predictions_test.parquet")

    df, _ = add_rate_features(load_dataset(cfg))
    df[df.split == "train"].sample(300, random_state=0)[art["features"]].to_parquet(
        out / "shap_background.parquet")
    return {p.name: p.stat().st_size for p in out.iterdir()}
