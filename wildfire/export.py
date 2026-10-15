"""Bundle the small set of files the Streamlit app needs, so it runs from a fresh clone."""
from __future__ import annotations

import copy
import shutil

import joblib
import pandas as pd

from .config import Config
from .features import TARGETS
from .train import add_rate_features, load_dataset


def make(cfg: Config) -> dict:
    out = cfg.root / "artifacts"
    out.mkdir(exist_ok=True)
    for old in out.glob("*"):          # keep the bundle exactly in sync with the latest run
        old.unlink()

    base = load_dataset(cfg)
    for target in TARGETS:
        art = joblib.load(cfg.path("models_dir") / f"boosted_{target}.joblib")
        model = copy.deepcopy(art["model"])
        model.val = None                    # drop the validation set carried for iteration selection
        joblib.dump({**art, "model": model}, out / f"model_{target}.joblib", compress=3)
        df, _ = add_rate_features(base, target)
        df[df.split == "train"].sample(300, random_state=0)[art["features"]].to_parquet(
            out / f"shap_background_{target}.parquet")

    raw = cfg.path("data_dir", "raw")
    shutil.copy(raw / "grid.parquet", out / "grid.parquet")
    shutil.copy(raw / "bc_boundary.geojson", out / "bc_boundary.geojson")
    pd.read_parquet(cfg.path("data_dir", "processed") / "predictions_targets_test.parquet").to_parquet(
        out / "predictions_test.parquet")
    return {p.name: p.stat().st_size for p in sorted(out.iterdir())}
