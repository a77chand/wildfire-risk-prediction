"""Nowcast: score every cell using the most recent weather and ignition records.

NASA POWER lags real time by ~3 days, so the output describes risk for the 7 days
following the latest available observation date, not a weather *forecast*.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import features as F
from . import grid as G
from . import ignitions as I
from . import weather as W
from .config import Config


def load_artifacts(directory: Path, pattern: str = "model_{t}.joblib") -> dict[str, dict]:
    """Load one model artefact per target from ``directory`` (``models/`` or ``artifacts/``)."""
    return {t: joblib.load(directory / pattern.format(t=t)) for t in F.TARGETS
            if (directory / pattern.format(t=t)).exists()}


def fetch_recent_weather(grid: pd.DataFrame, start: str, end: str, workers: int = 3) -> pd.DataFrame:
    """Recent weather for every cell. Cells that still fail after retries are skipped (and
    reported) rather than aborting the whole nowcast; they come back with no risk estimate."""
    def job(r):
        try:
            df = W.fetch_point(r.lat, r.lon, start, end)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  skipped cell {r.cell_id}: {exc}")
            return None
        df.insert(0, "cell_id", r.cell_id)
        return df

    with ThreadPoolExecutor(workers) as ex:
        frames = [f for f in ex.map(job, grid.itertuples()) if f is not None]
    return pd.concat(frames, ignore_index=True)


def nowcast(cfg: Config, today: pd.Timestamp | None = None, artifacts: dict[str, dict] | None = None,
            weather: pd.DataFrame | None = None,
            grid: pd.DataFrame | None = None) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Return (observation date, per-cell frame with features and ``risk_<target>`` columns).

    ``risk`` is an alias of ``risk_label`` (any ignition in the next 7 days).
    """
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    artifacts = artifacts or load_artifacts(cfg.path("models_dir"), "boosted_{t}.joblib")
    grid = grid if grid is not None else G.load_grid(cfg)
    if weather is None:
        cache = cfg.ensure("data_dir", "live") / f"weather_{today:%Y-%m-%d}.parquet"
        if cache.exists():
            weather = pd.read_parquet(cache)
        else:
            start = pd.Timestamp(today.year, 1, 15).strftime("%Y-%m-%d")  # 30-day features + FWI start-up
            weather = fetch_recent_weather(grid, start, today.strftime("%Y-%m-%d"))
            if weather["cell_id"].nunique() >= 0.95 * len(grid):
                weather.to_parquet(cache)

    # latest day for which (almost) every cell has data
    have = weather.dropna(subset=["tmax", "tdew", "precip", "wind_ms"]).groupby("date")["cell_id"].nunique()
    last = have[have >= 0.95 * len(grid)].index.max()
    weather = weather[weather["date"] <= last]

    ign = I.clean_bcws(I.fetch_bcws(cfg, year_from=today.year - 1, force=True),
                       cfg["label"]["fire_types"], cfg["label"]["min_size_ha"])
    dates, arrays, _, _ = F.daily_arrays(cfg, grid, weather, ign)
    rows = F.feature_rows(grid, dates, arrays, np.array([len(dates) - 1]))
    for target, art in artifacts.items():
        scored = art["encoder"].transform(rows)
        risk = art["model"].predict_proba(scored)[:, 1]
        rows[f"risk_{target}"] = np.where(rows["tmax"].isna(), np.nan, risk)  # no weather -> no estimate
        rows[f"rate_cell_{target}"] = scored["rate_cell"]
    if "risk_label" in rows:
        rows["risk"] = rows["risk_label"]
    in_season = cfg["season"]["start_month"] <= last.month <= cfg["season"]["end_month"]
    rows["in_season"] = bool(in_season)
    return last, rows
