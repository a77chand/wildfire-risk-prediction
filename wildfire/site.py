"""Data for the static web demo in ``web/`` (no server: the page reads these JSON files).

Two parts:

* the **back-test** (held-out test years), built once from the exported artefacts and the
  raw ignition records: ``static.json``, ``backtest_<target>.json``, ``fires_test.json``;
* the **current season**, rebuilt weekly by CI: every closed week of this season replayed
  with the deployed models (a forward test on data the model never saw) plus the live
  nowcast: ``season.json``.

Probabilities are stored as integers in units of 1e-5 to keep the files small.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import ignitions as I
from . import live
from .config import Config
from .features import TARGETS
from .grid import GridSpec

SCALE = 100_000
CAUSES = {"Lightning": 0, "Person": 1}            # anything else -> 2 (unknown)
NOW_WEATHER = ["fwi", "dc", "tmax", "rh", "wind_kmh", "precip_30d", "days_since_rain"]


def _art(cfg: Config) -> Path:
    return cfg.root / "artifacts"


def _ints(p) -> list[int]:
    p = np.asarray(p, dtype=float)
    return np.where(np.isnan(p), -1, np.rint(p * SCALE)).astype(int).tolist()


def _events(df: pd.DataFrame, target: str, cell_pos: dict[int, int], week_pos: dict) -> list[list[int]]:
    hit = df[df[target] > 0]
    return [[week_pos[d], cell_pos[c]] for d, c in zip(hit["date"], hit["cell_id"], strict=True)]


def _fires(cfg: Config, ign: pd.DataFrame, grid: pd.DataFrame, origin: pd.Timestamp,
           extra: tuple[str, ...] = ()) -> list[list]:
    """Ignition points inside the kept grid cells: [lat, lon, day since origin, cause, ha, *extra]."""
    spec = GridSpec(cfg.bbox, cfg.cell_deg)
    cid = spec.cell_id(*spec.locate(ign["lat"], ign["lon"]))
    ign = ign[np.isin(cid, grid["cell_id"].to_numpy())].sort_values("date")
    out = []
    for r in ign.itertuples():
        row = [round(r.lat, 3), round(r.lon, 3), int((r.date - origin).days),
               CAUSES.get(r.cause, 2), 0.0 if pd.isna(r.size_ha) else round(float(r.size_ha), 2)]
        out.append(row + [None if pd.isna(v) else v for v in (getattr(r, c, None) for c in extra)])
    return out


def _write(path: Path, obj) -> int:
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))
    return path.stat().st_size


def climatology(cfg: Config) -> dict[str, pd.Series]:
    """Climatology baseline as a (cell, month) table per target. It is fitted on training
    years only, so it is the same for every year and can be read off the test predictions."""
    p = pd.read_parquet(_art(cfg) / "predictions_test.parquet")
    p["month"] = p["date"].dt.month
    return {t: p.groupby(["cell_id", "month"])[f"c_{t}"].first() for t in TARGETS}


def backtest(cfg: Config, out: Path) -> dict[str, int]:
    """Held-out test years 2021-2025: the static half of the site (rebuild after retraining)."""
    art = _art(cfg)
    grid = pd.read_parquet(art / "grid.parquet").sort_values("cell_id")
    preds = pd.read_parquet(art / "predictions_test.parquet").sort_values(["date", "cell_id"])
    weeks = sorted(preds["date"].unique())
    cell_pos = {c: i for i, c in enumerate(grid["cell_id"])}
    week_pos = {d: i for i, d in enumerate(weeks)}
    clim = climatology(cfg)
    months = list(range(cfg["season"]["start_month"], cfg["season"]["end_month"] + 1))
    report = json.loads((cfg.path("reports_dir") / "targets.json").read_text())["targets"]
    temporal = json.loads((cfg.path("reports_dir") / "temporal_skill.json").read_text())

    sizes = {}
    static = {
        "cellDeg": cfg.cell_deg,
        "cells": [[int(r.cell_id), round(r.lat, 3), round(r.lon, 3)] for r in grid.itertuples()],
        "boundary": json.loads((art / "bc_boundary.geojson").read_text()),
        "months": months,
        "clim": {t: [_ints([clim[t].get((c, m), np.nan) for m in months]) for c in grid["cell_id"]]
                 for t in TARGETS},
        "weeks": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in weeks],
        "scores": {t: {"model": {k: round(v["Gradient boosting"], 4) for k, v in report[t]["test"].items()
                                 if k in ("roc_auc", "pr_auc")},
                       "climatology": {k: round(v["Climatology"], 4) for k, v in report[t]["test"].items()
                                       if k in ("roc_auc", "pr_auc")},
                       "spearman": {"model": temporal[t]["Gradient boosting"]["spearman"],
                                    "climatology": temporal[t]["Climatology"]["spearman"]}}
                   for t in TARGETS},
    }
    sizes["static.json"] = _write(out / "static.json", static)
    for t in TARGETS:
        risk = preds.pivot(index="date", columns="cell_id", values=f"p_{t}").loc[weeks, grid["cell_id"]]
        sizes[f"backtest_{t}.json"] = _write(out / f"backtest_{t}.json", {
            "risk": _ints(risk.to_numpy().ravel()), "events": _events(preds, t, cell_pos, week_pos)})

    origin = pd.Timestamp(weeks[0].year, 1, 1)
    ign = I.load_ignitions(cfg)
    ign = ign[ign["date"].dt.year.between(pd.Timestamp(weeks[0]).year, pd.Timestamp(weeks[-1]).year)]
    sizes["fires_test.json"] = _write(out / "fires_test.json", {
        "origin": origin.strftime("%Y-%m-%d"), "fires": _fires(cfg, ign, grid, origin)})
    return sizes


def season(cfg: Config, out: Path, today: pd.Timestamp | None = None) -> dict[str, int]:
    """The current season so far plus the live nowcast: the weekly-refreshed half of the site."""
    art = _art(cfg)
    grid = pd.read_parquet(art / "grid.parquet").sort_values("cell_id").reset_index(drop=True)
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    raw = I.fetch_recent(cfg, year_from=today.year - 1)
    ign = I.clean_bcws(raw, cfg["label"]["fire_types"], cfg["label"]["min_size_ha"],
                       extra=("FIRE_NUMBER", "GEOGRAPHIC_DESCRIPTION"))
    last, rows = live.season_to_date(cfg, today, live.load_artifacts(art), grid=grid,
                                     ign=ign[["date", "lat", "lon", "cause", "size_ha"]])
    rows = rows.sort_values(["date", "cell_id"]).reset_index(drop=True)
    dates = list(rows["date"].unique())
    cell_pos = {c: i for i, c in enumerate(grid["cell_id"])}
    week_pos = {d: i for i, d in enumerate(dates)}
    done = rows[rows["complete"]].copy()
    now = rows[~rows["complete"]].set_index("cell_id").loc[grid["cell_id"]]

    scores = {}
    if len(done):
        clim = climatology(cfg)
        month = done["date"].dt.month.clip(cfg["season"]["start_month"], cfg["season"]["end_month"])
        for t in TARGETS:
            c = pd.MultiIndex.from_arrays([done["cell_id"], month]).map(clim[t]).to_numpy(dtype=float)
            ok = ~np.isnan(done[f"risk_{t}"].to_numpy()) & ~np.isnan(c)
            d, y = done[ok], done[t].to_numpy()[ok].astype(int)
            preds = {"model": d[f"risk_{t}"].to_numpy(), "climatology": c[ok]}
            m = {k: ev.metrics(y, p) for k, p in preds.items()} if 0 < y.sum() < len(y) else {}
            spear = ev.temporal_skill(d, preds, label_col=t)["spearman"]
            scores[t] = {k: {"roc_auc": round(m[k]["roc_auc"], 4), "pr_auc": round(m[k]["pr_auc"], 4)}
                         for k in m} | {"spearman": {k: round(float(v), 4) for k, v in spear.items()},
                                        "events": int(y.sum())}

    origin = pd.Timestamp(today.year, 1, 1)
    fires = ign[ign["date"] > pd.Timestamp(dates[0])]        # windows start the day after
    payload = {
        "year": int(last.year),
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
        "lastObs": last.strftime("%Y-%m-%d"),
        "inSeason": bool(cfg["season"]["start_month"] <= last.month <= cfg["season"]["end_month"]),
        "weeks": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates],
        "complete": [bool(rows.loc[rows["date"] == d, "complete"].iloc[0]) for d in dates],
        "risk": {t: _ints(rows[f"risk_{t}"]) for t in TARGETS},
        "events": {t: _events(done, t, cell_pos, week_pos) for t in TARGETS},
        "origin": origin.strftime("%Y-%m-%d"),
        "fires": _fires(cfg, fires, grid, origin, extra=("FIRE_NUMBER", "GEOGRAPHIC_DESCRIPTION")),
        "now": {k: np.round(now[k].to_numpy(dtype=float), 1).tolist() for k in NOW_WEATHER},
        "scores": scores,
    }
    payload["now"] = {k: [None if np.isnan(v) else v for v in vals] for k, vals in payload["now"].items()}
    return {"season.json": _write(out / "season.json", payload)}


def make(cfg: Config, include_backtest: bool = False, today: pd.Timestamp | None = None) -> dict[str, int]:
    out = cfg.root / "web" / "data"
    out.mkdir(parents=True, exist_ok=True)
    sizes = backtest(cfg, out) if include_backtest or not (out / "static.json").exists() else {}
    return sizes | season(cfg, out, today)
