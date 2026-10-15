"""Ignition records used to build the prediction target.

Primary source: BC Wildfire Service historical incident points (agency-confirmed
wildfires, open data, no API key). Optional: NASA FIRMS satellite hotspots.
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from .config import Config
from .grid import WFS

BCWS_LAYER = "pub:WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_INCIDENTS_SP"
PAGE = 10_000


def fetch_bcws(cfg: Config, year_from: int | None = None, force: bool = False) -> pd.DataFrame:
    """Page through the BCWS WFS (server caps a response at 10k features)."""
    year_from = year_from or cfg["years"]["start"]
    out = cfg.ensure("data_dir", "raw") / f"bcws_incidents_{year_from}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)

    props = "FIRE_YEAR,IGNITION_DATE,FIRE_CAUSE,FIRE_TYPE,CURRENT_SIZE,LATITUDE,LONGITUDE,OBJECTID"
    frames, start = [], 0
    while True:
        r = requests.get(
            WFS,
            params={
                "service": "WFS", "version": "2.0.0", "request": "GetFeature",
                "typeNames": BCWS_LAYER, "outputFormat": "application/json",
                "cql_filter": f"FIRE_YEAR>={year_from}", "propertyName": props,
                "sortBy": "OBJECTID", "startIndex": start, "count": PAGE,
            },
            timeout=180,
        )
        r.raise_for_status()
        feats = r.json()["features"]
        if not feats:
            break
        frames.append(pd.DataFrame([f["properties"] for f in feats]))
        if len(feats) < PAGE:
            break
        start += PAGE
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out)
    return df


def clean_bcws(raw: pd.DataFrame, fire_types=("Fire",), min_size_ha: float = 0.0) -> pd.DataFrame:
    """Normalise to columns: date, lat, lon, cause, size_ha."""
    df = raw.copy()
    df = df[df["FIRE_TYPE"].isin(fire_types)]
    df["date"] = pd.to_datetime(df["IGNITION_DATE"].astype("string").str[:10], errors="coerce")
    df = df.rename(columns={
        "LATITUDE": "lat", "LONGITUDE": "lon", "FIRE_CAUSE": "cause", "CURRENT_SIZE": "size_ha",
    })
    df = df.dropna(subset=["date", "lat", "lon"])
    # Guard against typo'd dates (e.g. year 0212): the fire year is authoritative to within
    # one year (the agency fire year straddles the calendar year boundary).
    df = df[(df["date"].dt.year - df["FIRE_YEAR"]).abs() <= 1]
    df = df[df["size_ha"].fillna(0) >= min_size_ha]
    return df[["date", "lat", "lon", "cause", "size_ha"]].reset_index(drop=True)


def fetch_firms(cfg: Config, map_key: str, source: str = "VIIRS_SNPP_SP") -> pd.DataFrame:
    """NASA FIRMS hotspots over the bbox, in <=10-day chunks (needs a free MAP_KEY).

    Note: FIRMS includes non-wildfire heat sources (gas flares, industry), so this is
    an alternative *label*, not a drop-in replacement for confirmed ignitions.
    """
    west, south, east, north = cfg.bbox
    out = cfg.ensure("data_dir", "raw") / f"firms_{source}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    frames = []
    for start in pd.date_range(cfg.start_date, cfg.end_date, freq="10D"):
        url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source}/"
               f"{west},{south},{east},{north}/10/{start:%Y-%m-%d}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        if r.text.strip():
            frames.append(pd.read_csv(io.StringIO(r.text)))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out)
    return df


def clean_firms(raw: pd.DataFrame, min_confidence: str = "n") -> pd.DataFrame:
    """Normalise FIRMS (VIIRS) to the same columns as ``clean_bcws``."""
    df = raw[raw["confidence"].astype(str).str.lower().isin(["n", "h", "nominal", "high"])].copy()
    df["date"] = pd.to_datetime(df["acq_date"])
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
    df["cause"] = "Satellite"
    df["size_ha"] = float("nan")
    return df[["date", "lat", "lon", "cause", "size_ha"]].drop_duplicates().reset_index(drop=True)


def load_ignitions(cfg: Config) -> pd.DataFrame:
    lab = cfg["label"]
    if lab["source"] == "firms":
        import os

        key = os.environ.get("FIRMS_MAP_KEY")
        if not key:
            raise RuntimeError("Set FIRMS_MAP_KEY (free at firms.modaps.eosdis.nasa.gov/api/area).")
        return clean_firms(fetch_firms(cfg, key))
    raw = fetch_bcws(cfg)
    return clean_bcws(raw, lab["fire_types"], lab["min_size_ha"])
