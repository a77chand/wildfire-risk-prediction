"""Ignition records used to build the prediction target.

Primary source: BC Wildfire Service historical incident points (agency-confirmed
wildfires, open data, no API key). Optional: NASA FIRMS satellite hotspots.
"""
from __future__ import annotations

import io
import time

import pandas as pd
import requests

from .config import Config
from .grid import WFS

BCWS_LAYER = "pub:WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_INCIDENTS_SP"
# The historical layer only gains a fire year once it has closed (fire years run Apr-Mar);
# the season in progress lives in a separate "current fire points" layer with the same fields.
BCWS_CURRENT_LAYER = "pub:WHSE_LAND_AND_NATURAL_RESOURCE.PROT_CURRENT_FIRE_PNTS_SP"
PROPS = "FIRE_YEAR,IGNITION_DATE,FIRE_CAUSE,FIRE_TYPE,CURRENT_SIZE,LATITUDE,LONGITUDE,OBJECTID"
PAGE = 10_000


def _get(params: dict, retries: int = 5) -> dict:
    """The WFS intermittently answers 400/5xx when its database pool is busy; back off and retry."""
    for attempt in range(retries):
        try:
            r = requests.get(WFS, params=params, timeout=180)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(5 * 2 ** attempt)
    raise AssertionError("unreachable")


def _fetch_layer(layer: str, props: str, cql: str | None = None) -> pd.DataFrame:
    """Page through a BCWS WFS layer (server caps a response at 10k features)."""
    frames, start = [], 0
    while True:
        params = {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": layer, "outputFormat": "application/json", "propertyName": props,
            "sortBy": "OBJECTID", "startIndex": start, "count": PAGE,
        }
        if cql:
            params["cql_filter"] = cql
        feats = _get(params)["features"]
        if not feats:
            break
        frames.append(pd.DataFrame([f["properties"] for f in feats]))
        if len(feats) < PAGE:
            break
        start += PAGE
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=props.split(","))


def fetch_bcws(cfg: Config, year_from: int | None = None, force: bool = False) -> pd.DataFrame:
    """Historical (closed) fire years from ``year_from`` on, cached under ``data/raw``."""
    year_from = year_from or cfg["years"]["start"]
    out = cfg.ensure("data_dir", "raw") / f"bcws_incidents_{year_from}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    df = _fetch_layer(BCWS_LAYER, PROPS, f"FIRE_YEAR>={year_from}")
    df.to_parquet(out)
    return df


def fetch_bcws_current() -> pd.DataFrame:
    """Fires of the fire year in progress (never cached: it changes daily)."""
    return _fetch_layer(BCWS_CURRENT_LAYER, PROPS + ",FIRE_NUMBER,GEOGRAPHIC_DESCRIPTION")


def merge_layers(historical: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Historical + current records, without double-counting a fire year present in both
    (around the April rollover the closed year can appear in both layers)."""
    if current.empty:
        return historical
    hist = historical[~historical["FIRE_YEAR"].isin(current["FIRE_YEAR"].unique())]
    return pd.concat([hist, current], ignore_index=True)


def fetch_recent(cfg: Config, year_from: int) -> pd.DataFrame:
    """Every record from ``year_from`` up to today, for nowcasting. Closed fire years rarely
    change, so a cached copy is used if the historical layer is unavailable."""
    try:
        hist = fetch_bcws(cfg, year_from=year_from, force=True)
    except requests.RequestException:
        cache = cfg.path("data_dir", "raw") / f"bcws_incidents_{year_from}.parquet"
        if not cache.exists():
            raise
        print(f"historical layer unavailable; using cached {cache.name}")
        hist = pd.read_parquet(cache)
    return merge_layers(hist, fetch_bcws_current())


def clean_bcws(raw: pd.DataFrame, fire_types=("Fire",), min_size_ha: float = 0.0,
               extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Normalise to columns: date, lat, lon, cause, size_ha (plus any ``extra`` raw columns)."""
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
    keep = ["date", "lat", "lon", "cause", "size_ha"] + [c for c in extra if c in df.columns]
    return df[keep].reset_index(drop=True)


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
