"""Human-access proxy: density of mapped roads per grid cell.

Person-caused ignitions follow people. The BC Digital Road Atlas is 1.9M segments, far too
big to download, but the WFS can return a *count* for a bounding box, so one tiny request
per cell gives a road-segment count. Segment counts depend partly on digitising detail, so
they are used as a log-scaled relative proxy for access, not as kilometres of road.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from .config import Config
from .grid import WFS

ROADS_LAYER = "pub:WHSE_BASEMAPPING.DRA_DGTL_ROAD_ATLAS_MPAR_SP"
_HITS = re.compile(r'numberMatched="(\d+)"')


def count_roads(lat: float, lon: float, half: float, retries: int = 4) -> int:
    # EPSG:4326 via URN uses lat,lon axis order
    bbox = f"{lat - half},{lon - half},{lat + half},{lon + half},urn:ogc:def:crs:EPSG::4326"
    for attempt in range(retries):
        try:
            r = requests.get(WFS, params={
                "service": "WFS", "version": "2.0.0", "request": "GetFeature",
                "typeNames": ROADS_LAYER, "resultType": "hits", "bbox": bbox}, timeout=60)
            r.raise_for_status()
            m = _HITS.search(r.text)
            if m:
                return int(m.group(1))
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"road count failed for ({lat}, {lon})")


def attach_roads(cfg: Config, grid: pd.DataFrame, workers: int = 4) -> pd.DataFrame:
    """Add ``road_segments`` and ``road_log`` to the grid table (cached)."""
    cache = cfg.ensure("data_dir", "raw") / "roads.parquet"
    if cache.exists():
        roads = pd.read_parquet(cache)
    else:
        half = cfg.cell_deg / 2
        with ThreadPoolExecutor(workers) as ex:
            counts = list(ex.map(lambda r: count_roads(r.lat, r.lon, half), grid.itertuples()))
        roads = pd.DataFrame({"cell_id": grid["cell_id"], "road_segments": counts})
        roads.to_parquet(cache)
    grid = grid.drop(columns=["road_segments", "road_log"], errors="ignore").merge(
        roads, on="cell_id", how="left")
    grid["road_log"] = np.log1p(grid["road_segments"].fillna(0)).astype("float32")
    grid.to_parquet(cfg.ensure("data_dir", "raw") / "grid.parquet")
    return grid
