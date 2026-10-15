"""Study-area geometry: the BC boundary and a regular lat/lon grid clipped to it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from .config import Config

WFS = "https://openmaps.gov.bc.ca/geo/pub/ows"
BOUNDARY_LAYER = "pub:WHSE_LEGAL_ADMIN_BOUNDARIES.ABMS_PROVINCE_SP"


def fetch_boundary(cfg: Config, force: bool = False):
    """BC provincial boundary (BC Data Catalogue, open licence), cached as GeoJSON."""
    import geopandas as gpd

    out = cfg.ensure("data_dir", "raw") / "bc_boundary.geojson"
    if out.exists() and not force:
        return gpd.read_file(out)
    r = requests.get(
        WFS,
        params={
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": BOUNDARY_LAYER, "outputFormat": "application/json",
            "srsName": "EPSG:4326",
        },
        timeout=120,
    )
    r.raise_for_status()
    gdf = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
    gdf = gdf[["geometry"]].dissolve().reset_index(drop=True)
    # ~500 m simplification: plenty for a 0.5-degree grid and keeps maps light
    gdf["geometry"] = gdf.geometry.simplify(0.005)
    gdf.to_file(out, driver="GeoJSON")
    return gdf


class GridSpec:
    """Index arithmetic for a regular lat/lon grid anchored at the bbox south-west corner."""

    def __init__(self, bbox: tuple[float, float, float, float], cell_deg: float):
        self.west, self.south, self.east, self.north = bbox
        self.d = cell_deg
        self.ncols = int(np.ceil((self.east - self.west) / cell_deg))
        self.nrows = int(np.ceil((self.north - self.south) / cell_deg))

    def locate(self, lat, lon):
        """Return (row, col) for coordinates; -1 where outside the grid."""
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        row = np.floor((lat - self.south) / self.d).astype(int)
        col = np.floor((lon - self.west) / self.d).astype(int)
        bad = (row < 0) | (row >= self.nrows) | (col < 0) | (col >= self.ncols)
        return np.where(bad, -1, row), np.where(bad, -1, col)

    def cell_id(self, row, col):
        row = np.asarray(row)
        col = np.asarray(col)
        return np.where((row < 0) | (col < 0), -1, row * self.ncols + col)

    def center(self, row, col):
        return (self.south + (row + 0.5) * self.d, self.west + (col + 0.5) * self.d)


def build_grid(cfg: Config, boundary) -> pd.DataFrame:
    """Cells overlapping BC by at least ``min_land_fraction`` of their area."""
    from shapely.geometry import box

    spec = GridSpec(cfg.bbox, cfg.cell_deg)
    poly = boundary.geometry.union_all()
    min_frac = float(cfg["grid"]["min_land_fraction"])
    rows = []
    for r in range(spec.nrows):
        for c in range(spec.ncols):
            s, w = spec.south + r * spec.d, spec.west + c * spec.d
            cell = box(w, s, w + spec.d, s + spec.d)
            if not cell.intersects(poly):
                continue
            frac = cell.intersection(poly).area / cell.area
            if frac >= min_frac:
                lat, lon = spec.center(r, c)
                rows.append((int(spec.cell_id(r, c)), r, c, lat, lon, frac))
    grid = pd.DataFrame(rows, columns=["cell_id", "row", "col", "lat", "lon", "land_frac"])
    return grid.sort_values("cell_id").reset_index(drop=True)


def load_grid(cfg: Config) -> pd.DataFrame:
    """Grid table (with elevation once the weather has been fetched)."""
    path = cfg.ensure("data_dir", "raw") / "grid.parquet"
    if path.exists():
        return pd.read_parquet(path)
    grid = build_grid(cfg, fetch_boundary(cfg))
    grid.to_parquet(path)
    return grid
