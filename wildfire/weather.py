"""Daily weather per grid cell from NASA POWER (MERRA-2 reanalysis, no API key)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from .config import Config

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = ["T2M_MAX", "T2M", "T2MDEW", "PRECTOTCORR", "WS10M_MAX"]
FILL = -999.0


def parse_power(payload: dict) -> pd.DataFrame:
    """POWER JSON -> tidy daily frame (date, tmax, tmean, tdew, precip, wind_ms, elevation)."""
    param = payload["properties"]["parameter"]
    df = pd.DataFrame({k: pd.Series(v) for k, v in param.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(FILL, np.nan)
    df = df.rename(columns={
        "T2M_MAX": "tmax", "T2M": "tmean", "T2MDEW": "tdew",
        "PRECTOTCORR": "precip", "WS10M_MAX": "wind_ms",
    })
    df.index.name = "date"
    df["elevation"] = payload["geometry"]["coordinates"][2]
    return df.reset_index()


def fetch_point(lat: float, lon: float, start: str, end: str, retries: int = 6) -> pd.DataFrame:
    """One point, with exponential backoff that honours ``Retry-After`` on HTTP 429."""
    params = {
        "parameters": ",".join(PARAMS), "community": "AG", "latitude": lat, "longitude": lon,
        "start": start.replace("-", ""), "end": end.replace("-", ""), "format": "JSON",
    }
    for attempt in range(retries):
        wait = min(3 * 2 ** attempt, 90)
        try:
            r = requests.get(POWER_URL, params=params, timeout=120)
            if r.status_code == 200:
                return parse_power(r.json())
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
            wait = max(wait, float(r.headers.get("Retry-After", 0) or 0))
        except requests.RequestException:
            if attempt == retries - 1:
                raise
        time.sleep(wait)
    raise RuntimeError(f"POWER request failed after {retries} attempts for ({lat}, {lon})")


def fetch_weather(cfg: Config, grid: pd.DataFrame, workers: int = 6, verbose: bool = True) -> None:
    """Download one parquet per cell (resumable: existing files are skipped)."""
    wdir = cfg.ensure("data_dir", "raw", "weather")
    todo = [r for r in grid.itertuples() if not (wdir / f"{r.cell_id}.parquet").exists()]
    if verbose:
        print(f"weather: {len(grid) - len(todo)} cached, {len(todo)} to fetch")

    def job(r):
        df = fetch_point(r.lat, r.lon, cfg.start_date, cfg.end_date)
        df.to_parquet(wdir / f"{r.cell_id}.parquet")
        return r.cell_id

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(job, r) for r in todo]
        for f in as_completed(futs):
            f.result()
            done += 1
            if verbose and done % 50 == 0:
                print(f"  {done}/{len(todo)}")


def load_weather(cfg: Config, grid: pd.DataFrame) -> pd.DataFrame:
    wdir = cfg.path("data_dir", "raw", "weather")
    frames = []
    for cid in grid["cell_id"]:
        df = pd.read_parquet(wdir / f"{cid}.parquet")
        df.insert(0, "cell_id", cid)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def attach_elevation(cfg: Config, grid: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    elev = weather.groupby("cell_id")["elevation"].first()
    grid = grid.drop(columns=["elevation"], errors="ignore")
    grid = grid.merge(elev, left_on="cell_id", right_index=True, how="left")
    grid.to_parquet(cfg.ensure("data_dir", "raw") / "grid.parquet")
    return grid
