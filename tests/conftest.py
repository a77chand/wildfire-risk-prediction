import numpy as np
import pandas as pd
import pytest

from wildfire.config import Config, load_config


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config()


@pytest.fixture()
def tiny_grid(cfg) -> pd.DataFrame:
    """A 3x3 block of real grid cells (rows 4-6, cols 36-38 of the BC grid)."""
    from wildfire.grid import GridSpec

    spec = GridSpec(cfg.bbox, cfg.cell_deg)
    rows = []
    for r in range(4, 7):
        for c in range(36, 39):
            lat, lon = spec.center(r, c)
            rows.append({"cell_id": int(spec.cell_id(r, c)), "row": r, "col": c, "lat": lat,
                         "lon": lon, "land_frac": 1.0, "elevation": 800.0 + 100 * (r - 4)})
    return pd.DataFrame(rows)


@pytest.fixture()
def tiny_weather(tiny_grid) -> pd.DataFrame:
    """Two years of deterministic synthetic daily weather for the tiny grid."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", "2019-12-31")
    frames = []
    for cid in tiny_grid["cell_id"]:
        doy = dates.dayofyear.to_numpy()
        tmax = 8 + 14 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 3, len(dates))
        frames.append(pd.DataFrame({
            "cell_id": cid, "date": dates, "tmax": tmax, "tmean": tmax - 6,
            "tdew": tmax - 10 - rng.uniform(0, 4, len(dates)),
            "precip": np.where(rng.random(len(dates)) < 0.3, rng.exponential(4, len(dates)), 0.0),
            "wind_ms": rng.uniform(1, 8, len(dates)), "elevation": 800.0}))
    return pd.concat(frames, ignore_index=True)
