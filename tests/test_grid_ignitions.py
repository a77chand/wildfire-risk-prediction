import numpy as np
import pandas as pd

from wildfire.grid import GridSpec
from wildfire.ignitions import clean_bcws
from wildfire.weather import parse_power


def test_grid_locate_roundtrip():
    spec = GridSpec((-139.1, 48.3, -114.0, 60.0), 0.5)
    row, col = spec.locate([50.3, 59.99, 70.0], [-120.0, -114.1, -120.0])
    cid = spec.cell_id(row, col)
    assert cid[2] == -1                         # outside the bbox
    lat, lon = spec.center(row[0], col[0])
    assert abs(lat - 50.3) <= 0.25 and abs(lon + 120.0) <= 0.25
    assert cid[0] != cid[1]


def test_clean_bcws_filters_types_and_bad_dates():
    raw = pd.DataFrame({
        "FIRE_YEAR": [2020, 2020, 2020, 2020],
        "IGNITION_DATE": ["2020-07-01Z", "2020-07-02Z", "0212-01-12Z", None],
        "FIRE_TYPE": ["Fire", "Nuisance Fire", "Fire", "Fire"],
        "FIRE_CAUSE": ["Lightning"] * 4, "CURRENT_SIZE": [1.0, 0.01, 5.0, 2.0],
        "LATITUDE": [50.0] * 4, "LONGITUDE": [-120.0] * 4,
    })
    out = clean_bcws(raw)
    assert len(out) == 1 and out["date"].iloc[0] == pd.Timestamp("2020-07-01")


def test_parse_power_handles_fill_values_and_elevation():
    payload = {
        "geometry": {"coordinates": [-120.0, 50.0, 1261.1]},
        "properties": {"parameter": {
            "T2M_MAX": {"20200101": 5.0, "20200102": -999.0}, "T2M": {"20200101": 1.0, "20200102": 0.0},
            "T2MDEW": {"20200101": -2.0, "20200102": -1.0}, "PRECTOTCORR": {"20200101": 0.0, "20200102": 3.0},
            "WS10M_MAX": {"20200101": 2.0, "20200102": 4.0}}},
    }
    df = parse_power(payload)
    assert list(df.columns[:2]) == ["date", "tmax"] or "tmax" in df.columns
    assert np.isnan(df.loc[1, "tmax"]) and df["elevation"].iloc[0] == 1261.1
    assert df["date"].iloc[0] == pd.Timestamp("2020-01-01")
