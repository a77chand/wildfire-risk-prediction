import numpy as np
import pandas as pd
import pytest

from wildfire import features as F
from wildfire.config import load_config


@pytest.fixture()
def small_cfg():
    cfg = load_config()
    cfg.raw["grid"]["cell_deg"] = 0.5
    cfg.raw["years"] = {"start": 2018, "end": 2019}
    cfg.raw["splits"] = {"train_end": 2018, "val_end": 2018}
    return cfg


def _ign(dates, lat=50.3, lon=-120.7):
    return pd.DataFrame({"date": pd.to_datetime(dates), "lat": lat, "lon": lon,
                         "cause": "Lightning", "size_ha": 1.0})


def test_rh_from_dewpoint_bounds():
    assert F.rh_from_dewpoint(20.0, 20.0) == pytest.approx(100.0)
    assert 0 <= F.rh_from_dewpoint(30.0, -10.0) < 20


def test_neighbour_matrix(tiny_grid):
    m = F.neighbour_matrix(tiny_grid)
    assert m.shape == (9, 9) and (m == m.T).all() and m.diagonal().sum() == 0
    assert m[4].sum() == 8 and m[0].sum() == 3     # centre cell vs corner


def test_label_uses_only_the_following_week(small_cfg, tiny_grid, tiny_weather):
    ign = _ign(["2019-07-10"])  # lat 50.3 / lon -120.7 falls in the first tiny-grid cell (row 4, col 36)
    dates, arrays, counts, _ = F.daily_arrays(small_cfg, tiny_grid, tiny_weather, ign)
    cell0 = int(np.argmax(counts.sum(axis=1)))
    assert counts[cell0].sum() == 1
    d = dates.get_loc(pd.Timestamp("2019-07-10"))
    idx = np.array([d - 8, d - 7, d - 1, d, d + 1])
    rows = F.make_rows(tiny_grid, dates, arrays, counts, idx, horizon=7)
    r = rows[rows["cell_id"] == tiny_grid["cell_id"].iloc[cell0]].set_index("date")["label"]
    # t = d-8 -> window d-7..d-1 excludes d; t = d-7 .. d-1 include d; t = d (same day) excludes
    assert r[dates[d - 8]] == 0
    assert r[dates[d - 7]] == 1 and r[dates[d - 1]] == 1
    assert r[dates[d]] == 0 and r[dates[d + 1]] == 0


def test_features_do_not_see_the_future(small_cfg, tiny_grid, tiny_weather):
    """Changing weather or ignitions *after* day t must not change any feature at t."""
    ign1 = _ign(["2019-06-01"])
    ign2 = _ign(["2019-06-01", "2019-07-20"])
    w2 = tiny_weather.copy()
    late = w2["date"] > "2019-07-01"
    w2.loc[late, "tmax"] += 15.0
    w2.loc[late, "precip"] = 0.0
    d1, a1, *_ = F.daily_arrays(small_cfg, tiny_grid, tiny_weather, ign1)
    d2, a2, *_ = F.daily_arrays(small_cfg, tiny_grid, w2, ign2)
    t = d1.get_loc(pd.Timestamp("2019-07-01"))
    for k in F.FEATURES:
        np.testing.assert_allclose(a1[k][:, : t + 1], a2[k][:, : t + 1], err_msg=k, equal_nan=True)


def test_sample_dates_are_weekly_in_season(small_cfg):
    dates = pd.date_range("2018-01-01", "2019-12-31")
    idx = F.sample_dates(small_cfg, dates)
    s = dates[idx]
    assert set(s.month) <= set(range(4, 11))
    assert (np.diff(idx)[np.diff(idx) < 100] == 7).all()
    assert idx.max() + small_cfg.horizon < len(dates)


def test_local_rate_encoder_is_leave_one_year_out():
    rows = []
    for year, hit in ((2010, 1), (2011, 0), (2012, 0)):
        rows.append({"cell_id": 1, "date": pd.Timestamp(year, 7, 1), "year": year, "label": hit})
    df = pd.DataFrame(rows)
    enc = F.LocalRateEncoder(prior_weight=0.0).fit(df)
    out = enc.transform(df).set_index("year")["rate_cell"]
    assert out[2010] == 0.0                         # own positive label excluded
    assert out[2011] == pytest.approx(0.5)          # (1 hit) / (2 other years)
    new = df.assign(year=2020, date=pd.Timestamp(2020, 7, 1))
    assert enc.transform(new)["rate_cell"].iloc[0] == pytest.approx(1 / 3)  # unseen year: all history


def test_assign_split_is_chronological(cfg):
    s = F.assign_split(pd.Series([2010, 2019, 2024]), cfg)
    assert list(s) == ["train", "val", "test"]
