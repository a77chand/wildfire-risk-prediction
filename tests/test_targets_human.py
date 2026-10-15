import numpy as np
import pandas as pd

from wildfire import evaluate as ev
from wildfire import features as F
from wildfire import human
from wildfire.config import load_config


def _ign(rows):
    return pd.DataFrame(rows, columns=["date", "lat", "lon", "cause", "size_ha"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def test_target_masks_split_by_size_and_cause():
    ign = _ign([("2019-07-01", 50, -120, "Lightning", 0.01), ("2019-07-02", 50, -120, "Person", 50.0),
                ("2019-07-03", 50, -120, "Unknown", 12.0)])
    m = F.target_masks(ign, large_fire_ha=10)
    assert m["label"].all()
    assert m["label_large"].tolist() == [False, True, True]
    assert m["label_lightning"].tolist() == [True, False, False]
    assert m["label_person"].tolist() == [False, True, False]


def test_multi_target_labels_are_independent(tiny_grid, tiny_weather):
    cfg = load_config()
    cfg.raw["years"] = {"start": 2018, "end": 2019}
    ign = _ign([("2019-07-10", 50.3, -120.7, "Lightning", 0.02),
                ("2019-07-12", 50.3, -120.7, "Person", 40.0)])
    dates, arrays, counts, _ = F.daily_arrays(cfg, tiny_grid, tiny_weather, ign)
    d = dates.get_loc(pd.Timestamp("2019-07-08"))
    rows = F.make_rows(tiny_grid, dates, arrays, counts, np.array([d]), horizon=7)
    hit = rows[rows["label"] == 1]
    assert len(hit) == 1                              # both ignitions are in the same cell
    r = hit.iloc[0]
    assert r["label_lightning"] == 1 and r["label_person"] == 1 and r["label_large"] == 1
    assert r["n_next"] == 2
    assert rows["label_large"].sum() == 1


def test_local_rate_encoder_uses_the_requested_target():
    df = pd.DataFrame({
        "cell_id": 1, "year": [2010, 2011, 2012], "date": pd.to_datetime(["2010-07-01", "2011-07-01", "2012-07-01"]),
        "label": [1, 1, 1], "label_person": [0, 0, 0]})
    any_rate = F.LocalRateEncoder(0.0).fit(df).transform(df)["rate_cell"]
    person_rate = F.LocalRateEncoder(0.0, label_col="label_person").fit(df).transform(df)["rate_cell"]
    assert (any_rate == 1.0).all() and (person_rate == 0.0).all()


def test_weekly_topk_capture_ranks_within_each_week():
    # Week A has high base risk everywhere, week B low. A pooled top-k would only look at week A;
    # ranking per week must still find week B's event.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-06-01"] * 4 + ["2020-06-08"] * 4),
        "label": [0, 0, 0, 1, 0, 1, 0, 0]})
    p = np.array([.9, .8, .7, .95, .1, .3, .2, .15])
    tab = ev.weekly_topk_capture(df, {"m": p}, ks=(1, 2))
    assert tab.loc[1, "m"] == 1.0                      # best cell each week contains both events
    assert tab.loc[1, "Random"] == 0.25 and tab.loc[2, "Random"] == 0.5


class _Resp:
    status_code = 200

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def test_count_roads_parses_hits_and_uses_lat_lon_bbox(monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen.update(params)
        return _Resp('<wfs:FeatureCollection numberMatched="7198" numberReturned="0"/>')

    monkeypatch.setattr(human.requests, "get", fake_get)
    assert human.count_roads(50.25, -120.25, 0.25) == 7198
    # EPSG:4326 URN axis order is lat,lon
    assert seen["bbox"].startswith("50.0,-120.5,50.5,-120.0")
    assert seen["resultType"] == "hits"
