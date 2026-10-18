import json

import numpy as np
import pandas as pd

from wildfire import features as F
from wildfire import ignitions as I
from wildfire import live, site


def test_merge_layers_drops_fire_years_present_in_current_layer():
    hist = pd.DataFrame({"FIRE_YEAR": [2024, 2025, 2025], "OBJECTID": [1, 2, 3]})
    cur = pd.DataFrame({"FIRE_YEAR": [2025, 2025], "OBJECTID": [10, 11]})
    out = I.merge_layers(hist, cur)
    assert sorted(out["OBJECTID"]) == [1, 10, 11]          # 2025 comes only from the current layer
    assert I.merge_layers(hist, cur.iloc[:0]).equals(hist)


class _ConstModel:
    def predict_proba(self, X):
        p = np.full(len(X), 0.1)
        return np.c_[1 - p, p]


class _FlatEncoder:
    def transform(self, df):
        return df.assign(rate_cell_month=0.05, rate_cell=0.05)


def test_season_replay_never_scores_past_the_last_observation(cfg, tiny_grid, tiny_weather):
    weather = tiny_weather[tiny_weather["date"].between("2019-01-15", "2019-08-20")]
    cell = int(tiny_grid["cell_id"].iloc[4])
    lat, lon = tiny_grid.set_index("cell_id").loc[cell, ["lat", "lon"]]
    ign = pd.DataFrame({"date": pd.to_datetime(["2019-06-05", "2019-08-19", "2019-08-20"]), "lat": lat,
                        "lon": lon, "cause": ["Lightning", "Person", "Person"], "size_ha": [12.0, 0.1, 0.1]})
    arts = {t: {"model": _ConstModel(), "encoder": _FlatEncoder(), "features": F.MODEL_FEATURES}
            for t in F.TARGETS}

    last, rows = live.season_to_date(cfg, pd.Timestamp("2019-08-20"), arts, weather=weather,
                                     grid=tiny_grid, ign=ign)

    assert last == pd.Timestamp("2019-08-20")
    done, now = rows[rows["complete"]], rows[~rows["complete"]]
    assert (now["date"] == last).all() and len(now) == len(tiny_grid)
    assert (done["date"] + pd.Timedelta(days=cfg.horizon) <= last).all()   # every window has closed
    assert {f"risk_{t}" for t in F.TARGETS} <= set(rows.columns)
    # 5 Jun lies in the window of the 3 Jun prediction (4..10 Jun), 19 Aug in that of 12 Aug
    # (13..19 Aug). 20 Aug belongs to the 19 Aug week, whose window is still open: not scored.
    hit = done[done["label"] == 1]
    hit.index = hit["date"].dt.strftime("%m-%d")
    assert list(hit.index) == ["06-03", "08-12"] and (hit["cell_id"] == cell).all()
    assert hit["label_large"].tolist() == [1, 0]
    assert hit["label_lightning"].tolist() == [1, 0]
    assert hit["label_person"].tolist() == [0, 1]
    assert pd.Timestamp("2019-08-19") not in set(done["date"])


def test_site_encoding_has_no_nan():
    assert site._ints([0.5, np.nan, 1e-6]) == [50000, -1, 0]
    assert json.loads(json.dumps({"x": site._ints([np.nan])}, allow_nan=False)) == {"x": [-1]}
