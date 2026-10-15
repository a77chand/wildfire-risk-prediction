import numpy as np

from wildfire import fwi


def test_van_wagner_reference_day():
    """Reference values from Van Wagner & Pickett (1985): T=17, RH=42, W=25, rain=0, April."""
    prev = {"ffmc": np.array([85.0]), "dmc": np.array([6.0]), "dc": np.array([15.0])}
    out = fwi.step(np.array([17.0]), np.array([42.0]), np.array([25.0]), np.array([0.0]), 4, prev)
    expected = {"ffmc": 87.692, "dmc": 8.545, "dc": 19.013, "isi": 10.853, "bui": 8.490, "fwi": 10.096}
    for k, v in expected.items():
        assert abs(float(out[k][0]) - v) < 0.01, k


def test_rain_lowers_moisture_codes():
    prev = {"ffmc": np.array([90.0]), "dmc": np.array([40.0]), "dc": np.array([300.0])}
    args = (np.array([20.0]), np.array([50.0]), np.array([10.0]))
    dry = fwi.step(*args, np.array([0.0]), 7, prev)
    wet = fwi.step(*args, np.array([25.0]), 7, prev)
    for k in ("ffmc", "dmc", "dc", "fwi"):
        assert wet[k][0] < dry[k][0], k


def test_indices_non_negative_and_finite():
    rng = np.random.default_rng(1)
    n = 500
    prev = {"ffmc": rng.uniform(10, 95, n), "dmc": rng.uniform(0, 150, n), "dc": rng.uniform(0, 700, n)}
    out = fwi.step(rng.uniform(-10, 35, n), rng.uniform(5, 100, n), rng.uniform(0, 60, n),
                   rng.exponential(3, n), 6, prev)
    for k, v in out.items():
        assert np.isfinite(v).all(), k
        assert (v >= 0).all(), k


def test_hotter_drier_windier_increases_fwi():
    prev = {"ffmc": np.array([85.0]), "dmc": np.array([30.0]), "dc": np.array([200.0])}
    mild = fwi.step(np.array([12.0]), np.array([70.0]), np.array([5.0]), np.array([0.0]), 7, prev)
    harsh = fwi.step(np.array([32.0]), np.array([20.0]), np.array([30.0]), np.array([0.0]), 7, prev)
    assert harsh["fwi"][0] > mild["fwi"][0]


def test_run_season_carries_state_over_missing_days():
    t = np.full((1, 5), 20.0)
    rh = np.full((1, 5), 40.0)
    w = np.full((1, 5), 15.0)
    r = np.zeros((1, 5))
    t[0, 2] = np.nan
    out = fwi.run_season(t, rh, w, r, np.full(5, 7), {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0})
    assert np.isnan(out["fwi"][0, 2])
    assert np.isfinite(out["fwi"][0, [0, 1, 3, 4]]).all()
    assert out["dc"][0, 3] > out["dc"][0, 1]  # drought code keeps accumulating
