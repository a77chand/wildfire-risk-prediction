import numpy as np
import pandas as pd

from wildfire import evaluate as ev
from wildfire.models import ClimatologyBaseline


def test_metrics_perfect_and_random():
    y = np.array([0] * 90 + [1] * 10)
    perfect = ev.metrics(y, y.astype(float) * 0.9 + 0.05)
    assert perfect["roc_auc"] == 1.0 and perfect["capture_top10"] == 1.0
    rng = np.random.default_rng(0)
    rand = ev.metrics(rng.integers(0, 2, 5000), rng.random(5000))
    assert abs(rand["roc_auc"] - 0.5) < 0.05


def test_capture_and_lift():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.4, 0.5, 0.6, 0.05, 0.01])
    cap, prec = ev.capture_at(y, p, 0.1)       # top 1 of 10 = the 0.9 score
    assert cap == 0.5 and prec == 1.0


def test_climatology_baseline_learns_cell_month_rates():
    n = 400
    df = pd.DataFrame({"cell_id": np.repeat([1, 2], n // 2),
                       "date": pd.Timestamp("2020-07-15")})
    y = np.r_[np.ones(n // 2) * 0, (np.arange(n // 2) % 2)]     # cell 1: never; cell 2: half
    m = ClimatologyBaseline(prior_weight=1.0).fit(df, y)
    p = m.predict_proba(df)[:, 1]
    assert p[0] < 0.05 and 0.4 < p[-1] < 0.6
    unseen = pd.DataFrame({"cell_id": [99], "date": [pd.Timestamp("2020-07-15")]})
    assert 0 < m.predict_proba(unseen)[0, 1] < 1           # falls back to the month rate


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    n_dates, n_cells = 30, 60
    df = pd.DataFrame({"date": np.repeat(pd.date_range("2022-05-01", periods=n_dates, freq="7D"), n_cells)})
    score = rng.random(len(df))
    df["label"] = (rng.random(len(df)) < 0.05 + 0.2 * score).astype(int)
    out = ev.bootstrap_by_date(df, {"a": score, "ref": rng.random(len(df))}, "ref", n_boot=60)
    row = out[(out.model == "a") & (out.metric == "roc_auc")].iloc[0]
    assert row.lo < row["mean"] < row.hi and row.lo > 0.5


def test_temporal_skill_separates_week_level_from_constant_predictions():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-05-01", periods=40, freq="7D")
    intensity = rng.uniform(0.02, 0.4, len(dates))                 # weekly "danger"
    df = pd.DataFrame({"date": np.repeat(dates, 80)})
    df["label"] = (rng.random(len(df)) < np.repeat(intensity, 80)).astype(int)
    tracks = np.repeat(intensity, 80)                               # knows the weekly danger
    flat = np.full(len(df), intensity.mean())                       # climatology-like constant
    skill = ev.temporal_skill(df, {"tracks": tracks, "flat": flat})
    assert skill.loc["tracks", "spearman"] > 0.8 and skill.loc["tracks", "r2"] > 0.5
    assert skill.loc["flat", "r2"] <= 0.01
