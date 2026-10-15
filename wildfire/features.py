"""Feature engineering: daily weather -> FWI codes -> weekly cell-level training rows.

Timing contract (prevents leakage): a row for date ``t`` uses weather and ignition
records up to and including day ``t``; its label is whether any ignition occurs in the
cell on days ``t+1 .. t+horizon``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import fwi as fwi_mod
from .config import Config
from .grid import GridSpec

WEATHER_FEATURES = [
    "tmax", "rh", "vpd", "wind_kmh", "precip", "precip_7d", "precip_30d",
    "days_since_rain", "tmax_7d", "rh_min_7d",
]
FWI_FEATURES = ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]
CONTEXT_FEATURES = ["nbr_fwi", "nbr_dc", "fwi_max_7d", "fwi_chg_7d", "tmax_max_7d", "wind_max_7d"]
STATIC_FEATURES = ["lat", "lon", "elevation", "road_log"]
SEASON_FEATURES = ["doy_sin", "doy_cos"]
ACTIVITY_FEATURES = ["ign_cell_7d", "ign_cell_30d", "ign_nbr_30d"]

RATE_FEATURES = ["rate_cell_month", "rate_cell"]   # added by LocalRateEncoder at training time

FEATURES = (WEATHER_FEATURES + FWI_FEATURES + CONTEXT_FEATURES + STATIC_FEATURES
            + SEASON_FEATURES + ACTIVITY_FEATURES)
MODEL_FEATURES = FEATURES + RATE_FEATURES
FEATURE_GROUPS = {
    "Weather": WEATHER_FEATURES,
    "Fire Weather Index codes": FWI_FEATURES,
    "Neighbourhood & trend": CONTEXT_FEATURES,
    "Location / terrain / access": STATIC_FEATURES,
    "Seasonality": SEASON_FEATURES,
    "Recent ignitions": ACTIVITY_FEATURES,
    "Local ignition history": RATE_FEATURES,
}
LABEL = "label"

# Prediction targets: label column -> human description. "label" (any ignition) is the primary one.
TARGETS = {
    "label": "any ignition",
    "label_large": "large fire (>= 10 ha)",
    "label_lightning": "lightning-caused ignition",
    "label_person": "person-caused ignition",
}


def target_masks(ign: pd.DataFrame, large_fire_ha: float) -> dict[str, np.ndarray]:
    """Boolean mask over ``ign`` rows for each target."""
    return {
        "label": np.ones(len(ign), dtype=bool),
        "label_large": (ign["size_ha"].fillna(0) >= large_fire_ha).to_numpy(),
        "label_lightning": (ign["cause"] == "Lightning").to_numpy(),
        "label_person": (ign["cause"] == "Person").to_numpy(),
    }


# ----------------------------------------------------------------------------- helpers
def saturation_vp(t_c):
    """Saturation vapour pressure in hPa (Magnus formula)."""
    return 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))


def rh_from_dewpoint(t_c, td_c):
    return np.clip(100.0 * saturation_vp(td_c) / saturation_vp(t_c), 0.0, 100.0)


def _rolling(a: np.ndarray, window: int, how: str, min_periods: int = 1) -> np.ndarray:
    """Trailing rolling stat along time (axis 1) of a (cells, days) array."""
    r = pd.DataFrame(a.T).rolling(window, min_periods=min_periods)
    return getattr(r, how)().to_numpy().T


def _days_since(a: np.ndarray, threshold: float, cap: int = 60) -> np.ndarray:
    out = np.zeros_like(a, dtype=float)
    count = np.full(a.shape[0], float(cap))
    for t in range(a.shape[1]):
        wet = np.nan_to_num(a[:, t]) >= threshold
        count = np.where(wet, 0.0, np.minimum(count + 1.0, cap))
        out[:, t] = count
    return out


def neighbour_matrix(grid: pd.DataFrame) -> np.ndarray:
    """Dense 8-neighbour adjacency (cells x cells) on the grid's row/col indices."""
    r, c = grid["row"].to_numpy(), grid["col"].to_numpy()
    dr = np.abs(r[:, None] - r[None, :])
    dc = np.abs(c[:, None] - c[None, :])
    return ((dr <= 1) & (dc <= 1) & ~((dr == 0) & (dc == 0))).astype(np.float32)


def ignition_counts(cfg: Config, grid: pd.DataFrame, ign: pd.DataFrame, dates: pd.DatetimeIndex):
    """(cells, days) matrix of ignition counts; returns (counts, n_dropped_outside_grid)."""
    spec = GridSpec(cfg.bbox, cfg.cell_deg)
    row, col = spec.locate(ign["lat"], ign["lon"])
    cid = spec.cell_id(row, col)
    idx = pd.Series(np.arange(len(grid)), index=grid["cell_id"].to_numpy())
    ci = idx.reindex(cid).to_numpy()
    di = dates.get_indexer(pd.DatetimeIndex(ign["date"]).normalize())
    ok = ~np.isnan(ci) & (di >= 0)
    counts = np.zeros((len(grid), len(dates)), dtype=np.float32)
    np.add.at(counts, (ci[ok].astype(int), di[ok]), 1.0)
    return counts, int((~ok).sum())


# --------------------------------------------------------------------- daily feature cube
def daily_arrays(cfg: Config, grid: pd.DataFrame, weather: pd.DataFrame, ign: pd.DataFrame):
    """All per-day features as ``(cells, days)`` arrays plus the ignition-count matrix."""
    dates = pd.date_range(weather["date"].min(), weather["date"].max(), freq="D")
    cells = grid["cell_id"].to_numpy()

    def cube(col):
        piv = weather.pivot(index="date", columns="cell_id", values=col)
        piv = piv.reindex(index=dates, columns=cells)
        piv = piv.interpolate(limit=3, limit_direction="both")  # bridge short gaps only
        return piv.to_numpy().T

    tmax, tdew, precip, wind_ms = cube("tmax"), cube("tdew"), cube("precip"), cube("wind_ms")
    tdew = np.minimum(tdew, tmax)
    rh = rh_from_dewpoint(tmax, tdew)
    wind_kmh = wind_ms * 3.6
    vpd = np.maximum(saturation_vp(tmax) - saturation_vp(tdew), 0.0)

    a: dict[str, np.ndarray] = {
        "tmax": tmax, "rh": rh, "vpd": vpd, "wind_kmh": wind_kmh, "precip": precip,
        "precip_7d": _rolling(np.nan_to_num(precip), 7, "sum"),
        "precip_30d": _rolling(np.nan_to_num(precip), 30, "sum"),
        "days_since_rain": _days_since(precip, 1.0),
        "tmax_7d": _rolling(tmax, 7, "mean", 3),
        "rh_min_7d": _rolling(rh, 7, "min", 3),
    }

    # --- Canadian FWI System, restarted every spring --------------------------------
    f = cfg["fwi"]
    for k in FWI_FEATURES:
        a[k] = np.full(tmax.shape, np.nan)
    for year in sorted(set(dates.year)):
        s = pd.Timestamp(year, f["start_month"], f["start_day"])
        e = pd.Timestamp(year, cfg["season"]["end_month"], 1) + pd.offsets.MonthEnd(0)
        sel = np.where((dates >= s) & (dates <= e))[0]
        if len(sel) == 0:
            continue
        out = fwi_mod.run_season(
            tmax[:, sel], rh[:, sel], wind_kmh[:, sel], np.nan_to_num(precip[:, sel]),
            dates[sel].month.to_numpy(), f["init"],
        )
        for k in FWI_FEATURES:
            a[k][:, sel] = out[k]

    # --- neighbourhood context and short-term trends -------------------------------------
    nbr = neighbour_matrix(grid)
    deg = np.maximum(nbr.sum(axis=1, keepdims=True), 1.0)
    a["nbr_fwi"] = (nbr @ np.nan_to_num(a["fwi"])) / deg
    a["nbr_dc"] = (nbr @ np.nan_to_num(a["dc"])) / deg
    a["fwi_max_7d"] = _rolling(a["fwi"], 7, "max")
    prev = np.full_like(a["fwi"], np.nan)
    prev[:, 7:] = a["fwi"][:, :-7]
    a["fwi_chg_7d"] = a["fwi"] - prev
    a["tmax_max_7d"] = _rolling(tmax, 7, "max", 3)
    a["wind_max_7d"] = _rolling(wind_kmh, 7, "max", 3)

    # --- static + seasonal --------------------------------------------------------------
    nd = len(dates)
    for k in STATIC_FEATURES:
        a[k] = np.repeat(grid[k].to_numpy(dtype=float)[:, None], nd, axis=1)
    doy = dates.dayofyear.to_numpy()
    a["doy_sin"] = np.tile(np.sin(2 * np.pi * doy / 365.25), (len(cells), 1))
    a["doy_cos"] = np.tile(np.cos(2 * np.pi * doy / 365.25), (len(cells), 1))

    # --- recent ignition activity (known at prediction time) -------------------------
    large = float(cfg["label"].get("large_fire_ha", 10.0))
    counts, dropped = {}, 0
    for name, mask in target_masks(ign, large).items():
        counts[name], n_out = ignition_counts(cfg, grid, ign[mask], dates)
        if name == LABEL:
            dropped = n_out
    a["ign_cell_7d"] = _rolling(counts[LABEL], 7, "sum")
    a["ign_cell_30d"] = _rolling(counts[LABEL], 30, "sum")
    a["ign_nbr_30d"] = nbr @ a["ign_cell_30d"]
    return dates, a, counts, dropped


# ----------------------------------------------------------------------------- sampling
def sample_dates(cfg: Config, dates: pd.DatetimeIndex, with_label: bool = True) -> np.ndarray:
    """Indices of prediction dates: every ``stride`` days inside the season, per year."""
    se = cfg["season"]
    stride = int(cfg["sampling"]["stride_days"])
    keep = []
    for year in sorted(set(dates.year)):
        start = pd.Timestamp(year, se["start_month"], 1)
        end = pd.Timestamp(year, se["end_month"], 1) + pd.offsets.MonthEnd(0)
        for d in pd.date_range(start, end, freq=f"{stride}D"):
            i = dates.get_indexer([d])[0]
            if i < 0:
                continue
            if with_label and i + cfg.horizon >= len(dates):
                continue
            keep.append(i)
    return np.array(keep, dtype=int)


def feature_rows(grid: pd.DataFrame, dates, arrays, idx: np.ndarray) -> pd.DataFrame:
    """Long table of features: one row per (cell, selected date index)."""
    cols = {
        "cell_id": np.repeat(grid["cell_id"].to_numpy(), len(idx)),
        "date": np.tile(dates[idx].to_numpy(), len(grid)),
    }
    for k in FEATURES:
        cols[k] = arrays[k][:, idx].ravel().astype(np.float32)
    df = pd.DataFrame(cols)
    df["year"] = df["date"].dt.year
    return df


def make_rows(grid: pd.DataFrame, dates, arrays, counts: dict, idx: np.ndarray, horizon: int):
    """Training table: features at date t; each label = any matching ignition on days t+1 .. t+horizon."""
    nc = len(grid)
    df = feature_rows(grid, dates, arrays, idx)
    for name, c in counts.items():
        cum = np.concatenate([np.zeros((nc, 1), dtype=np.float32), np.cumsum(c, axis=1)], axis=1)
        future = (cum[:, idx + 1 + horizon] - cum[:, idx + 1]).ravel()
        df[name] = (future > 0).astype(np.int8)
        if name == LABEL:
            df["n_next"] = future
    return df.sort_values(["date", "cell_id"]).reset_index(drop=True)


def build_dataset(cfg: Config, grid, weather, ign) -> pd.DataFrame:
    dates, arrays, counts, dropped = daily_arrays(cfg, grid, weather, ign)
    if dropped:
        print(f"features: {dropped} ignitions fell outside the kept grid cells / date range")
    idx = sample_dates(cfg, dates, with_label=True)
    return make_rows(grid, dates, arrays, counts, idx, cfg.horizon)


# --------------------------------------------------------------------------------- splits
def assign_split(years: pd.Series, cfg: Config) -> pd.Series:
    s = cfg["splits"]
    return pd.Series(
        np.select([years <= s["train_end"], years <= s["val_end"]], ["train", "val"], "test"),
        index=years.index,
    )


# ---------------------------------------------------------------- local ignition history
class LocalRateEncoder:
    """Historical ignition rate per cell (and per cell x month), shrunk to the global rate.

    Fit on training years only. When transforming rows from a year the encoder was fitted
    on, that year's own labels are removed (leave-one-year-out), so training rows never see
    their own outcome. Rows from unseen years (validation / test / live) use all training
    years.
    """

    KEYS = {"rate_cell_month": ["cell_id", "month"], "rate_cell": ["cell_id"]}

    def __init__(self, prior_weight: float = 10.0, label_col: str = LABEL):
        self.prior_weight = prior_weight
        self.label_col = label_col

    def fit(self, df: pd.DataFrame):
        d = df.assign(month=df["date"].dt.month)
        self.global_rate_ = float(d[self.label_col].mean())
        self.years_ = set(d["year"].unique())
        self.tot_, self.yr_ = {}, {}
        for name, key in self.KEYS.items():
            g = d.groupby(key + ["year"])[self.label_col].agg(S="sum", N="count").reset_index()
            self.yr_[name] = g.set_index(key + ["year"])
            self.tot_[name] = g.groupby(key)[["S", "N"]].sum()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df[["cell_id", "date", "year"]].assign(month=df["date"].dt.month)
        out = df.copy()
        seen = d["year"].isin(self.years_).to_numpy()
        for name, key in self.KEYS.items():
            tot = d[key].merge(self.tot_[name], left_on=key, right_index=True, how="left")
            S, N = tot["S"].fillna(0).to_numpy(), tot["N"].fillna(0).to_numpy()
            own = d[key + ["year"]].merge(self.yr_[name], left_on=key + ["year"],
                                          right_index=True, how="left")
            S = S - np.where(seen, own["S"].fillna(0).to_numpy(), 0.0)
            N = N - np.where(seen, own["N"].fillna(0).to_numpy(), 0.0)
            w = self.prior_weight
            out[name] = ((S + w * self.global_rate_) / (N + w)).astype(np.float32)
        return out
