"""Canadian Forest Fire Weather Index (FWI) System, vectorised across grid cells.

Implements the equations of Van Wagner (1987), "Development and structure of the
Canadian Forest Fire Weather Index System" (Forestry Tech. Rep. 35):

    FFMC  fine fuel moisture code        (hours)
    DMC   duff moisture code             (~2 weeks)
    DC    drought code                   (~months)
    ISI   initial spread index  = f(FFMC, wind)
    BUI   buildup index         = f(DMC, DC)
    FWI   fire weather index    = f(ISI, BUI)

Inputs are *noon* temperature (C), relative humidity (%), wind (km/h) and 24 h
precipitation (mm). Daily-resolution reanalysis does not provide noon values, so the
pipeline feeds daily-maximum temperature, the relative humidity implied by the daily
mean dew point at that temperature, and daily-maximum wind. That is a documented
approximation (it runs slightly "hot/dry/windy"); the codes remain strongly
informative because they integrate weather history, which is the point of using them.
"""
from __future__ import annotations

import numpy as np

# Monthly day-length adjustment for DMC (Van Wagner 1987, ~46N) and DC seasonal term.
DMC_DAYLENGTH = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.2, 8.0, 7.0, 6.0])
DC_DAYLENGTH = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])


def ffmc(temp, rh, wind, rain, prev):
    mo = 147.2 * (101.0 - prev) / (59.5 + prev)
    rf = np.where(rain > 0.5, rain - 0.5, 0.0)
    wet = rf > 0
    rf_safe = np.where(wet, rf, 1.0)
    mr = mo + 42.5 * rf_safe * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf_safe))
    mr = np.where(mo > 150.0, mr + 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf_safe), mr)
    mo = np.where(wet, np.minimum(mr, 250.0), mo)

    ed = (0.942 * rh ** 0.679 + 11.0 * np.exp((rh - 100.0) / 10.0)
          + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh)))
    ew = (0.618 * rh ** 0.753 + 10.0 * np.exp((rh - 100.0) / 10.0)
          + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh)))

    ko = 0.424 * (1 - (rh / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind) * (1 - (rh / 100.0) ** 8)
    kd = ko * 0.581 * np.exp(0.0365 * temp)
    k1 = (0.424 * (1 - ((100.0 - rh) / 100.0) ** 1.7)
          + 0.0694 * np.sqrt(wind) * (1 - ((100.0 - rh) / 100.0) ** 8))
    kw = k1 * 0.581 * np.exp(0.0365 * temp)

    m = np.where(mo > ed, ed + (mo - ed) * 10.0 ** (-kd),
                 np.where(mo < ew, ew - (ew - mo) * 10.0 ** (-kw), mo))
    return 59.5 * (250.0 - m) / (147.2 + m)


def dmc(temp, rh, rain, prev, month):
    temp = np.maximum(temp, -1.1)
    le = DMC_DAYLENGTH[month - 1]
    rk = 1.894 * (temp + 1.1) * (100.0 - rh) * le * 1e-4
    wet = rain > 1.5
    re = np.where(wet, 0.92 * rain - 1.27, 1.0)
    prev_s = np.maximum(prev, 1e-6)
    mo = 20.0 + np.exp(5.6348 - prev_s / 43.43)
    b = np.where(prev_s <= 33, 100.0 / (0.5 + 0.3 * prev_s),
                 np.where(prev_s <= 65, 14.0 - 1.3 * np.log(prev_s), 6.2 * np.log(prev_s) - 17.2))
    mr = mo + 1000.0 * re / (48.77 + b * re)
    pr = np.maximum(244.72 - 43.43 * np.log(np.maximum(mr - 20.0, 1e-6)), 0.0)
    after_rain = np.where(wet, pr, prev)
    return after_rain + rk


def dc(temp, rain, prev, month):
    temp = np.maximum(temp, -2.8)
    lf = DC_DAYLENGTH[month - 1]
    pe = np.maximum((0.36 * (temp + 2.8) + lf) / 2.0, 0.0)
    wet = rain > 2.8
    rd = np.where(wet, 0.83 * rain - 1.27, 0.0)
    qo = 800.0 * np.exp(-prev / 400.0)
    qr = qo + 3.937 * rd
    dr = np.where(wet, np.maximum(400.0 * np.log(800.0 / qr), 0.0), prev)
    return dr + pe


def isi(wind, ffmc_):
    m = 147.2 * (101.0 - ffmc_) / (59.5 + ffmc_)
    f_w = np.exp(0.05039 * wind)
    f_f = 91.9 * np.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
    return 0.208 * f_w * f_f


def bui(dmc_, dc_):
    denom = dmc_ + 0.4 * dc_
    denom = np.where(denom == 0, 1.0, denom)
    low = 0.8 * dmc_ * dc_ / denom
    high = dmc_ - (1.0 - 0.8 * dc_ / denom) * (0.92 + (0.0114 * dmc_) ** 1.7)
    return np.maximum(np.where(dmc_ <= 0.4 * dc_, low, high), 0.0)


def fwi(isi_, bui_):
    f_d = np.where(bui_ <= 80.0, 0.626 * bui_ ** 0.809 + 2.0,
                   1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui_)))
    b = 0.1 * isi_ * f_d
    b_safe = np.where(b > 1.0, b, 2.0)
    s = np.exp(2.72 * (0.434 * np.log(b_safe)) ** 0.647)
    return np.where(b > 1.0, s, b)


def step(temp, rh, wind, rain, month, prev):
    """Advance all six indices by one day. ``prev`` is a dict with ffmc/dmc/dc arrays."""
    rh = np.clip(rh, 0.0, 100.0)
    f = ffmc(temp, rh, wind, rain, prev["ffmc"])
    d = dmc(temp, rh, rain, prev["dmc"], month)
    c = dc(temp, rain, prev["dc"], month)
    i = isi(wind, f)
    b = bui(d, c)
    return {"ffmc": f, "dmc": d, "dc": c, "isi": i, "bui": b, "fwi": fwi(i, b)}


def run_season(temp, rh, wind, rain, months, init):
    """Run the FWI System over one season.

    Arrays are shaped ``(n_cells, n_days)``; ``months`` is ``(n_days,)``. Returns a dict of
    ``(n_cells, n_days)`` arrays. Missing weather carries yesterday's codes forward.
    """
    n_cells, n_days = temp.shape
    state = {k: np.full(n_cells, v, dtype=float) for k, v in init.items()}
    out = {k: np.full((n_cells, n_days), np.nan) for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi")}
    for t in range(n_days):
        ok = ~(np.isnan(temp[:, t]) | np.isnan(rh[:, t]) | np.isnan(wind[:, t]) | np.isnan(rain[:, t]))
        new = step(
            np.where(ok, temp[:, t], 0.0), np.where(ok, rh[:, t], 50.0),
            np.where(ok, wind[:, t], 0.0), np.where(ok, rain[:, t], 0.0), int(months[t]), state,
        )
        for k in ("ffmc", "dmc", "dc"):
            state[k] = np.where(ok, new[k], state[k])
        for k in out:
            out[k][:, t] = np.where(ok, new[k], np.nan)
    return out
