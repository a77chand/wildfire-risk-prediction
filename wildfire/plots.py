"""Figures. Palette follows a validated colour-blind-safe categorical order; risk uses a
single-hue sequential ramp; the climatology baseline is neutral grey so it reads as 'reference'."""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve, roc_curve

from .features import FEATURE_GROUPS
from .maps import RISK_RAMP, cell_polygons

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MODEL_STYLE = {
    "Climatology": dict(color=INK2, ls="--", lw=1.6),
    "Logistic regression": dict(color=AQUA, ls="-", lw=1.8),
    "Random forest": dict(color=ORANGE, ls="-", lw=1.8),
    "Gradient boosting": dict(color=BLUE, ls="-", lw=2.4),
}
RISK_CMAP = LinearSegmentedColormap.from_list("risk", RISK_RAMP)
GROUP_COLORS = {
    "Weather": BLUE, "Fire Weather Index codes": ORANGE, "Location / terrain": AQUA,
    "Seasonality": "#eda100", "Recent ignitions": "#6250d6", "Local ignition history": "#e87ba4",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 100,
    })


_style()


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_model_comparison(y, preds: dict[str, np.ndarray], path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, p in preds.items():
        st = MODEL_STYLE[name]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        from sklearn.metrics import average_precision_score, roc_auc_score
        ax[0].plot(fpr, tpr, label=f"{name}  (AUC {roc_auc_score(y, p):.3f})", **st)
        ax[1].plot(rec, prec, label=f"{name}  (AP {average_precision_score(y, p):.3f})", **st)
    ax[0].plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test years)")
    ax[1].axhline(np.mean(y), color=MUTED, lw=0.8, ls=":")
    ax[1].text(0.99, np.mean(y) + 0.01, f"base rate {np.mean(y):.1%}", ha="right", color=MUTED, fontsize=8)
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test years)",
              ylim=(0, 1.0))
    for a in ax:
        a.legend(loc="lower right" if a is ax[0] else "upper right", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, path)


def fig_calibration(y, preds: dict[str, np.ndarray], path, bins: int = 10):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    hi = 0.0
    for name, p in preds.items():
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
        which = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        xs = [p[which == b].mean() for b in np.unique(which)]
        ys = [y[which == b].mean() for b in np.unique(which)]
        st = dict(MODEL_STYLE.get(name.replace(" (calibrated)", ""), MODEL_STYLE["Gradient boosting"]))
        if "calibrated" in name:
            st["ls"] = "--"
            st["color"] = ORANGE
        else:
            st["ls"] = "-"
            st["color"] = BLUE
        ax.plot(xs, ys, marker="o", ms=5, label=name, **{k: v for k, v in st.items()})
        hi = max(hi, max(xs + ys))
    hi *= 1.08
    ax.plot([0, hi], [0, hi], color=MUTED, lw=0.9, ls=":")
    ax.set(xlabel="Predicted probability (mean per bin)", ylabel="Observed ignition frequency",
           title="Calibration: does '30%' mean 30%?", xlim=(0, hi), ylim=(0, hi))
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def fig_per_year(auc: pd.DataFrame, pr: pd.DataFrame, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for a, tab, label in ((ax[0], auc, "ROC-AUC"), (ax[1], pr, "PR-AUC")):
        x = np.arange(len(tab))
        w = 0.38
        a.bar(x - w / 2, tab["Climatology"], w - 0.04, color=INK2, label="Climatology baseline")
        a.bar(x + w / 2, tab["Gradient boosting"], w - 0.04, color=BLUE, label="Gradient boosting")
        a.set_xticks(x, tab.index.astype(int))
        a.set(ylabel=label, title=f"{label} by test year")
        a.grid(axis="x", visible=False)
    ax[0].set_ylim(0.5, 1.0)
    ax[0].legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, path)


def fig_ablation(ab: pd.DataFrame, path):
    ab = ab.sort_values("pr_auc")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(ab.index, ab["pr_auc"], color=[BLUE if i == "All features" else MUTED for i in ab.index],
                   height=0.62)
    for b, (_, r) in zip(bars, ab.iterrows(), strict=False):
        ax.text(b.get_width() + 0.003, b.get_y() + b.get_height() / 2,
                f"{r['pr_auc']:.3f}  (AUC {r['roc_auc']:.3f})", va="center", fontsize=8.5, color=INK2)
    ax.set(xlabel="PR-AUC on test years", title="What does each feature group add?",
           xlim=(0, ab["pr_auc"].max() * 1.35))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def fig_shap_bar(sv, path, top: int = 15):
    imp = pd.Series(np.abs(sv.values).mean(0), index=sv.feature_names).sort_values().tail(top)
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(imp.index, imp.values, color=[GROUP_COLORS[group_of[f]] for f in imp.index], height=0.65)
    ax.set(xlabel="Mean |SHAP| (change in log-odds of ignition)",
           title="What drives the predictions?")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    ax.legend(handles, GROUP_COLORS.keys(), loc="lower right", fontsize=8.5, title="Feature group",
              title_fontsize=8.5)
    fig.tight_layout()
    return _save(fig, path)


def fig_shap_beeswarm(sv, path):
    import shap

    plt.figure(figsize=(7.5, 6))
    shap.plots.beeswarm(sv, max_display=14, show=False, plot_size=None)
    fig = plt.gcf()
    fig.tight_layout()
    return _save(fig, path)


def fig_seasonality(preds: pd.DataFrame, col: str, path):
    m = preds.assign(month=preds["date"].dt.month).groupby("month").agg(
        observed=("label", "mean"), predicted=(col, "mean"))
    names = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(m))
    ax.bar(x, m["observed"], width=0.6, color=GRID, edgecolor=MUTED, lw=0.6, label="Observed (share of cell-weeks with an ignition)")
    ax.plot(x, m["predicted"], color=BLUE, lw=2.4, marker="o", ms=6, label="Mean predicted probability")
    ax.set_xticks(x, [names[i - 4] for i in m.index])
    ax.set(ylabel="Probability of ≥1 ignition in the next 7 days", title="Seasonal cycle on test years")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, path)


def fig_risk_map(grid, cell_deg, boundary, risk: pd.Series, ignitions: pd.DataFrame | None,
                 title: str, path, vmax: float | None = None):
    """Static map: predicted probability per cell, observed ignitions of the following week as dots."""
    g = cell_polygons(grid, cell_deg)
    g["risk"] = g["cell_id"].map(risk)
    vmax = vmax or float(np.nanquantile(g["risk"], 0.995))
    fig, ax = plt.subplots(figsize=(8.2, 8))
    ax.set_facecolor("#f1f0ec")
    g.plot(column="risk", cmap=RISK_CMAP, vmin=0, vmax=vmax, ax=ax, linewidth=0.15,
           edgecolor=SURFACE, missing_kwds={"color": "#e4e3df"}, legend=True,
           legend_kwds={"shrink": 0.6, "label": "Predicted P(≥1 ignition in next 7 days)"})
    boundary.boundary.plot(ax=ax, color=INK, linewidth=0.9)
    if ignitions is not None and len(ignitions):
        ax.scatter(ignitions["lon"], ignitions["lat"], s=16, facecolor=SURFACE, edgecolor=INK,
                   linewidth=0.8, zorder=5, label=f"Observed ignitions ({len(ignitions)})")
        ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    ax.set_aspect(1 / np.cos(np.deg2rad(54.5)))
    ax.set(title=title, xlabel="Longitude", ylabel="Latitude")
    ax.grid(False)
    fig.tight_layout()
    return _save(fig, path)
