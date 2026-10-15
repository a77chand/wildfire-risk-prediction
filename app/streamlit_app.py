"""Interactive wildfire-risk explorer.   Run:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from wildfire import explain, live
from wildfire.config import load_config
from wildfire.features import TARGETS
from wildfire.maps import cell_polygons, folium_risk_map

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIGS = ROOT / "reports" / "figures"
LABELS = {
    "label": "Any ignition",
    "label_large": "Large fire (≥ 10 ha)",
    "label_lightning": "Lightning-caused",
    "label_person": "Person-caused",
}
assert set(LABELS) == set(TARGETS)

st.set_page_config(page_title="BC Wildfire Risk", page_icon="🔥", layout="wide")


@st.cache_resource
def load_all():
    return {
        "cfg": load_config(),
        "arts": live.load_artifacts(ART),
        "grid": pd.read_parquet(ART / "grid.parquet"),
        "boundary": gpd.read_file(ART / "bc_boundary.geojson"),
        "preds": pd.read_parquet(ART / "predictions_test.parquet"),
        "background": {t: pd.read_parquet(ART / f"shap_background_{t}.parquet") for t in TARGETS},
    }


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def run_nowcast():
    d = load_all()
    return live.nowcast(d["cfg"], artifacts=d["arts"], grid=d["grid"])


def show_map(cells, col, title, vmax, extra=None):
    m = folium_risk_map(cells, load_all()["boundary"], col, title, vmax=vmax, extra_cols=extra)
    html = m._repr_html_()  # generated locally from model output only (no user-supplied HTML)
    if hasattr(st, "iframe"):
        st.iframe(html, height=620)
    else:  # older Streamlit
        components.html(html, height=620)


d = load_all()
cfg, grid = d["cfg"], d["grid"]
cells = cell_polygons(grid, cfg.cell_deg)

st.title("🔥 British Columbia wildfire ignition risk")
st.caption(
    "Probability that a ~50 × 35 km grid cell sees a given type of wildfire ignition in the **next 7 days**, "
    "from weather, the Canadian Fire Weather Index System, road access and ignition history. "
    "A research / portfolio project, **not** an operational warning system."
)
target = st.radio("What to predict", list(LABELS), format_func=LABELS.get, horizontal=True)

tab_back, tab_live, tab_report = st.tabs(["Back-test explorer", "Live nowcast", "Model report"])

# ---------------------------------------------------------------------------- back-test
with tab_back:
    preds = d["preds"]
    weeks = sorted(preds["date"].unique())
    busiest = preds.groupby("date")[target].sum().idxmax()
    c1, c2 = st.columns([2, 1])
    week = c1.select_slider("Prediction date (held-out test years only)", options=weeks, value=busiest,
                            format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y"))
    model_name = c2.radio("Model", ["Gradient boosting", "Climatology"], horizontal=True)
    col = f"p_{target}" if model_name == "Gradient boosting" else f"c_{target}"
    sel = preds[preds["date"] == week].set_index("cell_id")
    g = cells.copy()
    g["risk"] = g["cell_id"].map(sel[col])
    g["events_next_7d"] = g["cell_id"].map(sel[target]).fillna(0).astype(int)
    show_map(g, "risk", f"{model_name}: {LABELS[target]}, after {pd.Timestamp(week):%d %b %Y}",
             None, ["events_next_7d"])

    st.markdown("**Crew-allocation game:** if you could only watch the highest-risk cells this week, "
                "how many of the cells that actually had an event would you cover?")
    k = st.slider("Cells watched this week", 5, 150, 30, step=5)
    total = int(sel[target].sum())
    a, b, c = st.columns(3)
    for box, name, colname in ((a, "Gradient boosting", f"p_{target}"), (b, "Climatology", f"c_{target}")):
        top = sel[colname].nlargest(k).index
        box.metric(f"{name}: cells covered", f"{int(sel.loc[top, target].sum())} / {total}")
    c.metric("Random choice (expected)", f"{total * k / len(sel):.1f} / {total}")
    st.caption("This week was never seen in training: models were fitted on 2005–2018.")

# ---------------------------------------------------------------------------- live
with tab_live:
    st.write(
        "Scores every cell using the latest NASA POWER weather (≈3 days behind real time) and the "
        "newest BC Wildfire Service incident records. This is a **nowcast of the 7 days after the "
        "latest observation**, not a weather forecast. First run takes a few minutes (rate-limited API)."
    )
    if st.button("Fetch latest data and score", type="primary"):
        st.session_state["live"] = True
    if st.session_state.get("live"):
        with st.spinner("Fetching weather for ~600 cells…"):
            last, rows = run_nowcast()
        if not bool(rows["in_season"].iloc[0]):
            st.warning("Outside the Apr–Oct season the model was trained for; treat values as indicative only.")
        rcol = f"risk_{target}"
        n_missing = int(rows[rcol].isna().sum())
        st.success(f"Observations through **{last:%d %b %Y}**."
                   + (f" {n_missing} cells had no weather data (shown grey)." if n_missing else ""))
        g = cells.merge(rows[["cell_id", rcol, "fwi", "dc", "tmax", "precip_30d"]], on="cell_id")
        show_map(g, rcol, f"{LABELS[target]}: risk in the 7 days after {last:%d %b %Y}", None,
                 ["fwi", "dc", "tmax", "precip_30d"])

        ranked = rows.dropna(subset=[rcol]).nlargest(10, rcol)
        st.subheader("Highest-risk cells")
        st.dataframe(ranked[["cell_id", "lat", "lon", rcol, "fwi", "dc", "tmax", "precip_30d"]].round(3),
                     hide_index=True)
        pick = st.selectbox("Explain a cell's score (SHAP, log-odds)", ranked["cell_id"])
        art = d["arts"][target]
        row = art["encoder"].transform(rows[rows["cell_id"] == pick])
        contrib = explain.single_row_contributions(art["model"], d["background"][target], row)
        st.bar_chart(contrib.head(10).iloc[::-1])
        st.caption("Positive bars push the probability up, negative bars push it down, "
                   "relative to an average cell-week.")

# ---------------------------------------------------------------------------- report
with tab_report:
    metrics = json.load(open(ROOT / "reports" / "metrics.json"))
    targets = json.load(open(ROOT / "reports" / "targets.json"))["targets"]
    st.subheader("Held-out test years (2021–2025), any-ignition target")
    t = pd.DataFrame(metrics["test"])[["roc_auc", "pr_auc", "brier_skill", "ece", "capture_top5", "lift_top5"]]
    st.dataframe(t.round(3))
    st.subheader("All targets")
    rows_ = []
    for key, e in targets.items():
        for model in ("Climatology", "Gradient boosting"):
            m = {k: v[model] for k, v in e["test"].items()}
            rows_.append({"target": LABELS[key], "model": model, "prevalence": m["prevalence"],
                          "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "brier_skill": m["brier_skill"]})
    st.dataframe(pd.DataFrame(rows_).round(3), hide_index=True)
    for name, cap in (("weekly_capture.png", "Operational value: share of events inside the K cells watched"),
                      ("model_comparison.png", "ROC and precision-recall"),
                      ("targets_summary.png", "Skill by target"),
                      ("target_drivers.png", "What drives each type of ignition"),
                      ("ablation.png", "Feature-group ablation"),
                      ("shap_importance.png", "SHAP importance"),
                      ("calibration.png", "Calibration"),
                      ("per_year.png", "Per-year skill")):
        if (FIGS / name).exists():
            st.image(str(FIGS / name), caption=cap)
