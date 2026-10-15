"""Interactive wildfire-risk explorer.   Run:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from wildfire import explain, live
from wildfire.config import load_config
from wildfire.maps import cell_polygons, folium_risk_map

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIGS = ROOT / "reports" / "figures"
BOOSTED = "Gradient boosting"

st.set_page_config(page_title="BC Wildfire Risk", page_icon="🔥", layout="wide")


@st.cache_resource
def load_all():
    cfg = load_config()
    return {
        "cfg": cfg,
        "art": joblib.load(ART / "model.joblib"),
        "grid": pd.read_parquet(ART / "grid.parquet"),
        "boundary": gpd.read_file(ART / "bc_boundary.geojson"),
        "preds": pd.read_parquet(ART / "predictions_test.parquet"),
        "background": pd.read_parquet(ART / "shap_background.parquet"),
    }


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def run_nowcast():
    d = load_all()
    last, rows = live.nowcast(d["cfg"], artifact=d["art"], grid=d["grid"])
    return last, rows


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
    "Probability that a ~50 × 35 km grid cell sees at least one wildfire ignition in the **next 7 days**, "
    "from weather, the Canadian Fire Weather Index System and ignition history. "
    "A research / portfolio project, **not** an operational warning system."
)

tab_back, tab_live, tab_report = st.tabs(["Back-test explorer", "Live nowcast", "Model report"])

# ---------------------------------------------------------------------------- back-test
with tab_back:
    preds = d["preds"]
    weeks = sorted(preds["date"].unique())
    busiest = preds.groupby("date")["n_next"].sum().idxmax()
    c1, c2 = st.columns([2, 1])
    week = c1.select_slider("Prediction date (held-out test years only)", options=weeks, value=busiest,
                            format_func=lambda x: pd.Timestamp(x).strftime("%d %b %Y"))
    model_name = c2.radio("Model", [BOOSTED, "Climatology"], horizontal=True)
    sel = preds[preds["date"] == week].set_index("cell_id")
    g = cells.copy()
    g["risk"] = g["cell_id"].map(sel[model_name])
    g["ignitions_next_7d"] = g["cell_id"].map(sel["n_next"]).fillna(0).astype(int)
    show_map(g, "risk", f"{model_name}: risk after {pd.Timestamp(week):%d %b %Y}", 0.9, ["ignitions_next_7d"])

    k = max(1, int(0.05 * len(sel)))
    top = sel[model_name].nlargest(k).index
    hit = int((sel.loc[top, "n_next"] > 0).sum())
    total = int((sel["n_next"] > 0).sum())
    a, b, c = st.columns(3)
    a.metric("Cells with an ignition that week", total)
    b.metric(f"…of them in the top 5% of risk ({k} cells)", hit)
    c.metric("Ignitions that week", int(sel["n_next"].sum()))
    st.caption("This week was never seen in training: the model was fitted on 2005–2018.")

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
        n_missing = int(rows["risk"].isna().sum())
        st.success(f"Observations through **{last:%d %b %Y}**."
                   + (f" {n_missing} cells had no weather data (shown grey)." if n_missing else ""))
        g = cells.merge(rows[["cell_id", "risk", "fwi", "dc", "tmax", "precip_30d"]], on="cell_id")
        show_map(g, "risk", f"Risk, 7 days after {last:%d %b %Y}", None, ["fwi", "dc", "tmax", "precip_30d"])

        ranked = rows.dropna(subset=["risk"]).nlargest(10, "risk")
        st.subheader("Highest-risk cells")
        st.dataframe(ranked[["cell_id", "lat", "lon", "risk", "fwi", "dc", "tmax", "precip_30d"]].round(3),
                     hide_index=True)
        pick = st.selectbox("Explain a cell's score (SHAP, log-odds)", ranked["cell_id"])
        row = rows[rows["cell_id"] == pick]
        contrib = explain.single_row_contributions(d["art"]["model"], d["background"], row)
        st.bar_chart(contrib.head(10).iloc[::-1])
        st.caption("Positive bars push the ignition probability up, negative bars push it down, "
                   "relative to an average cell-week.")

# ---------------------------------------------------------------------------- report
with tab_report:
    metrics = json.load(open(ROOT / "reports" / "metrics.json"))
    t = pd.DataFrame(metrics["test"])[["roc_auc", "pr_auc", "brier_skill", "ece", "capture_top5", "lift_top5"]]
    st.subheader("Held-out test years")
    st.dataframe(t.round(3))
    st.caption("capture_top5: share of ignition cell-weeks that fall in the top 5% of predicted risk. "
               "lift_top5: precision in that top 5% divided by the base rate.")
    for name, cap in (("model_comparison.png", "ROC and precision-recall"),
                      ("ablation.png", "Feature-group ablation"),
                      ("shap_importance.png", "SHAP importance"),
                      ("calibration.png", "Calibration"),
                      ("per_year.png", "Per-year skill")):
        if (FIGS / name).exists():
            st.image(str(FIGS / name), caption=cap)
