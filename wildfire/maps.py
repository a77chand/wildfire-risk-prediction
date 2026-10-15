"""Map helpers: grid cells as polygons, and an interactive Folium risk map."""
from __future__ import annotations

import pandas as pd


def cell_polygons(grid: pd.DataFrame, cell_deg: float):
    import geopandas as gpd
    from shapely.geometry import box

    h = cell_deg / 2
    geom = [box(r.lon - h, r.lat - h, r.lon + h, r.lat + h) for r in grid.itertuples()]
    return gpd.GeoDataFrame(grid.copy(), geometry=geom, crs="EPSG:4326")


# single-hue (orange) sequential ramp, light -> dark
RISK_RAMP = ["#fff1e6", "#fdd0a8", "#f9a870", "#eb6834", "#c4491b", "#8f3010", "#5e1f0a"]


def folium_risk_map(cells, boundary, prob_col: str = "risk", title: str = "Wildfire ignition risk",
                    vmax: float | None = None, extra_cols: list[str] | None = None):
    """Interactive map of cell-level probabilities with hover tooltips."""
    import branca.colormap as cm
    import folium

    vmax = vmax or max(0.05, float(cells[prob_col].quantile(0.99)))
    cmap = cm.LinearColormap(RISK_RAMP, vmin=0, vmax=vmax, caption=f"{title} (probability)")
    m = folium.Map(location=[54.7, -125.5], zoom_start=5, tiles="OpenStreetMap",
                   control_scale=True)
    tip_cols = [prob_col] + (extra_cols or [])
    data = cells[tip_cols + ["geometry"]].copy()
    data[prob_col] = data[prob_col].round(3)
    data = data.astype({c: object for c in tip_cols}).where(data.notna(), None)  # None -> grey cell
    folium.GeoJson(
        data.to_json(),
        style_function=lambda f: {
            "fillColor": ("#b5b4ae" if f["properties"][prob_col] is None
                          else cmap(min(f["properties"][prob_col], vmax))),
            "color": "#ffffff", "weight": 0.4, "fillOpacity": 0.8,
        },
        tooltip=folium.GeoJsonTooltip(fields=tip_cols, aliases=[c.replace("_", " ") for c in tip_cols]),
    ).add_to(m)
    folium.GeoJson(boundary.to_json(), style_function=lambda f: {
        "fillOpacity": 0, "color": "#0b0b0b", "weight": 1.2}).add_to(m)
    cmap.add_to(m)
    return m
