"""Command line: ``wildfire fetch | build | train | report | live | all``."""
from __future__ import annotations

import argparse

from . import features as F
from . import grid as G
from . import human
from . import ignitions as I
from . import report as R
from . import train as T
from . import weather as W
from .config import load_config


def cmd_fetch(cfg):
    grid = G.load_grid(cfg)
    print(f"grid: {len(grid)} cells")
    ign = I.load_ignitions(cfg)
    print(f"ignitions: {len(ign):,} ({ign['date'].min():%Y-%m-%d} .. {ign['date'].max():%Y-%m-%d})")
    W.fetch_weather(cfg, grid)
    grid = W.attach_elevation(cfg, grid, W.load_weather(cfg, grid))
    human.attach_roads(cfg, grid)


def cmd_build(cfg):
    grid = G.load_grid(cfg)
    df = F.build_dataset(cfg, grid, W.load_weather(cfg, grid), I.load_ignitions(cfg))
    out = cfg.ensure("data_dir", "processed") / "dataset.parquet"
    df.to_parquet(out)
    print(f"dataset: {df.shape} -> {out}  (positive rate {df[F.LABEL].mean():.1%})")


def cmd_live(cfg, out=None):
    from .live import nowcast
    from .maps import cell_polygons, folium_risk_map

    last, rows = nowcast(cfg)
    grid = G.load_grid(cfg)
    print(f"observations through {last:%Y-%m-%d}; in season: {bool(rows['in_season'].iloc[0])}")
    cols = ["cell_id", "lat", "lon", "risk_label", "risk_label_large", "risk_label_lightning", "risk_label_person", "fwi", "dc"]
    cols = [c for c in cols if c in rows.columns]
    print(rows.nlargest(10, "risk")[cols].round(3).to_string(index=False))
    if out:
        cells = cell_polygons(grid, cfg.cell_deg).merge(rows[["cell_id", "risk"]], on="cell_id")
        folium_risk_map(cells, G.fetch_boundary(cfg), "risk", f"Risk, 7 days after {last:%d %b %Y}"
                        ).save(out)
        print("map ->", out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wildfire", description=__doc__)
    ap.add_argument("command", choices=["fetch", "build", "tune", "train", "report", "export", "live", "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="skip bootstrap / spatial CV / ablation")
    ap.add_argument("--out", default=None, help="live: write an HTML map here")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.command in ("fetch", "all"):
        cmd_fetch(cfg)
    if a.command in ("build", "all"):
        cmd_build(cfg)
    if a.command == "tune":
        from . import tune

        print(tune.run(cfg).head(5).to_string())
    if a.command in ("train", "all"):
        T.run(cfg, quick=a.quick)
    if a.command in ("report", "all"):
        R.make(cfg)
    if a.command in ("export", "all"):
        from . import export

        for name, size in export.make(cfg).items():
            print(f"artifacts/{name}: {size / 1e3:.0f} kB")
    if a.command == "live":
        cmd_live(cfg, a.out)


if __name__ == "__main__":
    main()
