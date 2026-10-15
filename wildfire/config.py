"""Configuration: one YAML file defines the whole experiment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path = ROOT

    # --- convenience accessors -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return tuple(self.raw["region"]["bbox"])  # type: ignore[return-value]

    @property
    def cell_deg(self) -> float:
        return float(self.raw["grid"]["cell_deg"])

    @property
    def start_date(self) -> str:
        return f"{self.raw['years']['start']}-01-01"

    @property
    def end_date(self) -> str:
        return f"{self.raw['years']['end']}-12-31"

    @property
    def horizon(self) -> int:
        return int(self.raw["label"]["horizon_days"])

    @property
    def seed(self) -> int:
        return int(self.raw["model"]["seed"])

    def path(self, key: str, *parts: str) -> Path:
        p = self.root / self.raw["paths"][key]
        return p.joinpath(*parts)

    def ensure(self, key: str, *parts: str) -> Path:
        p = self.path(key, *parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as fh:
        return Config(yaml.safe_load(fh))
