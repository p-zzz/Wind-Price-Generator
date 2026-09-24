"""
YAML config surface for the generator. Replaces the thesis repo's pattern of
editing module-level constants directly in the script -- the config file *is*
the interface now, threaded explicitly through generate()/load_fitted_objects()
rather than read off module globals.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Shipped price model is in-domain only up to this wind.scale / solar.scale.
IN_DOMAIN_SCALE_MAX = 1.25


@dataclass
class WindConfig:
    scale: float
    onshore_capacity_mw: float
    offshore_capacity_mw: float


@dataclass
class SolarConfig:
    scale: float
    capacity_mw: float


@dataclass
class GasFlatConfig:
    regime: str = "post_crisis"
    custom_eur_mwh: float = 40.0


@dataclass
class GasTrajectoryConfig:
    segments: list = field(default_factory=list)
    ramp_hours: int = 720
    noise_std: float = 3.0
    noise_ar: float = 0.99
    noise_floor: float = 10.0


@dataclass
class GasConfig:
    mode: str  # "flat" | "trajectory"
    flat: GasFlatConfig
    trajectory: GasTrajectoryConfig


@dataclass
class PathsConfig:
    fitted_models_dir: Path
    data_dir: Path
    output_dir: Path


@dataclass
class Config:
    horizon_hours: int
    n_paths: int
    random_seed: int
    block_size: int
    start_date: str
    wind: WindConfig
    solar: SolarConfig
    gas: GasConfig
    paths: PathsConfig
    bootstrap_window_days: int = 14

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)

        gen = raw["generator"]
        wind = raw["wind"]
        solar = raw["solar"]
        gas = raw["gas"]
        paths = raw["paths"]

        for knob, scale in (("wind.scale", wind["scale"]), ("solar.scale", solar["scale"])):
            if scale > IN_DOMAIN_SCALE_MAX:
                print(
                    f"WARNING: {knob}={scale} is above {IN_DOMAIN_SCALE_MAX}x -- the "
                    f"shipped price model is in-domain only up to ~{IN_DOMAIN_SCALE_MAX}x. "
                    f"Aggregate statistics are usable with caution, but individual "
                    f"hours are extrapolated and should not be trusted (solar has a "
                    f"confirmed non-monotonic price response, trough near 3x). "
                    f"See README 'Known limitations'."
                )

        return cls(
            horizon_hours=int(gen["horizon_hours"]),
            n_paths=int(gen["n_paths"]),
            random_seed=int(gen["random_seed"]),
            block_size=int(gen["block_size"]),
            start_date=str(gen.get("start_date", "2024-01-01")),
            wind=WindConfig(
                scale=float(wind["scale"]),
                onshore_capacity_mw=float(wind["onshore_capacity_mw"]),
                offshore_capacity_mw=float(wind["offshore_capacity_mw"]),
            ),
            solar=SolarConfig(
                scale=float(solar["scale"]),
                capacity_mw=float(solar["capacity_mw"]),
            ),
            gas=GasConfig(
                mode=gas["mode"],
                flat=GasFlatConfig(**gas.get("flat", {})),
                trajectory=GasTrajectoryConfig(**gas.get("trajectory", {})),
            ),
            paths=PathsConfig(
                fitted_models_dir=Path(paths["fitted_models_dir"]),
                data_dir=Path(paths["data_dir"]),
                output_dir=Path(paths["output_dir"]),
            ),
            bootstrap_window_days=int(gen.get("bootstrap_window_days", 14)),
        )
