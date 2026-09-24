"""Scenario configuration: the YAML file is the generator's interface.

`Config.from_yaml` reads a file like ``config/example.yaml`` into the dataclasses
below, which are threaded explicitly through `generator.run.generate` (no module
globals). The dataclasses can also be built directly in Python, as
``examples/run_example.py`` does.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

#: Typical hub height of a new offshore turbine (~15 MW class), m. Used when the config
#: doesn't set wind.hub_height_m; set it to your turbine's, or null to drop the column.
DEFAULT_HUB_HEIGHT_M = 150.0

#: Shipped price model is in-domain only up to this wind.scale / solar.scale.
IN_DOMAIN_SCALE_MAX = 1.25


@dataclass
class WindConfig:
    """Wind capacity and hub-height settings.

    Attributes
    ----------
    scale : float
        Multiplier on the installed capacity baseline (onshore and offshore alike).
        Moves ``wind_generation_MW`` and, through ``wind_load_ratio``, the price.
        Never touches the simulated wind speed.
    onshore_capacity_mw, offshore_capacity_mw : float
        Installed capacity baseline in MW (``config/example.yaml``: DK1 as of
        2026-06-01, Energinet).
    hub_height_m : float or None, default 150
        Height in m of the ``wind_speed_hub_ms`` output column; ``None`` drops the
        column. Set it to your turbine's hub height.

    Warnings
    --------
    The price model is in-domain only up to ``scale`` ~1.25; see README
    "Known limitations".
    """

    scale: float
    onshore_capacity_mw: float
    offshore_capacity_mw: float
    hub_height_m: float | None = DEFAULT_HUB_HEIGHT_M   # wind_speed_hub_ms height; None = no column


@dataclass
class SiteConfig:
    """A wind-farm site to output wind for, besides the system wind.

    Attributes
    ----------
    name : str
        A site with a fitted model in ``data/sites/<name>.json``; shipped: ``thor``,
        ``ringkobing``, ``herning`` (see
        ``config/sites.yaml`` and ``scripts/build_site_model.py``).
    hub_height_m : float, default 150
        Hub height at the site, m.

    Notes
    -----
    Adds ``site_wind_speed_hub_ms``: the wind at this site, consistent hour by hour
    with the simulated system (Horns Rev) wind that drives the prices. Feed this, not
    ``wind_speed_ms``/``wind_speed_hub_ms``, to a wind-farm model evaluating a farm at
    this site. Uses its own random stream, so enabling it changes no other column.
    """

    name: str
    hub_height_m: float = DEFAULT_HUB_HEIGHT_M


@dataclass
class SolarConfig:
    """Solar capacity settings.

    Attributes
    ----------
    scale : float
        Multiplier on ``capacity_mw``. Moves ``solar_generation_MW`` and, through
        ``solar_load_ratio``, the price.
    capacity_mw : float
        Installed solar capacity baseline in MW.

    Warnings
    --------
    The price model is in-domain only up to ``scale`` ~1.25, and simulated solar
    is deterministic (no cloud variability); see README "Known limitations".
    """

    scale: float
    capacity_mw: float


@dataclass
class GasFlatConfig:
    """Constant gas price for the whole horizon (``gas.mode = "flat"``).

    Attributes
    ----------
    regime : {"pre_crisis", "crisis", "post_crisis", "custom"}
        Named historical level (`generator.gas.GAS_LEVELS`) or ``"custom"``.
    custom_eur_mwh : float
        Gas price in EUR/MWh, used only when ``regime == "custom"``.
    """

    regime: str = "post_crisis"
    custom_eur_mwh: float = 40.0


@dataclass
class GasTrajectoryConfig:
    """Multi-regime gas schedule (``gas.mode = "trajectory"``).

    Attributes
    ----------
    segments : list of dict
        ``{"regime": str, "years": float}`` entries walked in order (1 year = 8760 h).
        Segments past the horizon are ignored; a schedule shorter than the horizon
        holds its last level.
    ramp_hours : int
        Length of the linear ramp between two regime levels, in hours.
    noise_std : float
        Stationary standard deviation of the AR(1) noise, EUR/MWh.
    noise_ar : float
        AR(1) coefficient of the noise (0.99: slow, week-scale wander).
    noise_floor : float
        Hard lower bound on the gas price, EUR/MWh.
    """

    segments: list = field(default_factory=list)
    ramp_hours: int = 720
    noise_std: float = 3.0
    noise_ar: float = 0.99
    noise_floor: float = 10.0


@dataclass
class GasConfig:
    """Gas price settings: ``mode`` selects which of ``flat``/``trajectory`` applies.

    Notes
    -----
    Gas is a schedule, not a market model: it doesn't react to anything else in
    the simulation. Trajectory mode draws from the shared random stream (so it
    changes every later draw, see `Config`) and weakens the wind-price
    correlation compared with a flat level.
    """

    mode: str  # "flat" | "trajectory"
    flat: GasFlatConfig
    trajectory: GasTrajectoryConfig


@dataclass
class PathsConfig:
    """Where the generator reads models/data and writes scenario CSVs.

    Attributes
    ----------
    fitted_models_dir : Path
        Directory with the fitted models (the repo's ``models/``).
    data_dir : Path
        Directory with the historical data slices (the repo's ``data/``).
    output_dir : Path
        Directory `generator.run.save_scenarios` writes to (created if missing).
    """

    fitted_models_dir: Path
    data_dir: Path
    output_dir: Path


@dataclass
class Config:
    """Complete scenario configuration.

    Attributes
    ----------
    horizon_hours : int
        Simulation length in hours (8760 = 1 year).
    n_paths : int
        Number of stochastic paths; each is ``horizon_hours`` long.
    random_seed : int
        Seed of the single random stream the whole run draws from (see Notes).
    block_size : int
        Load/net-position bootstrap block length in hours (168 = one week).
    start_date : str
        First timestamp, local time (Europe/Copenhagen), e.g. ``"2024-01-01"``.
        Sets the calendar every model conditions on (season, hour of day).
    wind, solar, gas, paths
        See `WindConfig`, `SolarConfig`, `GasConfig`, `PathsConfig`.
    bootstrap_window_days : int, default 14
        Bootstrap blocks start at the same local hour and within this many days of
        the calendar date they fill.
    site : SiteConfig or None, default None
        Farm site to output ``site_wind_speed_hub_ms`` for (`SiteConfig`).

    Notes
    -----
    One ``numpy.random.Generator`` is created from ``random_seed`` and consumed in
    a fixed order: the gas trajectory (trajectory mode only), then per path the
    wind speed, onshore and offshore capacity factors, the load/net-position
    bootstrap, and the price draw. Consequences:

    * Changing ``wind.scale``, ``solar.scale`` or ``wind.hub_height_m`` uses exactly
      the same random draws, so runs differ only through the changed setting.
    * Changing ``gas.mode``, ``horizon_hours`` or the bootstrap settings shifts
      every later draw, so all stochastic outputs change.
    * Path ``i`` doesn't depend on ``n_paths``; the gas trajectory is drawn once
      and shared by all paths.
    * Solar consumes no random draws (it's deterministic).
    * The farm-site wind (``site``) draws from a separate stream seeded from
      ``(random_seed, 1)``, so adding or changing it leaves every other column
      unchanged.
    * Results can differ slightly between CPU and GPU.
    """

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
    site: SiteConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Read a scenario YAML file (see ``config/example.yaml``).

        Parameters
        ----------
        path : str or Path
            YAML file with ``generator``, ``wind``, ``solar``, ``gas`` and
            ``paths`` sections, and optionally ``site``. Optional keys: ``generator.start_date``
            (default ``"2024-01-01"``), ``generator.bootstrap_window_days`` (14),
            ``wind.hub_height_m`` (150; ``null`` = no hub-height column).

        Returns
        -------
        Config

        Warns
        -----
        Prints a warning when ``wind.scale`` or ``solar.scale`` exceeds
        `IN_DOMAIN_SCALE_MAX`.
        """
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
                    f"price model is in-domain only up to ~{IN_DOMAIN_SCALE_MAX}x. "
                    f"Aggregate statistics are usable with caution, but individual "
                    f"hours are extrapolated and should not be trusted; beyond ~2x results "
                    f"are outside the validated region, and beyond ~3x the price response "
                    f"saturates and capture rates stop falling (extrapolation, not "
                    f"physics). See README 'Known limitations'."
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
                hub_height_m=(
                    None if wind.get("hub_height_m", DEFAULT_HUB_HEIGHT_M) is None
                    else float(wind.get("hub_height_m", DEFAULT_HUB_HEIGHT_M))
                ),
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
            site=(
                SiteConfig(name=str(raw["site"]["name"]),
                           hub_height_m=float(raw["site"].get("hub_height_m", DEFAULT_HUB_HEIGHT_M)))
                if raw.get("site") else None
            ),
        )
