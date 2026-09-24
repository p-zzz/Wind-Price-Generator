"""Joint DK1 scenario generator: entry point (``python -m generator.run``) and API.

Pipeline, per path::

  1. Wind Transformer + quantile mapper  -> wind_speed_ms (10 m, Horns Rev; unscaled)
                                            + wind_speed_hub_ms (at wind.hub_height_m)
  2. Wind CF ARMA (onshore + offshore)   -> wind_generation_MW (x wind.scale)
  3. Solar CF model (deterministic)      -> solar_generation_MW (x solar.scale)
  4. Paired block bootstrap              -> actual_load_MW, net_position_MW
  5. Gas price (flat or trajectory)      -> gas_price_eur_mwh (shared by all paths)
  6. Price MDN v11 (10-seed ensemble)    -> day_ahead_price (EUR/MWh)

Output: ``<paths.output_dir>/scenarios_<timestamp>.csv`` (see `save_scenarios`).
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from statsmodels.tsa.stattools import acf as sm_acf

from .config import Config
from .gas import build_gas_trajectory, flat_gas_price, gas_regime_labels
from .io import load_bootstrap_source, load_fitted_objects, load_shear_table, load_site_model, load_wind_speed_seed
from .bootstrap import paired_block_bootstrap
from .price import simulate_price_mdn
from .solar import simulate_solar_cf
from .wind import WIND_COL, hub_height_wind_speed, run_transformer_simulation, simulate_wind_generation, site_wind_speed

BURN_IN = 200


def get_device() -> torch.device:
    """CUDA if available, else CPU (printed). Results can differ slightly between them."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device


def build_sim_index(n_hours: int, start: str = "2024-01-01") -> pd.DataFrame:
    """Build the hourly simulation calendar every model conditions on.

    Parameters
    ----------
    n_hours : int
        Number of hours.
    start : str
        First timestamp in local time, e.g. ``"2024-01-01"``.

    Returns
    -------
    pandas.DataFrame
        Index: hourly, tz-aware Europe/Copenhagen. Columns ``hour``, ``month``,
        ``dayofweek`` (local).

    Notes
    -----
    Steps are real elapsed hours, so on DST days the local hour 02:00 is missing
    (March) or appears twice (October), as in the training data.
    """
    idx = pd.date_range(start=start, periods=n_hours, freq="h", tz="Europe/Copenhagen")
    df = pd.DataFrame(index=idx)
    df["hour"] = idx.hour
    df["month"] = idx.month
    df["dayofweek"] = idx.dayofweek
    return df


def _gas_price_for_run(cfg: Config, n_hours: int, rng: np.random.Generator):
    if cfg.gas.mode == "flat":
        price = flat_gas_price(cfg.gas.flat.regime, cfg.gas.flat.custom_eur_mwh)
        label = f"{price:.1f} EUR/MWh (flat, regime={cfg.gas.flat.regime})"
        return price, label
    price = build_gas_trajectory(
        n_hours,
        cfg.gas.trajectory.segments,
        cfg.gas.trajectory.ramp_hours,
        cfg.gas.trajectory.noise_std,
        cfg.gas.trajectory.noise_ar,
        cfg.gas.trajectory.noise_floor,
        rng,
    )
    label = f"trajectory  mean={price.mean():.1f}  min={price.min():.1f}  max={price.max():.1f} EUR/MWh"
    return price, label


def gas_regimes_for(cfg: Config) -> np.ndarray:
    """Gas regime label of every simulated hour, for `print_validation`.

    Parameters
    ----------
    cfg : Config

    Returns
    -------
    numpy.ndarray of str, shape (cfg.horizon_hours,)
        The flat regime name, or per-hour trajectory regimes with ramp hours
        between two different regimes labelled ``"transition"``.
    """
    if cfg.gas.mode == "flat":
        return np.full(cfg.horizon_hours, cfg.gas.flat.regime, dtype=object)
    return gas_regime_labels(
        cfg.horizon_hours, cfg.gas.trajectory.segments, cfg.gas.trajectory.ramp_hours
    )


def generate(cfg: Config, fitted: dict, device: torch.device) -> pd.DataFrame:
    """Simulate ``cfg.n_paths`` joint hourly scenarios.

    Parameters
    ----------
    cfg : Config
        Scenario settings (horizon, seed, scale knobs, gas, paths).
    fitted : dict
        Output of `generator.io.load_fitted_objects`.
    device : torch.device
        Device for the neural networks (`get_device`).

    Returns
    -------
    pandas.DataFrame
        Long format: ``n_paths * horizon_hours`` rows, indexed by hourly
        Europe/Copenhagen timestamps that **repeat once per path**, with a
        ``path`` column (0-based). Columns and units:

        * ``wind_speed_ms`` -- 10 m wind speed at Horns Rev, m/s (not hub height).
        * ``wind_speed_hub_ms`` -- the same wind at ``cfg.wind.hub_height_m``, m/s
          (absent if that is ``None``).
        * ``site_wind_speed_hub_ms`` -- wind at the farm site ``cfg.site`` and its hub
          height, m/s (only if ``cfg.site`` is set). **Use this for a farm's power**
          at that site; it moves with the system wind as the real weather does.
        * ``wind_generation_MW`` (= ``_onshore_MW`` + ``_offshore_MW``),
          ``solar_generation_MW`` -- DK1 fleet output at the scaled capacity, MW.
        * ``solar_load_ratio``, ``wind_load_ratio`` -- generation / load; the
          price model's renewable inputs.
        * ``actual_load_MW``, ``net_position_MW`` -- bootstrapped from 2023-2025
          history, MW; net position is export-positive.
        * ``gas_price_eur_mwh`` -- TTF gas, EUR/MWh.
        * ``day_ahead_price`` -- DK1 day-ahead price, EUR/MWh, one draw from the
          price model's conditional distribution. Not clipped to market limits.

    Notes
    -----
    Randomness comes from one stream seeded by ``cfg.random_seed``; see
    `generator.config.Config`
    for which settings keep the draws identical. Solar is deterministic, and the
    gas trajectory is shared by all paths.

    Warnings
    --------
    Beyond ~1.25x on ``wind.scale``/``solar.scale`` the price model extrapolates;
    see README "Known limitations".
    """
    transformer = fitted["transformer"]
    mapper = fitted["mapper"]
    wind_pkl = fitted["wind_pkl"]
    price_pkl = fitted["price"]
    mdn = fitted["mdn"]
    norm_stats = price_pkl["norm_stats"]
    feature_cols = fitted["price_feature_cols"]

    wind_capacity_baseline = {
        "onshore": cfg.wind.onshore_capacity_mw,
        "offshore": cfg.wind.offshore_capacity_mw,
    }

    df_idx = build_sim_index(cfg.horizon_hours, cfg.start_date)
    pool = load_bootstrap_source(cfg.paths.data_dir)
    shear = load_shear_table(cfg.paths.data_dir) if cfg.wind.hub_height_m else None
    site_model = load_site_model(cfg.paths.data_dir, cfg.site.name) if cfg.site else None
    site_rng = np.random.default_rng([cfg.random_seed, 1])   # separate stream: see Config
    hist_load = pool["actual_load_MW"].to_numpy()
    hist_netpos = pool["net_position_MW"].to_numpy()

    rng = np.random.default_rng(cfg.random_seed)
    solar_cf = simulate_solar_cf(df_idx, fitted["solar"])
    solar_mw = solar_cf * cfg.solar.capacity_mw * cfg.solar.scale
    gas_price, gas_label = _gas_price_for_run(cfg, cfg.horizon_hours, rng)
    print(f"  Gas price: {gas_label}")

    k_lags = wind_pkl["hparams"]["k_lags"]
    K_wind = wind_pkl["hparams"]["K"]
    norm_mean = wind_pkl["norm_stats"]["mean"]
    norm_std = wind_pkl["norm_stats"]["std"]

    records = []
    for path in range(cfg.n_paths):
        print(f"  Path {path + 1}/{cfg.n_paths}")

        buf = load_wind_speed_seed(cfg.paths.data_dir, k_lags, norm_mean, norm_std)
        sim_df = run_transformer_simulation(
            transformer, wind_pkl["norm_stats"], K_wind, mapper, buf,
            df_idx.index, BURN_IN, rng, device,
        )
        wind_speed = sim_df[WIND_COL].values

        gen_onshore, gen_offshore, gen_total, cf_onshore, cf_offshore = simulate_wind_generation(
            df_idx, fitted["wind_cf_onshore"], fitted["wind_cf_offshore"],
            wind_capacity_baseline, cfg.wind.scale, wind_speed, BURN_IN, rng,
        )

        load, net_pos = paired_block_bootstrap(
            hist_load, hist_netpos, pool.index, df_idx.index,
            cfg.block_size, cfg.bootstrap_window_days, rng,
        )

        solar_load_ratio = solar_mw / load
        wind_load_ratio = gen_total / load

        price = simulate_price_mdn(
            df_idx.index, wind_speed, solar_load_ratio, load, net_pos, gas_price,
            wind_load_ratio, mdn, norm_stats, rng, feature_cols, device,
        )

        hub_cols = (
            {"wind_speed_hub_ms": hub_height_wind_speed(wind_speed, df_idx.index, cfg.wind.hub_height_m, shear)}
            if cfg.wind.hub_height_m else {}
        )
        if site_model is not None:
            hub_cols["site_wind_speed_hub_ms"] = site_wind_speed(
                wind_speed, df_idx.index, site_model, cfg.site.hub_height_m, site_rng)
        records.append(
            pd.DataFrame(
                {
                    "path": path,
                    "wind_speed_ms": wind_speed,
                    **hub_cols,
                    "wind_generation_MW": gen_total,
                    "wind_generation_onshore_MW": gen_onshore,
                    "wind_generation_offshore_MW": gen_offshore,
                    "solar_generation_MW": solar_mw,
                    "solar_load_ratio": solar_load_ratio,
                    "wind_load_ratio": wind_load_ratio,
                    "actual_load_MW": load,
                    "net_position_MW": net_pos,
                    "gas_price_eur_mwh": gas_price if not np.isscalar(gas_price) else np.full(cfg.horizon_hours, gas_price),
                    "day_ahead_price": price,
                },
                index=df_idx.index,
            )
        )

    return pd.concat(records)


# ------ Validation ------


def _wind_metrics(wind: np.ndarray) -> dict:
    acf_vals = sm_acf(wind, nlags=48, fft=True)
    k, _, _ = stats.weibull_min.fit(wind, floc=0)
    return {
        "mean (m/s)": float(wind.mean()),
        "std (m/s)": float(wind.std()),
        "weibull_k": float(k),
        "acf_lag24": float(acf_vals[24]),
        "acf_lag48": float(acf_vals[48]),
    }


def _price_metrics(price: np.ndarray, idx: pd.DatetimeIndex) -> dict:
    mj_mask = np.isin(idx.month, [5, 6])
    pct_neg_mj = float(100 * (price[mj_mask] < 0).mean()) if mj_mask.any() else float("nan")
    return {
        "mean (EUR/MWh)": float(price.mean()),
        "std (EUR/MWh)": float(price.std()),
        "pct_neg (%)": float(100 * (price < 0).mean()),
        "pct_neg_mayjun (%)": pct_neg_mj,
    }


def print_validation(df_scenarios: pd.DataFrame, gas_regimes: np.ndarray | None = None) -> None:
    """Print summary statistics of generated scenarios.

    Wind speed (mean, std, Weibull shape, autocorrelation at 24/48 h), price (mean,
    std, % negative hours overall and in May-June) and Spearman(wind speed, price),
    each as mean +/- std across paths.

    Parameters
    ----------
    df_scenarios : pandas.DataFrame
        Output of `generate`.
    gas_regimes : numpy.ndarray, optional
        Per-hour regime labels (`gas_regimes_for`). When given, Spearman is also
        reported per regime, since pooling regimes with different price levels
        dilutes it.

    Notes
    -----
    Expect Spearman(wind speed, price) to be negative (the merit-order effect);
    its size depends on the inputs, so there is no single target value. Wind-speed
    statistics barely vary across paths: the quantile mapper enforces the
    climatological distribution over each path's horizon.
    """
    wind_m, price_m, rho_vals = [], [], []
    regime_order = list(dict.fromkeys(gas_regimes)) if gas_regimes is not None else []
    rho_by_regime = {r: [] for r in regime_order}
    for path in sorted(df_scenarios["path"].unique()):
        sub = df_scenarios[df_scenarios["path"] == path]
        wind = sub["wind_speed_ms"].values
        price = sub["day_ahead_price"].values
        wind_m.append(_wind_metrics(wind))
        price_m.append(_price_metrics(price, sub.index))
        rho, _ = stats.spearmanr(wind, price)
        rho_vals.append(float(rho))
        for regime in regime_order:
            mask = gas_regimes == regime
            rho, _ = stats.spearmanr(wind[mask], price[mask])
            rho_by_regime[regime].append(float(rho))

    def _row(key, rows):
        vals = [r[key] for r in rows]
        return f"  {key:<25} {np.mean(vals):>10.3f}  +/-{np.std(vals):.3f}"

    print("\n------ Wind speed ------")
    for key in wind_m[0]:
        print(_row(key, wind_m))
    print("\n------ Price ------")
    for key in price_m[0]:
        print(_row(key, price_m))
    print("\n------ Joint: Spearman(wind speed, price) ------")
    print(f"  {'all hours':<25} {np.mean(rho_vals):>10.3f}  +/-{np.std(rho_vals):.3f}")
    for regime in regime_order:
        vals = rho_by_regime[regime]
        label = f"gas={regime} ({int((gas_regimes == regime).sum())} h)"
        print(f"  {label:<25} {np.mean(vals):>10.3f}  +/-{np.std(vals):.3f}")


def save_scenarios(df_scenarios: pd.DataFrame, output_dir: Path) -> Path:
    """Write scenarios to ``output_dir/scenarios_<YYYYmmdd_HHMMSS>.csv``.

    Parameters
    ----------
    df_scenarios : pandas.DataFrame
        Output of `generate`.
    output_dir : Path
        Created if missing.

    Returns
    -------
    Path
        The CSV written. Its first column, ``timestamp``, holds the local
        timestamps with their UTC offset (e.g. ``2024-03-31 03:00:00+02:00``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"scenarios_{ts}.csv"
    df_scenarios.reset_index(names="timestamp").to_csv(path, index=False)
    print(f"Saved: {path}  ({len(df_scenarios):,} rows)")
    return path


def main() -> None:
    """Command line: ``python -m generator.run --config config/example.yaml``.

    Loads the config and models, generates, prints `print_validation` and saves
    the CSV (`save_scenarios`).
    """
    parser = argparse.ArgumentParser(description="DK1 joint wind/price scenario generator")
    parser.add_argument("--config", type=str, default="config/example.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = get_device()

    print("=" * 60)
    print("DK1 joint scenario generator")
    print(f"  horizon_hours={cfg.horizon_hours}  n_paths={cfg.n_paths}  seed={cfg.random_seed}")
    print(f"  wind.scale={cfg.wind.scale}  solar.scale={cfg.solar.scale}  gas.mode={cfg.gas.mode}")
    if cfg.site:
        print(f"  site={cfg.site.name} (hub {cfg.site.hub_height_m:g} m) -> site_wind_speed_hub_ms")
    print("=" * 60)

    print("\n------ Loading fitted objects ------")
    fitted = load_fitted_objects(cfg.paths.fitted_models_dir, device)

    print(f"\n------ Generating {cfg.n_paths} path(s) ------")
    df_scenarios = generate(cfg, fitted, device)

    print("\n------ Validation ------")
    print_validation(df_scenarios, gas_regimes_for(cfg))

    print("\n------ Saving scenarios ------")
    save_scenarios(df_scenarios, cfg.paths.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
