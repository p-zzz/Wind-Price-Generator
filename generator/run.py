"""
Joint scenario generator -- Transformer + capacity-factor ARMA + solar CF +
paired block bootstrap + gas schedule -> Price MDN v11.

Pipeline:
  1. Wind Transformer + quantile mapper       -> wind_speed_ms (feeds price model, unscaled)
  2. Wind CF ARMA (onshore + offshore)        -> wind_generation_MW (x wind.scale)
  3. Solar CF model                           -> solar_generation_MW (x solar.scale)
  4. Paired block bootstrap                   -> actual_load_MW, net_position_MW
  5. Gas price (flat or trajectory)           -> gas_price_eur_mwh
  6. Price MDN v11                            -> day_ahead_price (EUR/MWh)

Output: CSV saved to <paths.output_dir>/scenarios_<timestamp>.csv
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
from .io import load_bootstrap_source, load_fitted_objects, load_wind_speed_seed
from .bootstrap import paired_block_bootstrap
from .price import simulate_price_mdn
from .solar import simulate_solar_cf
from .wind import WIND_COL, run_transformer_simulation, simulate_wind_generation

BURN_IN = 200


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device


def build_sim_index(n_hours: int, start: str = "2024-01-01") -> pd.DataFrame:
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
    """Per-hour gas regime label for cfg's horizon (for print_validation)."""
    if cfg.gas.mode == "flat":
        return np.full(cfg.horizon_hours, cfg.gas.flat.regime, dtype=object)
    return gas_regime_labels(
        cfg.horizon_hours, cfg.gas.trajectory.segments, cfg.gas.trajectory.ramp_hours
    )


def generate(cfg: Config, fitted: dict, device: torch.device) -> pd.DataFrame:
    transformer = fitted["transformer"]
    mapper = fitted["mapper"]
    wind_pkl = fitted["wind_pkl"]
    price_pkl = fitted["price"]
    mdn = fitted["mdn"]
    norm_stats = price_pkl["norm_stats"]
    K = price_pkl["hparams"]["K"]
    feature_cols = fitted["price_feature_cols"]

    wind_capacity_baseline = {
        "onshore": cfg.wind.onshore_capacity_mw,
        "offshore": cfg.wind.offshore_capacity_mw,
    }

    df_idx = build_sim_index(cfg.horizon_hours, cfg.start_date)
    pool = load_bootstrap_source(cfg.paths.data_dir)
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
            wind_load_ratio, mdn, norm_stats, K, rng, feature_cols, device,
        )

        records.append(
            pd.DataFrame(
                {
                    "path": path,
                    "wind_speed_ms": wind_speed,
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
    """gas_regimes: optional per-hour regime labels (gas_regimes_for(cfg)), shared
    by every path. When given, Spearman(wind speed, price) is also reported per
    regime, since pooling regimes with different price levels dilutes it."""
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
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"scenarios_{ts}.csv"
    df_scenarios.reset_index(names="timestamp").to_csv(path, index=False)
    print(f"Saved: {path}  ({len(df_scenarios):,} rows)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="DK1 joint wind/price scenario generator")
    parser.add_argument("--config", type=str, default="config/example.yaml")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = get_device()

    print("=" * 60)
    print("DK1 joint scenario generator")
    print(f"  horizon_hours={cfg.horizon_hours}  n_paths={cfg.n_paths}  seed={cfg.random_seed}")
    print(f"  wind.scale={cfg.wind.scale}  solar.scale={cfg.solar.scale}  gas.mode={cfg.gas.mode}")
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
