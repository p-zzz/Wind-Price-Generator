"""
DK1 wind capacity factor ARMA models -- onshore and offshore fit separately.

Same pipeline as models/wind/arma_horns_rev.py, applied to capacity factor
(gen_wind_{site}_MW / DK1 installed {site} capacity MW, monthly capacity
forward-filled to hourly -- see analysis/wind_capacity_factor_exploration.py)
instead of raw wind speed.

Pipeline (per site)
--------------------
1. OLS seasonal mean (month + hour dummies) on capacity factor
2. Deseasonalise -> residuals
3. Monthly std scaling -> standardised residuals (unit variance per month)
4. ARMA(p, q) with Gaussian innovations; order selected by AIC; seasonal AR at lag 24
5. Simulate: Gaussian draws from fitted ARMA -> unstandardise -> add seasonal mean -> clip [0, 1]
6. Validation plots saved to analysis/outputs/DK1/wind_model/
7. Fitted objects saved to models/wind/fitted/wind_cf_{site}_arma.pkl

Onshore and offshore are modelled independently throughout -- their capacity
factor series have different persistence structure (offshore retains a
lag-40-70h ACF hump that onshore doesn't), matching the ACF check in
analysis/wind_capacity_factor_exploration.py.
"""

from pathlib import Path
import pickle
import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

plt.style.use("seaborn-v0_8")

RAW_GEN_PATH  = Path("DATA/raw/generation/dk1_generation_by_type_wide.parquet")
CAPACITY_PATH = Path("DATA/raw/capacity_per_municipality/dk1_wind_capacity_monthly.parquet")
ERA5_PATH     = Path("DATA/processed/era5_sites_wind_hourly.parquet")
OUTDIR      = Path("analysis/outputs/DK1/wind_model")
WIND_OUTDIR = Path("analysis/outputs/DK1/wind")
FITTED_DIR  = Path("models/wind/fitted")
OUTDIR.mkdir(parents=True, exist_ok=True)
WIND_OUTDIR.mkdir(parents=True, exist_ok=True)
FITTED_DIR.mkdir(parents=True, exist_ok=True)

SITES = {
    "onshore": ("gen_wind_onshore_MW", "OnshoreWindCapacity"),
    "offshore": ("gen_wind_offshore_MW", "OffshoreWindCapacity"),
}
WIND_SPEED_COL = "horns_rev_wind_speed_10m_ms"
MODEL_VERSION = "wspeed_v1"
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


# ------ Data loading ------

def load_hourly_generation() -> pd.DataFrame:
    gen = pd.read_parquet(RAW_GEN_PATH)
    gen = gen.tz_convert("Europe/Copenhagen").sort_index()
    cols = [g for g, _ in SITES.values()]
    gen_h = gen[cols].resample("h").mean()
    full_idx = pd.date_range(gen_h.index.min(), gen_h.index.max(), freq="h", tz="Europe/Copenhagen")
    return gen_h.reindex(full_idx)


def load_hourly_capacity() -> pd.DataFrame:
    cap_m = pd.read_parquet(CAPACITY_PATH)
    if cap_m.index.tz is None:
        cap_m.index = cap_m.index.tz_localize("Europe/Copenhagen")
    else:
        cap_m.index = cap_m.index.tz_convert("Europe/Copenhagen")
    full_idx = pd.date_range(cap_m.index.min(), cap_m.index.max() + pd.DateOffset(months=1) - pd.Timedelta(hours=1),
                              freq="h", tz="Europe/Copenhagen")
    return cap_m.reindex(full_idx, method="ffill")


def load_hourly_wind_speed() -> pd.Series:
    era5 = pd.read_parquet(ERA5_PATH, columns=[WIND_SPEED_COL])
    era5 = era5.tz_convert("Europe/Copenhagen").sort_index()
    return era5[WIND_SPEED_COL]


def build_cf_df() -> pd.DataFrame:
    gen_h = load_hourly_generation()
    cap_h = load_hourly_capacity()
    wind_speed = load_hourly_wind_speed()
    common_idx = gen_h.index.intersection(cap_h.index).intersection(wind_speed.index)
    cf = pd.DataFrame(index=common_idx)
    cf["month"] = common_idx.month
    cf["hour"]  = common_idx.hour
    cf[WIND_SPEED_COL] = wind_speed.loc[common_idx]
    for site, (gen_col, cap_col) in SITES.items():
        g = gen_h.loc[common_idx, gen_col]
        c = cap_h.loc[common_idx, cap_col]
        cf[site] = (g / c).clip(lower=0.0, upper=1.0)
    return cf


# ------ Post-hoc seasonal quantile mapper (same approach as wind_transformer.py) ------

class WindCFQuantileMapper:
    """
    Post-hoc seasonal empirical quantile mapping for marginal calibration,
    same 4-season x 4-hour-block (6h each) binning as WindQuantileMapper in
    wind_transformer.py. Uses the observed data's own empirical order
    statistics per bin (interpolated), not a fitted parametric family --
    a parametric Weibull-per-bin version was tried first and rejected:
    offshore wind capacity factor is genuinely bimodal (a calm-period mode
    near CF~0.05-0.1 and a rated-power saturation plateau near CF~0.85-0.9,
    from many offshore turbines hitting rated output simultaneously during
    basin-wide high-wind periods) and Weibull is unimodal by construction --
    no parameterization of it can reproduce two humps. Empirical quantile
    mapping has no such constraint since it maps directly onto the observed
    shape itself.
    """

    _SEASONS = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
    }
    _HOUR_BLOCKS = {
        "night":     list(range(0,  6)),
        "morning":   list(range(6,  12)),
        "afternoon": list(range(12, 18)),
        "evening":   list(range(18, 24)),
    }

    def fit(self, observed: pd.Series):
        self.sorted_ = {}
        for season, months in self._SEASONS.items():
            for block, hours in self._HOUR_BLOCKS.items():
                mask = observed.index.month.isin(months) & observed.index.hour.isin(hours)
                vals = np.clip(observed[mask].values, 0.0, 1.0)
                sorted_vals = np.sort(vals)
                self.sorted_[(season, block)] = sorted_vals
                print(f"  Empirical ({season:6s}, {block:9s}): n={mask.sum()}  "
                      f"min={sorted_vals.min():.3f}  median={np.median(sorted_vals):.3f}  max={sorted_vals.max():.3f}")
        return self

    def transform(self, sim: pd.Series) -> pd.Series:
        out = sim.values.copy().astype(np.float64)
        for season, months in self._SEASONS.items():
            for block, hours in self._HOUR_BLOCKS.items():
                mask = sim.index.month.isin(months) & sim.index.hour.isin(hours)
                if not mask.any():
                    continue
                vals = out[mask]
                n = len(vals)
                ranks = scipy.stats.rankdata(vals).astype(np.float64)
                probs = (ranks - 0.5) / n

                target_sorted = self.sorted_[(season, block)]
                m = len(target_sorted)
                target_probs = (np.arange(1, m + 1) - 0.5) / m
                corrected = np.interp(probs, target_probs, target_sorted)
                out[mask] = np.clip(corrected, 0.0, 1.0)
        return pd.Series(out, index=sim.index, name=sim.name)


# ------ Stage 1: seasonal mean via OLS (+ Horns Rev wind speed) ------
#
# Horns Rev wind speed is a strong proxy for DK1-wide wind conditions
# (Spearman ~0.86 with wind_total_MW in the real overlap data) -- including
# it here ties the CF simulation to the already-simulated wind_speed_ms in
# the generator, instead of two independently-drawn stochastic processes
# with no relationship to each other.

def fit_seasonal_mean(df: pd.DataFrame, site: str):
    formula = f"{site} ~ {WIND_SPEED_COL} + C(month) + C(hour)"
    model = smf.ols(formula=formula, data=df).fit()
    print(f"[{site}] Seasonal mean R2 (+ wind speed): {model.rsquared:.4f}")
    return model


# ------ Stage 2: monthly std scaling ------

def compute_monthly_std(residuals: pd.Series) -> dict:
    monthly_std = {}
    for m in range(1, 13):
        mask = residuals.index.month == m
        std = residuals[mask].std()
        monthly_std[m] = std if std > 0 else 1.0
    return monthly_std


def standardise(residuals: pd.Series, monthly_std: dict) -> pd.Series:
    divisor = residuals.index.month.map(monthly_std)
    return residuals / divisor.values


def unstandardise(std_resid: pd.Series, monthly_std: dict) -> pd.Series:
    multiplier = std_resid.index.month.map(monthly_std)
    return std_resid * multiplier.values


# ------ Stage 3: ARMA order selection ------

def select_arma_order(
    series: np.ndarray,
    p_range: range = range(1, 5),
    q_range: range = range(0, 3),
) -> tuple[int, int, float]:
    best_aic = np.inf
    best_p, best_q = 1, 0
    results = []

    search_series = series[:10_000]
    for p, q in itertools.product(p_range, q_range):
        try:
            fit = ARIMA(search_series, order=(p, 0, q)).fit()
            results.append((p, q, fit.aic))
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_p, best_q = p, q
        except Exception:
            pass

    print("\nARMA grid search (AIC):")
    for p, q, aic in sorted(results, key=lambda x: x[2])[:8]:
        marker = " <-- best" if (p == best_p and q == best_q) else ""
        print(f"  ARMA({p},{q})  AIC={aic:.2f}{marker}")

    return best_p, best_q, best_aic


def fit_arma(series: np.ndarray, p: int, q: int):
    seasonal_order = (1, 0, 0, 24)
    try:
        model = ARIMA(series, order=(p, 0, q), seasonal_order=seasonal_order).fit()
        print(f"\nFitted ARMA({p},{q}) + SAR(24)  AIC={model.aic:.2f}")
    except Exception as e:
        print(f"SAR(24) failed ({e}), fitting ARMA({p},{q}) without seasonal term.")
        model = ARIMA(series, order=(p, 0, q)).fit()
    return model


# ------ Stage 4: Simulation ------

def simulate_cf(df: pd.DataFrame, site: str, mean_model, monthly_std: dict, arma_model, n_sim: int = 1,
                 mapper: "WindCFQuantileMapper | None" = None) -> pd.DataFrame:
    n = len(df)
    seasonal_mean = mean_model.predict(df)
    sims = {}

    for i in range(n_sim):
        burn_in = 200
        sim_resid = arma_model.simulate(nsimulations=n + burn_in, random_state=rng.integers(int(1e9)))
        sim_resid = sim_resid[burn_in:]

        sim_std_series = pd.Series(sim_resid, index=df.index)
        resid = unstandardise(sim_std_series, monthly_std)

        cf_sim = seasonal_mean.values + resid.values
        cf_sim = np.clip(cf_sim, 0.0, 1.0)
        cf_series = pd.Series(cf_sim, index=df.index)
        if mapper is not None:
            cf_series = mapper.transform(cf_series)

        if i == 0:
            print(f"[simulate_cf:{site}] seasonal_mean mean    = {seasonal_mean.mean():.4f}")
            print(f"[simulate_cf:{site}] sim_0: ARMA innov std = {np.std(sim_resid):.4f}")
            print(f"[simulate_cf:{site}] sim_0: cf_sim mean    = {cf_series.mean():.4f}")

        sims[f"sim_{i}"] = cf_series.values

    return pd.DataFrame(sims, index=df.index)


# ------ Validation plots ------

def plot_qq(arma_resid: np.ndarray, site: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    stats.probplot(arma_resid, dist="norm", plot=ax)
    ax.set_title(f"QQ plot: ARMA residuals vs Gaussian ({site})")
    path = OUTDIR / f"wind_cf_{site}_qq_arma_resid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_acf_resid(arma_resid: np.ndarray, site: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(arma_resid, lags=72, ax=ax, alpha=0.05)
    ax.set_title(f"ACF of ARMA residuals ({site}, lags 0-72h)")
    ax.set_xlabel("Lag (hours)")
    path = OUTDIR / f"wind_cf_{site}_acf_arma_resid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_acf_squared_resid(arma_resid: np.ndarray, site: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(arma_resid ** 2, lags=72, ax=ax, alpha=0.05)
    ax.set_title(f"ACF of squared ARMA residuals ({site}) -- volatility clustering")
    ax.set_xlabel("Lag (hours)")
    path = OUTDIR / f"wind_cf_{site}_acf_squared_resid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_kde_overlay(observed: pd.Series, sim_df: pd.DataFrame, site: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    observed.plot.kde(ax=ax, label="Observed", linewidth=2, color="black")
    for col in sim_df.columns[:20]:
        sim_df[col].plot.kde(ax=ax, alpha=0.3, color="steelblue", linewidth=0.8, label="_nolegend_")
    ax.plot([], [], color="steelblue", alpha=0.5, linewidth=0.8, label="Simulated")
    ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel("Capacity factor")
    ax.set_title(f"Marginal distribution: observed vs simulated ({site})")
    ax.legend()
    path = OUTDIR / f"wind_cf_{site}_kde_marginal.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_monthly_stats(observed: pd.Series, sim_df: pd.DataFrame, site: str):
    obs_monthly_mean = observed.groupby(observed.index.month).mean()
    obs_monthly_std  = observed.groupby(observed.index.month).std()

    sim_means, sim_stds = [], []
    for col in sim_df.columns:
        s = sim_df[col]
        sim_means.append(s.groupby(s.index.month).mean())
        sim_stds.append(s.groupby(s.index.month).std())

    sim_mean_arr = pd.concat(sim_means, axis=1)
    sim_std_arr  = pd.concat(sim_stds, axis=1)

    months = np.arange(1, 13)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.fill_between(months, sim_mean_arr.quantile(0.10, axis=1), sim_mean_arr.quantile(0.90, axis=1),
                     alpha=0.3, color="steelblue", label="Sim 10-90%")
    ax.plot(months, sim_mean_arr.mean(axis=1), color="steelblue", label="Sim mean")
    ax.plot(months, obs_monthly_mean.values, color="black", linewidth=2, label="Observed")
    ax.set_title(f"Monthly mean CF ({site})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Capacity factor")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.fill_between(months, sim_std_arr.quantile(0.10, axis=1), sim_std_arr.quantile(0.90, axis=1),
                     alpha=0.3, color="coral", label="Sim 10-90%")
    ax.plot(months, sim_std_arr.mean(axis=1), color="coral", label="Sim std")
    ax.plot(months, obs_monthly_std.values, color="black", linewidth=2, label="Observed")
    ax.set_title(f"Monthly std CF ({site})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Capacity factor")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = OUTDIR / f"wind_cf_{site}_monthly_stats.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_hourly_mean(observed: pd.Series, sim_df: pd.DataFrame, site: str):
    WINTER = [11, 12, 1, 2]
    SUMMER = [5, 6, 7, 8]
    hours  = np.arange(24)

    def diurnal(series, months=None):
        if months is not None:
            series = series[series.index.month.isin(months)]
        return series.groupby(series.index.hour).mean()

    def sim_diurnal(months=None):
        rows = []
        for col in sim_df.columns:
            s = sim_df[col]
            rows.append(diurnal(s, months))
        return pd.concat(rows, axis=1)

    season_specs = [
        ("Annual", None,   "black",     "dimgray"),
        ("Winter", WINTER, "steelblue", "steelblue"),
        ("Summer", SUMMER, "tomato",    "tomato"),
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    for label, months, obs_col, sim_col in season_specs:
        obs_vals  = diurnal(observed, months).values
        sim_table = sim_diurnal(months)
        ax.plot(hours, obs_vals,               color=obs_col, lw=2,   ls="-",  label=f"Obs {label}")
        ax.plot(hours, sim_table.mean(axis=1), color=sim_col, lw=1.5, ls="--", label=f"Sim {label}")

    ax.set_title(f"Diurnal cycle: mean CF by hour ({site})")
    ax.set_xlabel("Hour of day (Copenhagen time)")
    ax.set_ylabel("Capacity factor")
    ax.legend(fontsize=8)
    path = OUTDIR / f"wind_cf_{site}_diurnal_mean.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_arma_acf_comparison(observed: pd.Series, site: str, mean_model, monthly_std: dict, arma_model,
                              wind_speed_pool: pd.Series, mapper: "WindCFQuantileMapper | None" = None):
    obs_acf = sm_acf(observed.values, nlags=72, fft=True)

    n_paths, sim_hours, burn_in = 25, 43800, 200
    lrng = np.random.default_rng(42)
    # Real wind speed slice (own real index) so wind speed and calendar stay
    # naturally aligned -- a fabricated calendar index would pair arbitrary
    # wind speed values with the wrong month/hour.
    synth_idx = wind_speed_pool.index[-sim_hours:]
    synth_df = pd.DataFrame({
        site: 0.0,
        "month": synth_idx.month,
        "hour": synth_idx.hour,
        WIND_SPEED_COL: wind_speed_pool.loc[synth_idx].values,
    }, index=synth_idx)
    seasonal_mean = mean_model.predict(synth_df)

    sim_acfs = []
    for i in range(n_paths):
        sim_resid = arma_model.simulate(nsimulations=sim_hours + burn_in, random_state=int(lrng.integers(int(1e9))))[burn_in:]
        resid = unstandardise(pd.Series(sim_resid, index=synth_df.index), monthly_std)
        cf_sim = np.clip(seasonal_mean.values + resid.values, 0.0, 1.0)
        cf_series = pd.Series(cf_sim, index=synth_df.index)
        if mapper is not None:
            cf_series = mapper.transform(cf_series)
        cf_sim = cf_series.values
        sim_acfs.append(sm_acf(cf_sim, nlags=72, fft=True))
        print(f"  [{site}] ARMA path {i:2d}  mean={cf_sim.mean():.4f}  std={cf_sim.std():.4f}")

    sim_acfs = np.array(sim_acfs)
    sim_mean, sim_std = sim_acfs.mean(axis=0), sim_acfs.std(axis=0)
    lags = np.arange(73)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(lags, sim_mean - sim_std, sim_mean + sim_std, color="steelblue", alpha=0.3,
                     label=f"Simulated mean ± 1 std (n={n_paths})")
    ax.plot(lags, sim_mean, color="steelblue", lw=1.5)
    ax.plot(lags, obs_acf, color="black", lw=2, label="Observed")
    ax.axhline(0, color="grey", lw=0.7, linestyle="--")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ACF")
    ax.set_title(f"ACF 0-72h: observed vs ARMA -- {site}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = WIND_OUTDIR / f"wind_cf_{site}_arma_acf_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_arma_qq(observed: pd.Series, site: str, mean_model, monthly_std: dict, arma_model,
                  wind_speed_pool: pd.Series, mapper: "WindCFQuantileMapper | None" = None):
    n_paths, sim_hours, burn_in = 25, 43800, 200
    lrng = np.random.default_rng(42)
    synth_idx = wind_speed_pool.index[-sim_hours:]
    synth_df = pd.DataFrame({
        site: 0.0,
        "month": synth_idx.month,
        "hour": synth_idx.hour,
        WIND_SPEED_COL: wind_speed_pool.loc[synth_idx].values,
    }, index=synth_idx)
    seasonal_mean = mean_model.predict(synth_df)

    all_sim = []
    for i in range(n_paths):
        sim_resid = arma_model.simulate(nsimulations=sim_hours + burn_in, random_state=int(lrng.integers(int(1e9))))[burn_in:]
        resid = unstandardise(pd.Series(sim_resid, index=synth_df.index), monthly_std)
        cf_sim = np.clip(seasonal_mean.values + resid.values, 0.0, 1.0)
        cf_series = pd.Series(cf_sim, index=synth_df.index)
        if mapper is not None:
            cf_series = mapper.transform(cf_series)
        all_sim.append(cf_series.values)

    pooled = np.concatenate(all_sim)
    percentiles = np.arange(1, 100)
    obs_q = np.percentile(observed.values, percentiles)
    sim_q = np.percentile(pooled, percentiles)
    lo, hi = min(obs_q.min(), sim_q.min()), max(obs_q.max(), sim_q.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(obs_q, sim_q, s=20, color="steelblue", zorder=3)
    ax.plot([lo, hi], [lo, hi], color="red", lw=1.5, label="45° line")
    ax.set_xlabel("Observed quantiles")
    ax.set_ylabel("Simulated quantiles")
    ax.set_title(f"QQ plot -- {site} capacity factor: observed vs ARMA")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = WIND_OUTDIR / f"wind_cf_{site}_arma_qq.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ------ Per-site pipeline ------

def fit_site(df: pd.DataFrame, site: str):
    print("\n" + "=" * 60)
    print(f"SITE: {site}")
    print("=" * 60)

    observed = df[site].dropna()
    site_df = df.loc[observed.index, [site, "month", "hour", WIND_SPEED_COL]]
    print(f"Loaded {len(site_df)} hourly observations ({site_df.index[0]} .. {site_df.index[-1]})")

    pkl_path = FITTED_DIR / f"wind_cf_{site}_arma.pkl"
    arma_resid_clean = None
    reuse = pkl_path.exists()
    if reuse:
        with open(pkl_path, "rb") as f:
            prev = pickle.load(f)
        reuse = prev.get("model_version") == MODEL_VERSION
        if not reuse:
            print(f"\n-- Existing fitted ARMA is model_version={prev.get('model_version')!r}, "
                  f"need {MODEL_VERSION!r} -- forcing full refit -- {site}")

    if reuse:
        print(f"\n-- Reusing existing fitted ARMA ({pkl_path}) -- skipping order selection/refit -- {site}")
        mean_model  = prev["mean_model"]
        monthly_std = prev["monthly_std"]
        arma_model  = prev["arma_model"]
        best_p, best_q = prev["arma_order"]
        seasonal_mean = mean_model.predict(site_df)
        arma_resid = arma_model.resid
        arma_resid_clean = arma_resid[~np.isnan(arma_resid)]
    else:
        print(f"\n-- Fitting seasonal mean (OLS: wind speed + month + hour dummies) -- {site}")
        mean_model = fit_seasonal_mean(site_df, site)
        seasonal_mean = mean_model.predict(site_df)
        residuals = observed - seasonal_mean
        print(f"Residual std: {residuals.std():.4f}")

        print(f"\n-- Computing monthly std -- {site}")
        monthly_std = compute_monthly_std(residuals)
        for m, s in monthly_std.items():
            print(f"  Month {m:2d}: std = {s:.4f}")
        std_resid = standardise(residuals, monthly_std)
        print(f"Standardised residual std: {std_resid.std():.4f} (target ~1)")

        print(f"\n-- ARMA order selection -- {site}")
        best_p, best_q, _ = select_arma_order(std_resid.values)

        print(f"\n-- Fitting best ARMA with SAR(24) -- {site}")
        arma_model = fit_arma(std_resid.values, best_p, best_q)
        arma_resid = arma_model.resid
        arma_resid_clean = arma_resid[~np.isnan(arma_resid)]

        arma_fitted_std = pd.Series(std_resid.values - arma_resid, index=site_df.index).fillna(0)
        arma_fitted_unstd = unstandardise(arma_fitted_std, monthly_std)
        full_fitted = seasonal_mean.values + arma_fitted_unstd.values

        obs_vals = observed.values
        ss_tot = np.sum((obs_vals - obs_vals.mean()) ** 2)
        ss_res_full = np.sum((obs_vals - full_fitted) ** 2)
        ss_res_mean = np.sum((obs_vals - seasonal_mean.values) ** 2)

        print(f"\n-- Model metrics -- {site}")
        print(f"  R² (full model)        : {1 - ss_res_full / ss_tot:.4f}")
        print(f"  R² (seasonal mean only): {1 - ss_res_mean / ss_tot:.4f}")
        print(f"  RMSE                   : {np.sqrt(np.mean((obs_vals - full_fitted)**2)):.4f}")
        print(f"  MAE                    : {np.mean(np.abs(obs_vals - full_fitted)):.4f}")
        print(f"  ARMA order             : ({best_p}, {best_q})")
        print(f"  ARMA + SAR(24) AIC     : {arma_model.aic:.2f}")
        print(f"  ARMA + SAR(24) BIC     : {arma_model.bic:.2f}")
        print(f"  ARMA log-likelihood    : {arma_model.llf:.2f}")

    print(f"\n-- Fitting WindCFQuantileMapper (16 bins) -- {site}")
    mapper = WindCFQuantileMapper().fit(observed)

    print(f"\n-- Simulating CF paths (N=50, quantile-mapped) -- {site}")
    sim_df = simulate_cf(site_df, site, mean_model, monthly_std, arma_model, n_sim=50, mapper=mapper)

    print(f"\n-- Generating validation plots -- {site}")
    if arma_resid_clean is not None:
        plot_qq(arma_resid_clean, site)
        plot_acf_resid(arma_resid_clean, site)
        plot_acf_squared_resid(arma_resid_clean, site)
    plot_kde_overlay(observed, sim_df, site)
    plot_monthly_stats(observed, sim_df, site)
    plot_hourly_mean(observed, sim_df, site)

    wind_speed_pool = df[WIND_SPEED_COL].dropna()

    print(f"\n-- ARMA ACF comparison (free-running paths, quantile-mapped) -- {site}")
    plot_arma_acf_comparison(observed, site, mean_model, monthly_std, arma_model,
                             wind_speed_pool, mapper=mapper)

    print(f"\n-- ARMA QQ plot (quantile-mapped) -- {site}")
    plot_arma_qq(observed, site, mean_model, monthly_std, arma_model,
                wind_speed_pool, mapper=mapper)

    fitted = {
        "mean_model":     mean_model,
        "monthly_std":    monthly_std,
        "arma_order":     (best_p, best_q),
        "arma_model":     arma_model,
        "quantile_mapper": mapper,
        "site":           site,
        "cf_col":         site,
        "model_version":  MODEL_VERSION,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(fitted, f)
    print(f"\nFitted objects saved to: {pkl_path}")


# ------ Main ------

def main():
    print("=" * 60)
    print("DK1 wind capacity factor ARMA -- onshore + offshore")
    print("=" * 60)

    df = build_cf_df()
    print(f"CF data window: {df.index.min()} .. {df.index.max()}  ({len(df)} hours)")

    for site in SITES:
        fit_site(df, site)

    print("\nDone.")


if __name__ == "__main__":
    main()
