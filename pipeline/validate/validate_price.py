"""
Validate synthetic price paths against observed data.
Reads from DATA/synthetic/{MODEL_NAME}/price_paths.parquet.
Generates 8 validation plots (1×3 regime panels) and summary tables.
No simulation logic.
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import gaussian_kde, spearmanr
from statsmodels.tsa.stattools import acf as sm_acf

plt.style.use("seaborn-v0_8")


# ------ Config ------

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "price_mdn_v11"
PRICE_COL = "day_ahead_price"
SIM_PATH = Path("DATA/synthetic") / MODEL_NAME / "price_paths.parquet"
PLOT_DIR = Path("analysis/outputs/DK1/price_model") / MODEL_NAME
PLOT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42

FEATURES_PATH = Path("DATA/processed/dk1_features_hourly.parquet")
ERA5_PATH = Path("DATA/processed/era5_sites_wind_hourly.parquet")
GAS_PATH = Path("DATA/raw/gas/Dutch TTF Natural Gas Futures Historical Data.csv")

WIND_COL = "horns_rev_wind_speed_10m_ms"
GAS_COL = "ttf_gas_price_eur_mwh"
GAS_FILL_BEFORE = pd.Timestamp("2018-03-01")
GAS_FILL_VALUE = 20.0

REGIMES = {
    "normal": (None, 2021),
    "crisis": (2022, 2022),
    "post_crisis": (2023, None),
}
REGIME_LABELS = {
    "normal": "Normal (≤2021)",
    "crisis": "Crisis (2022)",
    "post_crisis": "Post-crisis (≥2023)",
}
REGIME_COLORS = {
    "normal": "#1f77b4",
    "crisis": "#d62728",
    "post_crisis": "#2ca02c",
}

_SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
_SEASON_COLORS = {
    "DJF": "#1f77b4",
    "MAM": "#2ca02c",
    "JJA": "#d62728",
    "SON": "#ff7f0e",
}

METRIC_KEYS = [
    "mean (EUR/MWh)",
    "std (EUR/MWh)",
    "median (EUR/MWh)",
    "q05 (EUR/MWh)",
    "q95 (EUR/MWh)",
    "pct_neg (%)",
    "pct_above_200 (%)",
]


# ------ Data loading ------


def load_data():
    print("------ Loading observed prices ------")
    feat = pd.read_parquet(FEATURES_PATH, columns=[PRICE_COL])
    obs = feat.tz_convert("Europe/Copenhagen").sort_index()[PRICE_COL].dropna()
    print(f"  {len(obs)} obs  ({obs.index[0].date()} .. {obs.index[-1].date()})")

    print("------ Loading observed wind ------")
    era5 = pd.read_parquet(ERA5_PATH, columns=[WIND_COL])
    wind = era5.tz_convert("Europe/Copenhagen").sort_index()[WIND_COL].dropna()
    print(f"  {len(wind)} obs  ({wind.index[0].date()} .. {wind.index[-1].date()})")

    print("------ Loading simulated paths ------")
    sim_df = pd.read_parquet(SIM_PATH)
    if sim_df.index.tz is None:
        sim_df.index = pd.DatetimeIndex(sim_df.index).tz_localize("Europe/Copenhagen")
    else:
        sim_df.index = sim_df.index.tz_convert("Europe/Copenhagen")
    n_paths = len(sim_df.columns)
    print(
        f"  {n_paths} paths x {len(sim_df)} h  "
        f"({sim_df.index[0].date()} .. {sim_df.index[-1].date()})"
    )

    return obs, wind, sim_df


# ------ Helpers ------


def _regime_mask(idx, regime):
    lo, hi = REGIMES[regime]
    if lo is None:
        return idx.year <= hi
    if hi is None:
        return idx.year >= lo
    return (idx.year >= lo) & (idx.year <= hi)


def _get_regime(obs, sim_df, regime):
    obs_r = obs[_regime_mask(obs.index, regime)]
    sim_r = sim_df.loc[_regime_mask(sim_df.index, regime)]
    return obs_r, sim_r


def _pool_sim(sim_df):
    return np.concatenate([sim_df[c].values for c in sim_df.columns])


def _season_metrics(vals):
    return {
        "mean (EUR/MWh)": float(np.mean(vals)),
        "std (EUR/MWh)": float(np.std(vals)),
        "median (EUR/MWh)": float(np.median(vals)),
        "q05 (EUR/MWh)": float(np.percentile(vals, 5)),
        "q95 (EUR/MWh)": float(np.percentile(vals, 95)),
        "pct_neg (%)": float(100 * np.mean(vals < 0)),
        "pct_above_200 (%)": float(100 * np.mean(vals > 200)),
    }


# ------ Plot 1 — KDE marginal (1×3 per regime) ------


def plot_kde_marginal(obs, sim_df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, regime in zip(axes, REGIMES):
        obs_r, sim_r = _get_regime(obs, sim_df, regime)
        all_v = np.concatenate([obs_r.values, _pool_sim(sim_r)])
        x_lo = np.percentile(all_v, 0.5) - 5
        x_hi = np.percentile(all_v, 99.5) + 5
        if regime == "normal":
            x_hi = min(x_hi, 150.0)
        x = np.linspace(x_lo, x_hi, 500)
        ax.plot(x, gaussian_kde(obs_r.values)(x), color="black", lw=2, label="Observed")
        for c in list(sim_r.columns)[:20]:
            ax.plot(
                x,
                gaussian_kde(sim_r[c].values)(x),
                color="steelblue",
                alpha=0.3,
                lw=0.8,
            )
        ax.set_xlabel("Price (EUR/MWh)")
        ax.set_ylabel("Density")
        ax.set_title(REGIME_LABELS[regime])
        ax.legend(fontsize=8)
    # fig.suptitle(f"KDE marginal — {MODEL_NAME}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "kde_marginal.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "kde_marginal.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: kde_marginal.png / kde_marginal.pdf")


# ------ Plot 2 — Monthly mean price (1×3 per regime) ------


def plot_monthly_mean(obs, sim_df):
    months = np.arange(1, 13)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, regime in zip(axes, REGIMES):
        obs_r, sim_r = _get_regime(obs, sim_df, regime)
        color = REGIME_COLORS[regime]
        obs_mm = obs_r.groupby(obs_r.index.month).mean().reindex(months)
        sim_means = np.array(
            [
                sim_r[c].groupby(sim_r.index.month).mean().reindex(months).values
                for c in sim_r.columns
            ]
        )
        p10 = np.nanpercentile(sim_means, 10, axis=0)
        p90 = np.nanpercentile(sim_means, 90, axis=0)
        sm = np.nanmean(sim_means, axis=0)
        ax.fill_between(months, p10, p90, color=color, alpha=0.25, label="10-90th pct")
        ax.plot(months, sm, color=color, lw=1.5, label="Simulated mean")
        ax.plot(months, obs_mm.values, color="black", lw=2, label="Observed")
        ax.set_xlabel("Month")
        ax.set_ylabel("EUR/MWh")
        ax.set_xticks(months)
        ax.set_title(REGIME_LABELS[regime])
        ax.legend(fontsize=8)
    # fig.suptitle(f"Monthly mean price — {MODEL_NAME}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "monthly_mean.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "monthly_mean.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: monthly_mean.png / monthly_mean.pdf")


# ------ Plot 3 — ACF comparison, daily prices (1×3 per regime) ------


def plot_acf_comparison(obs, sim_df):
    lags = np.arange(61)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, regime in zip(axes, REGIMES):
        obs_r, sim_r = _get_regime(obs, sim_df, regime)
        color = REGIME_COLORS[regime]
        obs_daily = obs_r.resample("D").mean().dropna()
        obs_acf = sm_acf(obs_daily.values, nlags=60, fft=True)
        sim_acfs = np.array(
            [
                sm_acf(
                    sim_r[c].resample("D").mean().dropna().values, nlags=60, fft=True
                )
                for c in sim_r.columns
            ]
        )
        sim_mean = sim_acfs.mean(axis=0)
        sim_std = sim_acfs.std(axis=0)
        ax.plot(lags, obs_acf, color="black", lw=2, label="Observed")
        ax.plot(lags, sim_mean, color=color, lw=1.5, label="Simulated mean")
        ax.fill_between(
            lags,
            sim_mean - sim_std,
            sim_mean + sim_std,
            color=color,
            alpha=0.25,
            label="±1 std",
        )
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("ACF")
        ax.set_title(REGIME_LABELS[regime])
        ax.legend(fontsize=8)
    # fig.suptitle(f"ACF comparison (daily prices) — {MODEL_NAME}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "acf_comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "acf_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: acf_comparison.png / acf_comparison.pdf")


# ------ Plot 4 — Negative price frequency by month (1×3 per regime) ------


def plot_neg_price_by_month(obs, sim_df):
    months = np.arange(1, 13)
    month_abbr = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, regime in zip(axes, REGIMES):
        obs_r, sim_r = _get_regime(obs, sim_df, regime)
        color = REGIME_COLORS[regime]
        obs_neg = np.array(
            [
                float((obs_r.values[obs_r.index.month == m] < 0).mean())
                if (obs_r.index.month == m).any()
                else 0.0
                for m in months
            ]
        )
        sim_neg = np.array(
            [
                [
                    float((sim_r[c].values[sim_r.index.month == m] < 0).mean())
                    if (sim_r.index.month == m).any()
                    else 0.0
                    for m in months
                ]
                for c in sim_r.columns
            ]
        ).mean(axis=0)
        width = 0.35
        x = np.arange(12)
        ax.bar(
            x - width / 2,
            obs_neg,
            width=width,
            color="black",
            alpha=0.7,
            label="Observed",
        )
        ax.bar(
            x + width / 2,
            sim_neg,
            width=width,
            color=color,
            alpha=0.7,
            label="Simulated mean",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(month_abbr, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Fraction of hours < 0")
        ax.set_title(REGIME_LABELS[regime])
        ax.legend(fontsize=8)
    # fig.suptitle(f"Negative price frequency by month — {MODEL_NAME}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "neg_price_by_month.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "neg_price_by_month.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: neg_price_by_month.png / neg_price_by_month.pdf")


# ------ Plot 5 — QQ plot (1×3 per regime) ------


def plot_qq(obs, sim_df):
    pcts = np.linspace(1, 99, 200)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, regime in zip(axes, REGIMES):
        obs_r, sim_r = _get_regime(obs, sim_df, regime)
        color = REGIME_COLORS[regime]
        obs_q = np.percentile(obs_r.values, pcts)
        sim_q = np.percentile(_pool_sim(sim_r), pcts)
        lo = min(obs_q.min(), sim_q.min())
        hi = max(obs_q.max(), sim_q.max())
        ax.scatter(obs_q, sim_q, s=8, color=color, alpha=0.8)
        ax.plot([lo, hi], [lo, hi], color="red", lw=1.5, label="45°")
        ax.set_xlabel("Observed quantiles (EUR/MWh)")
        ax.set_ylabel("Simulated quantiles (EUR/MWh)")
        ax.set_title(REGIME_LABELS[regime])
        ax.legend(fontsize=8)
    # fig.suptitle(f"QQ plot — {MODEL_NAME}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "qq_plot.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "qq_plot.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: qq_plot.png / qq_plot.pdf")


# ------ Plot 6 — KDE per regime, mean ± 1 std band (1×3) ------


def plot_regime_comparison(obs, sim_df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, regime in zip(axes, REGIMES):
        obs_r, sim_r = _get_regime(obs, sim_df, regime)
        color = REGIME_COLORS[regime]
        all_v = np.concatenate([obs_r.values, _pool_sim(sim_r)])
        x_lo = np.percentile(all_v, 0.5) - 5
        x_hi = np.percentile(all_v, 99.5) + 5
        x = np.linspace(x_lo, x_hi, 500)
        path_kdes = np.array([gaussian_kde(sim_r[c].values)(x) for c in sim_r.columns])
        sim_mean = path_kdes.mean(axis=0)
        sim_std = path_kdes.std(axis=0)
        ax.fill_between(
            x,
            np.maximum(sim_mean - sim_std, 0),
            sim_mean + sim_std,
            color=color,
            alpha=0.3,
            label="Simulated mean ±1 std",
        )
        ax.plot(x, sim_mean, color=color, lw=1.5)
        ax.plot(x, gaussian_kde(obs_r.values)(x), color="black", lw=2, label="Observed")
        ax.set_xlabel("Price (EUR/MWh)")
        ax.set_ylabel("Density")
        ax.set_title(REGIME_LABELS[regime])
        ax.legend(fontsize=8)
    # fig.suptitle(f"KDE per regime — {MODEL_NAME}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "regime_comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "regime_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: regime_comparison.png / regime_comparison.pdf")


# ------ Plot 7 — Diurnal price profile by season (1×3 per regime) ------


def plot_diurnal_by_season(obs, sim_df):
    hours = np.arange(24)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, regime in zip(axes, REGIMES):
        obs_r, sim_r = _get_regime(obs, sim_df, regime)
        for sname, months in _SEASONS.items():
            color = _SEASON_COLORS[sname]
            obs_s = obs_r[obs_r.index.month.isin(months)]
            obs_d = obs_s.groupby(obs_s.index.hour).mean().reindex(hours).values
            sim_paths = np.array(
                [
                    (lambda s: s.groupby(s.index.hour).mean().reindex(hours).values)(
                        sim_r[c][sim_r.index.month.isin(months)]
                    )
                    for c in sim_r.columns
                ]
            )
            sim_d = np.nanmean(sim_paths, axis=0)
            ax.plot(hours, obs_d, color=color, lw=2, ls="-", label=f"{sname} obs")
            ax.plot(hours, sim_d, color=color, lw=1.5, ls="--")
        ax.set_xlabel("Hour of day (Copenhagen time)")
        ax.set_ylabel("Mean price (EUR/MWh)")
        ax.set_title(REGIME_LABELS[regime])
        ax.set_xticks(range(0, 24, 3))
        ax.legend(fontsize=7)
    # fig.suptitle(f"Diurnal price profile by season — {MODEL_NAME}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "diurnal_by_season.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "diurnal_by_season.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: diurnal_by_season.png / diurnal_by_season.pdf")


# ------ Plot 8 — Wind-price scatter by decile (1×3 per regime) ------


def plot_wind_price_scatter(obs, wind, sim_df):
    wind_sim = wind.reindex(sim_df.index)
    obs_sim = obs.reindex(sim_df.index)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, regime in zip(axes, REGIMES):
        ms = _regime_mask(sim_df.index, regime)
        w_r = wind_sim[ms].values
        o_r = obs_sim[ms].values
        sim_r = {c: sim_df[c][ms].values for c in sim_df.columns}
        color = REGIME_COLORS[regime]

        valid = ~np.isnan(w_r)
        w_r = w_r[valid]
        o_r = o_r[valid]
        sim_r = {c: v[valid] for c, v in sim_r.items()}

        bin_edges = np.percentile(w_r, np.arange(0, 110, 10)).astype(float)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        bin_mids = np.array(
            [
                w_r[(w_r >= bin_edges[i]) & (w_r < bin_edges[i + 1])].mean()
                for i in range(10)
            ]
        )
        obs_means = np.array(
            [
                o_r[(w_r >= bin_edges[i]) & (w_r < bin_edges[i + 1])].mean()
                for i in range(10)
            ]
        )
        sim_bin = np.array(
            [
                [
                    sim_r[c][(w_r >= bin_edges[i]) & (w_r < bin_edges[i + 1])].mean()
                    for i in range(10)
                ]
                for c in sim_df.columns
            ]
        )
        p10 = np.percentile(sim_bin, 10, axis=0)
        p90 = np.percentile(sim_bin, 90, axis=0)
        sm = sim_bin.mean(axis=0)

        deciles = np.arange(1, 11)
        ax.fill_between(deciles, p10, p90, color=color, alpha=0.25, label="10-90th pct")
        ax.plot(deciles, sm, color=color, lw=1.5, label="Simulated mean")
        ax.plot(
            deciles, obs_means, color="black", lw=2, marker="o", ms=4, label="Observed"
        )
        ax.set_xlabel("Wind speed decile")
        ax.set_ylabel("Mean price (EUR/MWh)")
        ax.set_title(REGIME_LABELS[regime])
        ax.set_xticks(deciles)
        ax.legend(fontsize=8)
    # fig.suptitle(f"Wind-price by decile — {MODEL_NAME}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "wind_price_scatter.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "wind_price_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: wind_price_scatter.png / wind_price_scatter.pdf")


# ------ Plot A: 1-week sample time series ------


def plot_sample_timeseries(obs, sim_df):
    rng = np.random.default_rng(RANDOM_SEED)
    common = obs.index.intersection(sim_df.index)

    def _pick_start(months, year_min=None):
        mask = common.month.isin(months)
        if year_min is not None:
            mask = mask & (common.year >= year_min)
        limit = common[-1] - pd.Timedelta(hours=167)
        candidates = common[mask & (common <= limit)]
        return candidates[rng.integers(len(candidates))]

    t0_djf = _pick_start([12, 1, 2])
    t0_jja = _pick_start([6, 7, 8], year_min=2023)
    panels = [("Winter (DJF)", t0_djf), ("Summer (JJA, post-crisis)", t0_jja)]
    paths_5 = list(sim_df.columns[:5])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, (title, t0) in zip(axes, panels):
        t1 = t0 + pd.Timedelta(hours=167)
        obs_w = obs.loc[t0:t1]
        ax.plot(
            obs_w.index, obs_w.values, color="black", lw=2, label="Observed", zorder=3
        )
        for i, col in enumerate(paths_5):
            sim_w = sim_df[col].loc[t0:t1]
            ax.plot(
                sim_w.index,
                sim_w.values,
                color="steelblue",
                alpha=0.5,
                lw=0.8,
                label="Simulated" if i == 0 else None,
            )
        ax.set_title(title)
        ax.set_ylabel("EUR/MWh")
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.legend(fontsize=8)

    # fig.suptitle(f"1-week sample time series — {MODEL_NAME}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "sample_timeseries.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "sample_timeseries.svg", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "sample_timeseries.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: sample_timeseries.png / sample_timeseries.svg / sample_timeseries.pdf")


# ------ Plot B: Power spectral density ------


def plot_psd_comparison(obs, sim_df):
    fs = 1.0
    nperseg = 8760

    freqs, pxx_obs = welch(obs.values, fs=fs, nperseg=nperseg)
    freqs = freqs[1:]
    pxx_obs = pxx_obs[1:]
    periods = 1.0 / freqs
    in_range = (periods >= 2.0) & (periods <= 8760.0)
    periods = periods[in_range]
    pxx_obs = pxx_obs[in_range]

    sim_psds = []
    for c in sim_df.columns:
        _, pxx_s = welch(sim_df[c].values, fs=fs, nperseg=nperseg)
        sim_psds.append(pxx_s[1:][in_range])
    sim_arr = np.array(sim_psds)
    sim_mean = sim_arr.mean(axis=0)
    sim_std = sim_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(
        periods,
        np.maximum(sim_mean - sim_std, 1e-12),
        sim_mean + sim_std,
        color="steelblue",
        alpha=0.4,
        label="Simulated mean ±1 std",
    )
    ax.plot(periods, sim_mean, color="#1f77b4", lw=2, label="Simulated mean")
    ax.plot(periods, pxx_obs, color="black", lw=1, label="Observed")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, 5000)
    ax.set_ylim(bottom=1)
    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("Power spectral density")
    # ax.set_title(f"Power spectral density — {MODEL_NAME}")
    ax.legend(fontsize=8)

    trans = ax.get_xaxis_transform()
    for period_val, lbl in [(24, "24h"), (168, "168h")]:
        ax.axvline(period_val, color="grey", ls="--", lw=1.0)
        ax.text(period_val * 1.05, 0.97, lbl, fontsize=8, va="top", transform=trans)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "psd_comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "psd_comparison.svg", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "psd_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: psd_comparison.png / psd_comparison.svg / psd_comparison.pdf")


# ------ Plot B (freq): Power spectral density — frequency x-axis ------


def plot_psd_comparison_freq(obs, sim_df):
    fs = 1.0
    nperseg = 8760

    freqs, pxx_obs = welch(obs.values, fs=fs, nperseg=nperseg)
    freqs = freqs[1:]
    pxx_obs = pxx_obs[1:]
    periods = 1.0 / freqs
    in_range = (periods >= 2.0) & (periods <= 8760.0)
    freqs = freqs[in_range]
    pxx_obs = pxx_obs[in_range]

    sim_psds = []
    for c in sim_df.columns:
        _, pxx_s = welch(sim_df[c].values, fs=fs, nperseg=nperseg)
        sim_psds.append(pxx_s[1:][in_range])
    sim_arr = np.array(sim_psds)
    sim_mean = sim_arr.mean(axis=0)
    sim_std = sim_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(
        freqs,
        np.maximum(sim_mean - sim_std, 1e-12),
        sim_mean + sim_std,
        color="steelblue",
        alpha=0.4,
        label="Simulated mean ±1 std",
    )
    ax.plot(freqs, sim_mean, color="#1f77b4", lw=2, label="Simulated mean")
    ax.plot(freqs, pxx_obs, color="black", lw=1, label="Observed")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1 / 5000, 1 / 10)
    ax.set_ylim(bottom=1)
    ax.set_xlabel("Frequency (cycles/hour)")
    ax.set_ylabel("Power spectral density")
    # ax.set_title(f"Power spectral density — {MODEL_NAME}")
    ax.legend(fontsize=8)

    trans = ax.get_xaxis_transform()
    for period_val, lbl in [(24, "24h"), (168, "168h")]:
        ax.axvline(1 / period_val, color="grey", ls="--", lw=1.0)
        ax.text(1 / period_val * 1.05, 0.97, lbl, fontsize=8, va="top", transform=trans)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "psd_comparison_freq.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "psd_comparison_freq.svg", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "psd_comparison_freq.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: psd_comparison_freq.png / psd_comparison_freq.svg / psd_comparison_freq.pdf")


# ------ Seasonal summary table ------
#
# Season x regime grid: pooling all years within a season blends pre-crisis
# winters (~EUR30-40) with the 2022 crisis winter (~EUR200+) with post-crisis
# winters (~EUR90), which isn't meaningful given how different those regimes
# are -- so every season is additionally split by regime (REGIMES/
# _regime_mask, same partition the plots already use).


def print_seasonal_summary(obs, sim_df):
    seasons = list(_SEASONS.keys())
    col_w = 10
    csv_rows = []

    for regime in REGIMES:
        obs_regime_mask = _regime_mask(obs.index, regime)
        sim_regime_mask = _regime_mask(sim_df.index, regime)

        obs_rows = {}
        sim_rows = {}
        for sname, months in _SEASONS.items():
            obs_mask = obs_regime_mask & obs.index.month.isin(months)
            obs_vals = obs[obs_mask].values
            obs_rows[sname] = (
                _season_metrics(obs_vals)
                if len(obs_vals)
                else {k: float("nan") for k in METRIC_KEYS}
            )

            sim_mask = sim_regime_mask & sim_df.index.month.isin(months)
            if sim_mask.sum() == 0:
                sim_rows[sname] = {k: float("nan") for k in METRIC_KEYS}
            else:
                path_metrics = [
                    _season_metrics(sim_df[c][sim_mask].values)
                    for c in sim_df.columns
                ]
                sim_rows[sname] = {
                    k: float(np.mean([p[k] for p in path_metrics])) for k in METRIC_KEYS
                }

        print(f"\n-- {REGIME_LABELS[regime]} --")
        header = f"{'Metric':<22}" + "".join(
            f"{s + '_obs':>{col_w}}  {s + '_sim':>{col_w}}  " for s in seasons
        )
        print(header)
        print("-" * len(header))
        for k in METRIC_KEYS:
            row = f"{k:<22}"
            for s in seasons:
                row += f"{obs_rows[s][k]:>{col_w}.3f}  {sim_rows[s][k]:>{col_w}.3f}  "
            print(row)

        for s in seasons:
            csv_rows.append(
                {
                    "Regime": REGIME_LABELS[regime],
                    "Season": s,
                    **{f"{k}_obs": obs_rows[s][k] for k in METRIC_KEYS},
                    **{f"{k}_sim": sim_rows[s][k] for k in METRIC_KEYS},
                }
            )

    df_csv = pd.DataFrame(csv_rows).set_index(["Regime", "Season"])
    df_csv.to_csv(PLOT_DIR / "seasonal_summary.csv")
    print(f"\n  Saved: seasonal_summary.csv")


# ------ Summary stats ------
#
# Price regimes differ sharply in both level and variability (crisis 2022
# mean/std several times pre-/post-crisis), so every metric here -- not just
# mean -- is broken out per regime (REGIMES/_regime_mask, same partition the
# plots already use), plus one pooled "full period" block for reference.


def _summary_metrics(vals, mj_vals, wind_vals):
    rho, _ = (
        spearmanr(wind_vals, vals, nan_policy="omit")
        if len(wind_vals) > 2
        else (float("nan"), None)
    )
    return {
        "mean (EUR/MWh)": float(np.mean(vals)) if len(vals) else float("nan"),
        "std (EUR/MWh)": float(np.std(vals)) if len(vals) else float("nan"),
        "pct_neg (%)": float(100 * np.mean(vals < 0)) if len(vals) else float("nan"),
        "pct_neg_mayjun (%)": float(100 * np.mean(mj_vals < 0))
        if len(mj_vals)
        else float("nan"),
        "spearman_wind_price": float(rho),
    }


def print_summary_stats(obs, wind, sim_df):
    idx_obs = obs.index
    mj_mask_obs = np.isin(idx_obs.month, [5, 6])

    regime_keys = ["full"] + list(REGIMES.keys())
    regime_labels = {"full": "Full period (pooled)", **REGIME_LABELS}
    col_w = 16

    for regime in regime_keys:
        obs_mask = (
            np.ones(len(idx_obs), dtype=bool)
            if regime == "full"
            else _regime_mask(idx_obs, regime)
        )
        obs_vals = obs.values[obs_mask]
        obs_mj = obs.values[obs_mask & mj_mask_obs]
        w_r = wind.reindex(idx_obs[obs_mask]).values
        obs_m = _summary_metrics(obs_vals, obs_mj, w_r)

        per_path = []
        for c in sim_df.columns:
            v = sim_df[c].values
            idx = sim_df.index
            sim_mask = (
                np.ones(len(idx), dtype=bool)
                if regime == "full"
                else _regime_mask(idx, regime)
            )
            v_r = v[sim_mask]
            v_mj = v[sim_mask & np.isin(idx.month, [5, 6])]
            w_p = wind.reindex(idx[sim_mask]).values
            per_path.append(_summary_metrics(v_r, v_mj, w_p))

        print(f"\n-- {regime_labels[regime]} --")
        header = f"{'Metric':<28}{'Observed':>{col_w}}{'Simulated mean':>{col_w}}"
        print(header)
        print("-" * len(header))
        for k in obs_m:
            sim_v = float(np.nanmean([p[k] for p in per_path]))
            print(f"{k:<28}{obs_m[k]:>{col_w}.3f}{sim_v:>{col_w}.3f}")


# ------ Main ------


def main():
    print("=" * 60)
    print(f"validate_price — MODEL_NAME={MODEL_NAME}")
    print("=" * 60)

    obs, wind, sim_df = load_data()

    print(f"\n------ Generating plots → {PLOT_DIR} ------")
    plot_kde_marginal(obs, sim_df)
    plot_monthly_mean(obs, sim_df)
    plot_acf_comparison(obs, sim_df)
    plot_neg_price_by_month(obs, sim_df)
    plot_qq(obs, sim_df)
    plot_regime_comparison(obs, sim_df)
    plot_diurnal_by_season(obs, sim_df)
    plot_wind_price_scatter(obs, wind, sim_df)
    plot_sample_timeseries(obs, sim_df)
    plot_psd_comparison(obs, sim_df)
    plot_psd_comparison_freq(obs, sim_df)

    print("\n------ Summary stats ------")
    print_summary_stats(obs, wind, sim_df)

    print("\n------ Seasonal summary ------")
    print_seasonal_summary(obs, sim_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
