"""
Validate synthetic wind paths against observed ERA5 data.
Reads from DATA/synthetic/{MODEL_NAME}/wind_paths.parquet.
Generates 8 validation plots to analysis/outputs/DK1/wind_model/{MODEL_NAME}/.
No simulation logic.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import acf as sm_acf
import matplotlib.dates as mdates
from scipy.signal import welch

plt.style.use("seaborn-v0_8")


# ------ Config ------

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "transformer_r3"
WIND_COL   = "horns_rev_wind_speed_10m_ms"
ERA5_PATH  = Path("DATA/processed/era5_sites_wind_hourly.parquet")
SIM_PATH   = Path("DATA/synthetic") / MODEL_NAME / "wind_paths.parquet"
PLOT_DIR   = Path("analysis/outputs/DK1/wind_model") / MODEL_NAME
PLOT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42

_SEASONS = {
    "Winter (DJF)": [12, 1, 2],
    "Spring (MAM)": [3, 4, 5],
    "Summer (JJA)": [6, 7, 8],
    "Autumn (SON)": [9, 10, 11],
}

_HOUR_BLOCKS = [
    ("00-03h", list(range(0,  4))),
    ("04-07h", list(range(4,  8))),
    ("08-11h", list(range(8,  12))),
    ("12-15h", list(range(12, 16))),
    ("16-19h", list(range(16, 20))),
    ("20-23h", list(range(20, 24))),
]


# ------ Helpers ------

def _acf_array(series, nlags=72):
    return sm_acf(series.values, nlags=nlags, fft=True)


def _spell_lengths(arr, threshold=12.0):
    spells, length = [], 0
    for v in arr:
        if v >= threshold:
            length += 1
        else:
            if length > 0:
                spells.append(length)
            length = 0
    if length > 0:
        spells.append(length)
    return np.array(spells, dtype=int)


def _weibull_k(arr):
    shape, _, _ = scipy.stats.weibull_min.fit(np.clip(arr, 1e-3, None), floc=0)
    return float(shape)


def _monthly_spearman(series):
    monthly_mean = series.groupby(series.index.month).mean()
    rho, _ = scipy.stats.spearmanr(monthly_mean.index, monthly_mean.values)
    return float(rho)


def _season_mask(index, months):
    return index.month.isin(months)


def _hour_block_mask(index, hours):
    return index.hour.isin(hours)


# ------ Data loading ------

def load_data():
    print("------ Loading observed data ------")
    obs_df = pd.read_parquet(ERA5_PATH, columns=[WIND_COL])
    obs    = obs_df.tz_convert("Europe/Copenhagen")[WIND_COL].dropna()
    print(f"  Observed: {len(obs)} obs  ({obs.index[0].date()} .. {obs.index[-1].date()})")

    print("------ Loading simulated paths ------")
    sim_df = pd.read_parquet(SIM_PATH)
    sim_df.index = pd.DatetimeIndex(sim_df.index).tz_localize("Europe/Copenhagen") \
        if sim_df.index.tz is None else sim_df.index.tz_convert("Europe/Copenhagen")
    n_paths = len(sim_df.columns)
    print(f"  Simulated: {n_paths} paths x {len(sim_df)} h  ({sim_df.index[0].date()} .. {sim_df.index[-1].date()})")
    return obs, sim_df


# ------ Plot 1 — ACF comparison ------

def plot_acf_comparison(obs, sim_df):
    lags = np.arange(73)
    obs_acf = _acf_array(obs, nlags=72)

    sim_acfs = np.array([_acf_array(sim_df[c], nlags=72) for c in sim_df.columns])
    sim_mean = sim_acfs.mean(axis=0)
    sim_std  = sim_acfs.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lags, obs_acf, color="black", lw=2, label="Observed")
    ax.plot(lags, sim_mean, color="steelblue", lw=1.5, label="Simulated mean")
    ax.fill_between(lags, sim_mean - sim_std, sim_mean + sim_std,
                    color="steelblue", alpha=0.25, label="±1 std")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ACF")
    # ax.set_title(f"ACF comparison — {MODEL_NAME}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "acf_comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "acf_comparison.svg", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "acf_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: acf_comparison.png / acf_comparison.svg / acf_comparison.pdf")


# ------ Plot 2 — KDE marginal overlay ------

def plot_kde_marginal(obs, sim_df):
    x = np.linspace(-1, obs.max() + 2, 400)

    fig, ax = plt.subplots(figsize=(8, 4))
    kde_obs = gaussian_kde(obs.values)
    ax.plot(x, kde_obs(x), color="black", lw=2, label="Observed")

    for c in list(sim_df.columns)[:20]:
        kde_s = gaussian_kde(sim_df[c].values)
        ax.plot(x, kde_s(x), color="steelblue", alpha=0.3, lw=0.8)

    ax.set_xlim(left=-1)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Density")
    # ax.set_title(f"KDE marginal — {MODEL_NAME}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "kde_marginal.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "kde_marginal.svg", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "kde_marginal.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: kde_marginal.png / kde_marginal.svg / kde_marginal.pdf")


# ------ Plot 3 — Monthly mean and std ------

def plot_monthly_stats(obs, sim_df):
    months = np.arange(1, 13)
    obs_monthly_mean = obs.groupby(obs.index.month).mean().reindex(months)
    obs_monthly_std  = obs.groupby(obs.index.month).std().reindex(months)

    sim_means = np.array([
        sim_df[c].groupby(sim_df.index.month).mean().reindex(months).values
        for c in sim_df.columns
    ])
    sim_stds = np.array([
        sim_df[c].groupby(sim_df.index.month).std().reindex(months).values
        for c in sim_df.columns
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, obs_stat, sim_stat, ylabel in zip(
        axes,
        [obs_monthly_mean, obs_monthly_std],
        [sim_means, sim_stds],
        ["Mean (m/s)", "Std (m/s)"],
    ):
        p10 = np.percentile(sim_stat, 10, axis=0)
        p90 = np.percentile(sim_stat, 90, axis=0)
        sim_m = sim_stat.mean(axis=0)
        ax.fill_between(months, p10, p90, color="steelblue", alpha=0.25, label="10-90th pct")
        ax.plot(months, sim_m, color="steelblue", lw=1.5, label="Simulated mean")
        ax.plot(months, obs_stat.values, color="black", lw=2, label="Observed")
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        ax.set_xticks(months)
        ax.legend(fontsize=8)

    # axes[0].set_title(f"Monthly mean — {MODEL_NAME}")
    # axes[1].set_title(f"Monthly std — {MODEL_NAME}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "monthly_stats.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "monthly_stats.svg", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "monthly_stats.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: monthly_stats.png / monthly_stats.svg / monthly_stats.pdf")


# ------ Plot 4 — Diurnal cycle ------

def plot_diurnal_mean(obs, sim_df):
    hours = np.arange(24)

    def _diurnal(series, mask=None):
        s = series[mask] if mask is not None else series
        return s.groupby(s.index.hour).mean().reindex(hours).values

    season_defs = [
        ("Annual",       None),
        ("Winter",       obs.index.month.isin([11, 12, 1, 2])),
        ("Summer",       obs.index.month.isin([5, 6, 7, 8])),
    ]
    sim_masks = [
        None,
        sim_df.index.month.isin([11, 12, 1, 2]),
        sim_df.index.month.isin([5, 6, 7, 8]),
    ]
    colors = ["steelblue", "darkorange", "forestgreen"]

    fig, ax = plt.subplots(figsize=(9, 4))
    for (label, obs_mask), sim_mask, color in zip(season_defs, sim_masks, colors):
        obs_d = _diurnal(obs, obs_mask)
        sim_d = np.array([
            _diurnal(sim_df[c], sim_df[c][sim_mask].index if sim_mask is not None else None)
            for c in sim_df.columns
        ]).mean(axis=0)

        if sim_mask is not None:
            sim_vals = [
                sim_df[c][sim_mask].groupby(sim_df[c][sim_mask].index.hour).mean()
                .reindex(hours).values
                for c in sim_df.columns
            ]
            sim_d = np.nanmean(sim_vals, axis=0)
        else:
            sim_vals = [
                sim_df[c].groupby(sim_df.index.hour).mean().reindex(hours).values
                for c in sim_df.columns
            ]
            sim_d = np.nanmean(sim_vals, axis=0)

        ax.plot(hours, obs_d, color=color, lw=2, ls="-",  label=f"{label} obs")
        ax.plot(hours, sim_d, color=color, lw=1.5, ls="--", label=f"{label} sim")

    ax.set_xlabel("Hour of day (Copenhagen time)")
    ax.set_ylabel("m/s")
    # ax.set_title(f"Diurnal cycle — {MODEL_NAME}")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "diurnal_mean.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "diurnal_mean.svg", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "diurnal_mean.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: diurnal_mean.png / diurnal_mean.svg / diurnal_mean.pdf")


# ------ Plot 5 — Storm spell distribution ------

def plot_storm_spell(obs, sim_df):
    bins = np.arange(0, 121, 4)

    obs_spells  = _spell_lengths(obs.values)
    obs_counts, _ = np.histogram(obs_spells, bins=bins)
    obs_rate    = obs_counts / (len(obs) / 1000)

    sim_rates = []
    for c in sim_df.columns:
        sp = _spell_lengths(sim_df[c].values)
        counts, _ = np.histogram(sp, bins=bins)
        sim_rates.append(counts / (len(sim_df) / 1000))
    sim_mean_rate = np.array(sim_rates).mean(axis=0)

    centres = 0.5 * (bins[:-1] + bins[1:])
    width   = (bins[1] - bins[0]) * 0.4

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(centres - width / 2, obs_rate,      width=width, color="black",     alpha=0.7, label="Observed")
    ax.bar(centres + width / 2, sim_mean_rate, width=width, color="steelblue", alpha=0.7, label="Simulated mean")
    ax.set_yscale("log")
    ax.set_xlim(0, 120)
    ax.set_xlabel("Spell length (consecutive hours above 12 m/s)")
    ax.set_ylabel("Spells per 1000 hours (log scale)")
    # ax.set_title(f"Storm spell distribution — {MODEL_NAME}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "storm_spell.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "storm_spell.svg", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "storm_spell.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: storm_spell.png / storm_spell.svg / storm_spell.pdf")


# ------ Plot 6 — QQ plot ------

def plot_qq(obs, sim_df):
    pcts = np.linspace(1, 99, 200)
    obs_q = np.percentile(obs.values, pcts)
    sim_pool = np.concatenate([sim_df[c].values for c in sim_df.columns])
    sim_q = np.percentile(sim_pool, pcts)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(obs_q, sim_q, s=8, color="steelblue", alpha=0.7)
    lo = min(obs_q.min(), sim_q.min())
    hi = max(obs_q.max(), sim_q.max())
    ax.plot([lo, hi], [lo, hi], color="red", lw=1.5, label="45°")
    ax.set_xlabel("Observed quantiles (m/s)")
    ax.set_ylabel("Simulated quantiles (m/s)")
    # ax.set_title(f"QQ plot — {MODEL_NAME}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "qq_plot.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "qq_plot.svg", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "qq_plot.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: qq_plot.png / qq_plot.svg / qq_plot.pdf")


# ------ Plot 7 — Seasonal x hour block marginals ------

def plot_seasonal_hour_marginals(obs, sim_df):
    season_names = list(_SEASONS.keys())
    n_seasons    = len(season_names)
    n_blocks     = len(_HOUR_BLOCKS)

    fig, axes = plt.subplots(n_seasons, n_blocks, figsize=(18, 12), sharex=False, sharey=False)
    x_grid = np.linspace(0, 25, 300)

    for r, sname in enumerate(season_names):
        s_months = _SEASONS[sname]
        obs_s_mask = _season_mask(obs.index, s_months)

        for c_idx, (blabel, bhours) in enumerate(_HOUR_BLOCKS):
            ax = axes[r][c_idx]
            obs_h_mask  = _hour_block_mask(obs.index, bhours)
            obs_vals    = obs[obs_s_mask & obs_h_mask].values

            sim_vals = np.concatenate([
                sim_df[col][
                    _season_mask(sim_df.index, s_months) &
                    _hour_block_mask(sim_df.index, bhours)
                ].values
                for col in sim_df.columns
            ])

            if len(obs_vals) > 2:
                kde_obs = gaussian_kde(obs_vals)
                ax.plot(x_grid, kde_obs(x_grid), color="black", lw=1.2)
            if len(sim_vals) > 2:
                kde_sim = gaussian_kde(sim_vals)
                ax.plot(x_grid, kde_sim(x_grid), color="steelblue", lw=1.2)

            short_season = sname.split()[0]
            ax.set_title(f"{short_season} / {blabel}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

    # fig.suptitle(
    #     f"Seasonal × hour-block marginals — {MODEL_NAME}\n"
    #     "black=observed, steelblue=simulated (pooled paths)",
    #     fontsize=10,
    # )
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "seasonal_hour_marginals.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "seasonal_hour_marginals.svg", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "seasonal_hour_marginals.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: seasonal_hour_marginals.png / seasonal_hour_marginals.svg / seasonal_hour_marginals.pdf")


# ------ Plot 8 — Per-season ACF ------

def plot_seasonal_acf(obs, sim_df):
    season_names = list(_SEASONS.keys())
    lags = np.arange(73)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, sname in enumerate(season_names):
        ax = axes[i]
        s_months = _SEASONS[sname]

        obs_sub  = obs[_season_mask(obs.index, s_months)]
        obs_acf  = sm_acf(obs_sub.values, nlags=72, fft=True)

        sim_acfs = []
        for col in sim_df.columns:
            sub = sim_df[col][_season_mask(sim_df.index, s_months)]
            sim_acfs.append(sm_acf(sub.values, nlags=72, fft=True))
        sim_acfs = np.array(sim_acfs)
        sim_mean = sim_acfs.mean(axis=0)
        sim_std  = sim_acfs.std(axis=0)

        ax.plot(lags, obs_acf, color="black", lw=2, label="Observed")
        ax.plot(lags, sim_mean, color="steelblue", lw=1.5, label="Simulated mean")
        ax.fill_between(lags, sim_mean - sim_std, sim_mean + sim_std,
                        color="steelblue", alpha=0.25, label="±1 std")
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_title(sname)
        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("ACF")
        ax.legend(fontsize=8)

    # fig.suptitle(f"Per-season ACF — {MODEL_NAME}", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "seasonal_acf.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "seasonal_acf.svg", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "seasonal_acf.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: seasonal_acf.png / seasonal_acf.svg / seasonal_acf.pdf")


# ------ Plot A: 1-week sample time series ------

def plot_sample_timeseries(obs, sim_df):
    rng    = np.random.default_rng(RANDOM_SEED)
    common = obs.index.intersection(sim_df.index)

    def _pick_start(months):
        mask       = common.month.isin(months)
        limit      = common[-1] - pd.Timedelta(hours=167)
        candidates = common[mask & (common <= limit)]
        return candidates[rng.integers(len(candidates))]

    t0_djf  = _pick_start([12, 1, 2])
    t0_jja  = _pick_start([6, 7, 8])
    panels  = [("Winter (DJF)", t0_djf), ("Summer (JJA)", t0_jja)]
    paths_5 = list(sim_df.columns[:5])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, (title, t0) in zip(axes, panels):
        t1    = t0 + pd.Timedelta(hours=167)
        obs_w = obs.loc[t0:t1]
        ax.plot(obs_w.index, obs_w.values, color="black", lw=2, label="Observed", zorder=3)
        for i, col in enumerate(paths_5):
            sim_w = sim_df[col].loc[t0:t1]
            ax.plot(sim_w.index, sim_w.values, color="steelblue", alpha=0.5, lw=0.8,
                    label="Simulated" if i == 0 else None)
        ax.set_title(title)
        ax.set_ylabel("m/s")
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.legend(fontsize=8)

    fig.suptitle(f"1-week sample time series — {MODEL_NAME}")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "sample_timeseries.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "sample_timeseries.svg", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: sample_timeseries.png / sample_timeseries.svg")


# ------ Plot B: Power spectral density ------

def plot_psd_comparison(obs, sim_df):
    fs      = 1.0
    nperseg = 8760

    freqs, pxx_obs = welch(obs.values, fs=fs, nperseg=nperseg)
    freqs    = freqs[1:]
    pxx_obs  = pxx_obs[1:]
    periods  = 1.0 / freqs
    in_range = (periods >= 2.0) & (periods <= 8760.0)
    periods  = periods[in_range]
    pxx_obs  = pxx_obs[in_range]

    sim_psds = []
    for c in sim_df.columns:
        _, pxx_s = welch(sim_df[c].values, fs=fs, nperseg=nperseg)
        sim_psds.append(pxx_s[1:][in_range])
    sim_arr  = np.array(sim_psds)
    sim_mean = sim_arr.mean(axis=0)
    sim_std  = sim_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(periods,
                    np.maximum(sim_mean - sim_std, 1e-12),
                    sim_mean + sim_std,
                    color="steelblue", alpha=0.4, label="Simulated mean ±1 std")
    ax.plot(periods, sim_mean, color="#1f77b4", lw=2, label="Simulated mean")
    ax.plot(periods, pxx_obs,  color="black",   lw=1, label="Observed")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2, 8760)
    ax.set_xlabel("Period (hours)")
    ax.set_ylabel("Power spectral density")
    ax.set_title(f"Power spectral density — {MODEL_NAME}")
    ax.legend(fontsize=8)

    trans = ax.get_xaxis_transform()
    for period_val, lbl in [(24, "24h"), (168, "168h")]:
        ax.axvline(period_val, color="grey", ls="--", lw=1.0)
        ax.text(period_val * 1.05, 0.97, lbl, fontsize=8, va="top", transform=trans)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "psd_comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "psd_comparison.svg", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: psd_comparison.png / psd_comparison.svg")


# ------ Plot B (freq): Power spectral density — frequency x-axis ------

def plot_psd_comparison_freq(obs, sim_df):
    fs      = 1.0
    nperseg = 8760

    freqs, pxx_obs = welch(obs.values, fs=fs, nperseg=nperseg)
    freqs    = freqs[1:]
    pxx_obs  = pxx_obs[1:]
    periods  = 1.0 / freqs
    in_range = (periods >= 2.0) & (periods <= 8760.0)
    freqs    = freqs[in_range]
    pxx_obs  = pxx_obs[in_range]

    sim_psds = []
    for c in sim_df.columns:
        _, pxx_s = welch(sim_df[c].values, fs=fs, nperseg=nperseg)
        sim_psds.append(pxx_s[1:][in_range])
    sim_arr  = np.array(sim_psds)
    sim_mean = sim_arr.mean(axis=0)
    sim_std  = sim_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(freqs,
                    np.maximum(sim_mean - sim_std, 1e-12),
                    sim_mean + sim_std,
                    color="steelblue", alpha=0.4, label="Simulated mean ±1 std")
    ax.plot(freqs, sim_mean, color="#1f77b4", lw=2, label="Simulated mean")
    ax.plot(freqs, pxx_obs,  color="black",   lw=1, label="Observed")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1/8760, 1/2)
    ax.set_xlabel("Frequency (cycles/hour)")
    ax.set_ylabel("Power spectral density")
    # ax.set_title(f"Power spectral density — {MODEL_NAME}")
    ax.legend(fontsize=8)

    trans = ax.get_xaxis_transform()
    for period_val, lbl in [(24, "24h"), (168, "168h")]:
        ax.axvline(1/period_val, color="grey", ls="--", lw=1.0)
        ax.text(1/period_val * 1.05, 0.97, lbl, fontsize=8, va="top", transform=trans)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "psd_comparison_freq.png", dpi=150, bbox_inches="tight")
    fig.savefig(PLOT_DIR / "psd_comparison_freq.svg", bbox_inches="tight")
    fig.savefig(PLOT_DIR / "psd_comparison_freq.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: psd_comparison_freq.png / psd_comparison_freq.svg / psd_comparison_freq.pdf")


# ------ Seasonal summary table ------

def print_seasonal_summary(obs, sim_df):
    season_defs = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }
    metric_keys = [
        "mean (m/s)", "std (m/s)", "median (m/s)",
        "q01 (m/s)", "q05 (m/s)", "q95 (m/s)", "q99 (m/s)",
        "pct_above_12ms (%)", "pct_below_1ms (%)",
        "Weibull k", "ACF lag-24h", "ACF lag-48h",
    ]

    def _season_metrics(vals):
        acf_v  = sm_acf(vals, nlags=48, fft=True)
        shape, _, _ = scipy.stats.weibull_min.fit(np.clip(vals, 1e-3, None), floc=0)
        return {
            "mean (m/s)":          float(np.mean(vals)),
            "std (m/s)":           float(np.std(vals)),
            "median (m/s)":        float(np.median(vals)),
            "q01 (m/s)":           float(np.percentile(vals, 1)),
            "q05 (m/s)":           float(np.percentile(vals, 5)),
            "q95 (m/s)":           float(np.percentile(vals, 95)),
            "q99 (m/s)":           float(np.percentile(vals, 99)),
            "pct_above_12ms (%)":  float(100 * np.mean(vals > 12.0)),
            "pct_below_1ms (%)":   float(100 * np.mean(vals < 1.0)),
            "Weibull k":           float(shape),
            "ACF lag-24h":         float(acf_v[24]),
            "ACF lag-48h":         float(acf_v[48]),
        }

    obs_rows = {}
    sim_rows = {}
    for sname, months in season_defs.items():
        obs_vals = obs[_season_mask(obs.index, months)].values
        obs_rows[sname] = _season_metrics(obs_vals)

        path_metrics = []
        for col in sim_df.columns:
            sim_vals = sim_df[col][_season_mask(sim_df.index, months)].values
            path_metrics.append(_season_metrics(sim_vals))
        sim_rows[sname] = {k: float(np.mean([p[k] for p in path_metrics])) for k in metric_keys}

    col_w = 10
    season_names = list(season_defs.keys())
    header = f"{'Metric':<22}" + "".join(
        f"{'':>{col_w}}{s}_obs{'':<0}{' ':>{col_w - len(s) - 4 + col_w}}{s}_sim"
        for s in season_names
    )
    header = (
        f"{'Metric':<22}"
        + "".join(f"{s+'_obs':>{col_w}}  {s+'_sim':>{col_w}}  " for s in season_names)
    )
    print(header)
    print("-" * len(header))
    for k in metric_keys:
        row = f"{k:<22}"
        for s in season_names:
            row += f"{obs_rows[s][k]:>{col_w}.3f}  {sim_rows[s][k]:>{col_w}.3f}  "
        print(row)

    import pandas as pd
    records = {}
    for sname in season_names:
        for k in metric_keys:
            records[(sname, "obs")] = records.get((sname, "obs"), {})
            records[(sname, "obs")][k] = obs_rows[sname][k]
            records[(sname, "sim")] = records.get((sname, "sim"), {})
            records[(sname, "sim")][k] = sim_rows[sname][k]

    csv_rows = []
    for sname in season_names:
        row = {"Season": sname}
        for k in metric_keys:
            row[f"{k}_obs"] = obs_rows[sname][k]
            row[f"{k}_sim"] = sim_rows[sname][k]
        csv_rows.append(row)

    df_csv = pd.DataFrame(csv_rows).set_index("Season")
    csv_path = PLOT_DIR / "seasonal_summary.csv"
    df_csv.to_csv(csv_path)
    print(f"\n  Saved: seasonal_summary.csv")


# ------ Summary stats table ------

def print_summary_table(obs, sim_df):
    obs_vals = obs.values
    obs_acf  = _acf_array(obs, nlags=48)

    obs_metrics = {
        "mean (m/s)":       float(np.mean(obs_vals)),
        "std (m/s)":        float(np.std(obs_vals)),
        "Weibull k":        _weibull_k(obs_vals),
        "ACF lag-24h":      float(obs_acf[24]),
        "ACF lag-48h":      float(obs_acf[48]),
        "% above 12 m/s":  float(100 * np.mean(obs_vals > 12.0)),
        "Monthly Spearman": _monthly_spearman(obs),
    }

    per_path = []
    for col in sim_df.columns:
        v    = sim_df[col].values
        acf_ = _acf_array(sim_df[col], nlags=48)
        per_path.append({
            "mean (m/s)":       float(np.mean(v)),
            "std (m/s)":        float(np.std(v)),
            "Weibull k":        _weibull_k(v),
            "ACF lag-24h":      float(acf_[24]),
            "ACF lag-48h":      float(acf_[48]),
            "% above 12 m/s":  float(100 * np.mean(v > 12.0)),
            "Monthly Spearman": _monthly_spearman(sim_df[col]),
        })

    keys = list(obs_metrics.keys())
    sim_mean_vals = {k: float(np.mean([p[k] for p in per_path])) for k in keys}
    sim_std_vals  = {k: float(np.std( [p[k] for p in per_path])) for k in keys}

    col_w = 18
    header = f"{'Metric':<22}{'Observed':>{col_w}}{'Sim mean':>{col_w}}{'Sim std':>{col_w}}"
    print(header)
    print("-" * len(header))
    for k in keys:
        print(
            f"{k:<22}"
            f"{obs_metrics[k]:>{col_w}.4f}"
            f"{sim_mean_vals[k]:>{col_w}.4f}"
            f"{sim_std_vals[k]:>{col_w}.4f}"
        )


# ------ Main ------

def main():
    print("=" * 60)
    print(f"validate_wind — MODEL_NAME={MODEL_NAME}")
    print("=" * 60)

    obs, sim_df = load_data()

    print(f"\n------ Generating plots → {PLOT_DIR} ------")
    plot_acf_comparison(obs, sim_df)
    plot_kde_marginal(obs, sim_df)
    plot_monthly_stats(obs, sim_df)
    plot_diurnal_mean(obs, sim_df)
    plot_storm_spell(obs, sim_df)
    plot_qq(obs, sim_df)
    plot_seasonal_hour_marginals(obs, sim_df)
    plot_seasonal_acf(obs, sim_df)
    plot_sample_timeseries(obs, sim_df)
    plot_psd_comparison(obs, sim_df)
    plot_psd_comparison_freq(obs, sim_df)

    print("\n------ Summary stats ------")
    print_summary_table(obs, sim_df)

    print("\n------ Seasonal summary ------")
    print_seasonal_summary(obs, sim_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
