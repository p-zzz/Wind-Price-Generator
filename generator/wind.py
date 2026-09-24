"""Wind: 10 m wind speed (Transformer), fleet generation (CF ARMA), hub-height wind.

Two separate streams:

* ``wind_speed_ms`` -- hourly 10 m wind speed at the Horns Rev ERA5 point, from an
  autoregressive Transformer mixture model plus a seasonal Weibull quantile mapper.
  It feeds the price model directly and is never scaled: ``wind.scale`` multiplies
  capacity, not wind speed.
* ``wind_generation_MW`` -- DK1 onshore + offshore fleet output: simulated capacity
  factor (conditioned on the same wind speed) x installed capacity x ``wind.scale``.

`hub_height_wind_speed` converts the 10 m wind to turbine hub height for output only.
Simulation code is extracted from the thesis repo's ``models/wind/`` scripts
(training code stripped); the classes stay importable under their original module
names so the fitted pickles load (`generator.io`).
"""

import math

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn

#: Name of the 10 m Horns Rev wind-speed series in the fitted models and data files.
WIND_COL = "horns_rev_wind_speed_10m_ms"


# ------ Wind Transformer architecture ------


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class WindTransformerMDN(nn.Module):
    def __init__(
        self,
        k_lags: int = 72,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        K: int = 8,
        log_std_min: float = -4.0,
        log_std_max: float = 6.0,
    ):
        super().__init__()
        self.K = K
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.lag_embed = nn.Linear(1, d_model)
        self.pos_enc = SinusoidalPE(d_model, max_len=k_lags + 4)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        mask = torch.tril(torch.ones(k_lags, k_lags, dtype=torch.bool), diagonal=-1)
        self.register_buffer("causal_mask", mask)

        mlp_in = d_model + 4
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mdn_head = nn.Linear(64, K * 3)

    def forward(self, lags: torch.Tensor, cal: torch.Tensor):
        x = self.lag_embed(lags.unsqueeze(-1))
        x = self.pos_enc(x)
        x = self.transformer(x, mask=self.causal_mask)
        x0 = x[:, 0, :]
        h = self.mlp(torch.cat([x0, cal], dim=-1))
        out = self.mdn_head(h)
        K = self.K
        logits = out[:, :K]
        mu = out[:, K : 2 * K]
        log_std = out[:, 2 * K :].clamp(self.log_std_min, self.log_std_max)
        return torch.softmax(logits, dim=-1), mu, log_std.exp()


class WindQuantileMapper:
    """Post-hoc seasonal quantile mapper for wind speed (parametric Weibull per
    season x 6h-block bin) -- enforces the correct marginal distribution on
    simulated Transformer paths."""

    _SEASONS = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
    }
    _HOUR_BLOCKS = {
        "night": list(range(0, 6)),
        "morning": list(range(6, 12)),
        "afternoon": list(range(12, 18)),
        "evening": list(range(18, 24)),
    }

    def fit(self, observed: pd.Series):
        self.params_ = {}
        for season, months in self._SEASONS.items():
            for block, hours in self._HOUR_BLOCKS.items():
                mask = observed.index.month.isin(months) & observed.index.hour.isin(
                    hours
                )
                vals = np.maximum(observed[mask].values, 1e-6)
                c, _, scale = stats.weibull_min.fit(vals, floc=0)
                self.params_[(season, block)] = (float(c), float(scale))
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
                ranks = stats.rankdata(vals).astype(np.float64)
                probs = np.clip(ranks / n, 1e-6, 1.0 - 1e-6)
                c, scale = self.params_[(season, block)]
                corrected = stats.weibull_min.ppf(probs, c=c, scale=scale)
                out[mask] = np.maximum(corrected, 0.0)
        return pd.Series(out, index=sim.index, name=sim.name)


def run_transformer_simulation(
    model: nn.Module,
    norm_stats: dict,
    K: int,
    mapper: WindQuantileMapper,
    buf: list,
    index: pd.DatetimeIndex,
    burn_in: int,
    rng: np.random.Generator,
    device: torch.device,
) -> pd.DataFrame:
    """Simulate hourly 10 m wind speed at Horns Rev on the given calendar.

    Parameters
    ----------
    model : torch.nn.Module
        Fitted `WindTransformerMDN`.
    norm_stats : dict
        ``{"mean", "std"}`` wind-speed normalisation, m/s.
    K : int
        Number of mixture components of ``model``.
    mapper : WindQuantileMapper
        Maps the raw simulated path onto the climatological Weibull distribution
        per season x 6-hour block.
    buf : list of float
        Normalised lag buffer, oldest first (`generator.io.load_wind_speed_seed`).
        **Mutated in place.**
    index : pandas.DatetimeIndex
        Hourly, tz-aware Europe/Copenhagen simulation calendar
        (`generator.run.build_sim_index`).
    burn_in : int
        Hours simulated before ``index[0]`` and discarded, so the path starts from
        an in-season state.
    rng : numpy.random.Generator
        Draws the mixture component and the Gaussian sample every hour.
    device : torch.device

    Returns
    -------
    pandas.DataFrame
        One column `WIND_COL`, m/s, indexed by ``index``.

    Raises
    ------
    ValueError
        If ``index`` is not tz-aware Europe/Copenhagen.

    Notes
    -----
    Model and mapper were trained on local-time hour and month, so DST is handled by
    construction (absolute-hour steps, local wall-clock features); leap days need no
    special case. The quantile mapper ranks each path over its whole horizon, so every
    path has the same climatological distribution per season x block: a run cannot
    produce an unusually windy or calm period overall.
    """
    if str(index.tz) != "Europe/Copenhagen":
        raise ValueError(f"index must be tz-aware Europe/Copenhagen, got tz={index.tz}")
    model.eval()
    norm_floor = float(-norm_stats["mean"] / norm_stats["std"])

    def _step(ts):
        cal = np.array(
            [
                np.sin(2 * np.pi * ts.hour / 24),
                np.cos(2 * np.pi * ts.hour / 24),
                np.sin(2 * np.pi * ts.month / 12),
                np.cos(2 * np.pi * ts.month / 12),
            ],
            dtype=np.float32,
        )
        lags_arr = np.array(list(reversed(buf)), dtype=np.float32)
        lags_t = torch.from_numpy(lags_arr).unsqueeze(0).to(device)
        cal_t = torch.from_numpy(cal).unsqueeze(0).to(device)
        pi, mu, sigma = model(lags_t, cal_t)
        pi_np = pi.squeeze(0).cpu().numpy()
        mu_np = mu.squeeze(0).cpu().numpy()
        sg_np = sigma.squeeze(0).cpu().numpy()
        k = int(rng.choice(K, p=pi_np))
        sample = float(rng.normal(mu_np[k], sg_np[k]))
        return max(sample, norm_floor)

    # Burn-in runs over the burn_in hours immediately preceding index[0], so the
    # AR state has warmed up under the right season when the kept path starts.
    burn_idx = pd.date_range(
        end=index[0] - pd.Timedelta(hours=1), periods=burn_in, freq="h"
    )
    synth_index = index

    with torch.no_grad():
        for ts in burn_idx:
            s = _step(ts)
            buf.append(s)
            buf.pop(0)

        results = []
        for ts in synth_index:
            s = _step(ts)
            buf.append(s)
            buf.pop(0)
            results.append(s)

    wind_raw = np.maximum(
        np.array(results, dtype=np.float32) * norm_stats["std"] + norm_stats["mean"], 0.0
    )
    sim_df = pd.DataFrame({WIND_COL: wind_raw}, index=synth_index)
    sim_df[WIND_COL] = mapper.transform(sim_df[WIND_COL])
    return sim_df


# ------ Wind capacity-factor ARMA (onshore/offshore generation) ------


class WindCFQuantileMapper:
    """Post-hoc seasonal empirical quantile mapping for wind capacity factor.
    Empirical (not parametric) because offshore CF is genuinely bimodal (a calm
    mode and a rated-power saturation plateau) -- no unimodal parametric family
    can reproduce two humps."""

    _SEASONS = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
    }
    _HOUR_BLOCKS = {
        "night": list(range(0, 6)),
        "morning": list(range(6, 12)),
        "afternoon": list(range(12, 18)),
        "evening": list(range(18, 24)),
    }

    def fit(self, observed: pd.Series):
        self.sorted_ = {}
        for season, months in self._SEASONS.items():
            for block, hours in self._HOUR_BLOCKS.items():
                mask = observed.index.month.isin(months) & observed.index.hour.isin(hours)
                vals = np.clip(observed[mask].values, 0.0, 1.0)
                self.sorted_[(season, block)] = np.sort(vals)
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
                ranks = stats.rankdata(vals).astype(np.float64)
                probs = (ranks - 0.5) / n

                target_sorted = self.sorted_[(season, block)]
                m = len(target_sorted)
                target_probs = (np.arange(1, m + 1) - 0.5) / m
                corrected = np.interp(probs, target_probs, target_sorted)
                out[mask] = np.clip(corrected, 0.0, 1.0)
        return pd.Series(out, index=sim.index, name=sim.name)


def _cf_unstandardise(std_resid: pd.Series, monthly_std: dict) -> pd.Series:
    multiplier = std_resid.index.month.map(monthly_std)
    return std_resid * multiplier.values


def simulate_wind_cf(
    df: pd.DataFrame,
    cf_pkl: dict,
    wind_speed_ms: np.ndarray,
    burn_in: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate an hourly wind capacity factor (onshore or offshore fleet).

    Seasonal mean conditioned on calendar and the simulated wind speed, plus an
    ARMA residual, then an empirical quantile mapper per season x 6-hour block.
    Conditioning on wind speed ties generation to the same wind path the price
    model sees.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation calendar (`generator.run.build_sim_index`).
    cf_pkl : dict
        Fitted onshore or offshore CF model (`generator.io.load_fitted_objects`).
    wind_speed_ms : numpy.ndarray, shape (n_hours,)
        Simulated 10 m wind speed, m/s.
    burn_in : int
        ARMA hours discarded at the start.
    rng : numpy.random.Generator
        Seeds the ARMA residual simulation.

    Returns
    -------
    numpy.ndarray, shape (n_hours,)
        Capacity factor, fraction of installed capacity in [0, 1].
    """
    mean_model = cf_pkl["mean_model"]
    monthly_std = cf_pkl["monthly_std"]
    arma_model = cf_pkl["arma_model"]
    mapper = cf_pkl["quantile_mapper"]
    n = len(df)

    df_in = df.copy()
    df_in[WIND_COL] = wind_speed_ms
    seasonal_mean = mean_model.predict(df_in)
    sim_resid = arma_model.simulate(nsimulations=n + burn_in, random_state=rng.integers(int(1e9)))
    sim_resid = sim_resid[burn_in:]
    resid = _cf_unstandardise(pd.Series(sim_resid, index=df.index), monthly_std)

    cf_sim = np.clip(seasonal_mean.values + resid.values, 0.0, 1.0)
    cf_series = mapper.transform(pd.Series(cf_sim, index=df.index))
    return cf_series.values


def simulate_wind_generation(
    df: pd.DataFrame,
    cf_onshore_pkl: dict,
    cf_offshore_pkl: dict,
    capacity_baseline: dict,
    scale: float | dict,
    wind_speed_ms: np.ndarray,
    burn_in: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate DK1 onshore and offshore wind generation at scaled capacity.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation calendar (`generator.run.build_sim_index`).
    cf_onshore_pkl, cf_offshore_pkl : dict
        Fitted CF models (`generator.io.load_fitted_objects`).
    capacity_baseline : dict
        ``{"onshore": MW, "offshore": MW}`` installed capacity.
    scale : float or dict
        Capacity multiplier: one float for both, or ``{"onshore": x, "offshore": y}``.
        Scales capacity only -- the capacity factors and wind speed are unchanged.
    wind_speed_ms : numpy.ndarray, shape (n_hours,)
        Simulated 10 m wind speed, m/s.
    burn_in : int
        ARMA hours discarded at the start.
    rng : numpy.random.Generator
        Consumed identically for any ``scale``.

    Returns
    -------
    gen_onshore, gen_offshore, gen_total : numpy.ndarray, shape (n_hours,)
        Generation, MW.
    cf_onshore, cf_offshore : numpy.ndarray, shape (n_hours,)
        Unscaled capacity factors in [0, 1].
    """
    if isinstance(scale, dict):
        scale_onshore, scale_offshore = scale["onshore"], scale["offshore"]
    else:
        scale_onshore = scale_offshore = scale

    cf_onshore = simulate_wind_cf(df, cf_onshore_pkl, wind_speed_ms, burn_in, rng)
    cf_offshore = simulate_wind_cf(df, cf_offshore_pkl, wind_speed_ms, burn_in, rng)
    cap_onshore = capacity_baseline["onshore"] * scale_onshore
    cap_offshore = capacity_baseline["offshore"] * scale_offshore
    gen_onshore = cf_onshore * cap_onshore
    gen_offshore = cf_offshore * cap_offshore
    return gen_onshore, gen_offshore, gen_onshore + gen_offshore, cf_onshore, cf_offshore


# ------ Hub-height wind speed ------


def hub_height_wind_speed(
    wind_speed_10m: np.ndarray,
    index: pd.DatetimeIndex,
    hub_height_m: float,
    shear: pd.DataFrame,
) -> np.ndarray:
    """Extrapolate the 10 m wind speed to turbine hub height.

    ``v_hub = v10 * (hub_height_m / 10) ** alpha``, with ``alpha`` looked up per
    calendar month x 10 m wind-speed band.

    Parameters
    ----------
    wind_speed_10m : numpy.ndarray, shape (n_hours,)
        10 m wind speed, m/s.
    index : pandas.DatetimeIndex
        Local-time calendar of ``wind_speed_10m`` (for the month).
    hub_height_m : float
        Hub height, m.
    shear : pandas.DataFrame
        Shear table (`generator.io.load_shear_table`).

    Returns
    -------
    numpy.ndarray, shape (n_hours,)
        Hub-height wind speed, m/s.

    Notes
    -----
    ``alpha`` is measured from ERA5's own 10 m / 100 m wind at Horns Rev, 1990-2026
    (about 0.10 on average, 0.05-0.12 by cell; unbiased on 100 m mean speed). Output
    only: every model runs on the 10 m wind. Valid offshore in the Danish North Sea;
    it is not an onshore site's wind. A generic exponent such as 0.2 overstates
    offshore hub-height wind by about a third.
    """
    months = np.asarray(index.month)
    alpha = np.full(len(wind_speed_10m), np.nan)
    for row in shear.itertuples():
        cell = (months == row.month) & (wind_speed_10m >= row.v10_min_ms) & (wind_speed_10m < row.v10_max_ms)
        alpha[cell] = row.alpha
    if np.isnan(alpha).any():
        raise ValueError("shear table does not cover every month x wind-speed band")
    return wind_speed_10m * (hub_height_m / 10.0) ** alpha


# ------ Wind at a chosen farm site ------


def _hour_block_cells(index: pd.DatetimeIndex, per_day: int = 4) -> np.ndarray:
    """Cell id per hour: month x ``per_day`` equal blocks of the local day
    (``per_day=24``: month x hour, 0..287; ``per_day=4``: month x 6-hour block)."""
    return per_day * (np.asarray(index.month) - 1) + np.asarray(index.hour) // (24 // per_day)


def _lagged(u: np.ndarray, lags) -> np.ndarray:
    """(n, len(lags)) matrix of u shifted by each lag (u[t - lag]), edge-padded."""
    n = len(u)
    out = np.empty((n, len(lags)))
    for j, lag in enumerate(lags):
        src = np.clip(np.arange(n) - lag, 0, n - 1)
        out[:, j] = u[src]
    return out


def site_wind_speed(
    system_wind_10m: np.ndarray,
    index: pd.DatetimeIndex,
    site_model: dict,
    hub_height_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Wind at a farm site, hour by hour, consistent with the simulated system wind.

    Gaussian copula per calendar month x local hour, fitted on decades of paired ERA5
    hours (``scripts/build_site_model.py``): the system (Horns Rev) 10 m wind is turned
    into normal scores, the site's score is a regression on the system scores over a
    window of hours around ``t`` (so weather reaching inland sites hours after Horns Rev
    is reproduced) plus an AR(1) process (the site's own, persistent deviations), and
    the score is mapped back through the site's own 100 m wind distribution. The
    100 m wind is then taken to hub height with the site's ERA5-measured shear.

    Parameters
    ----------
    system_wind_10m : numpy.ndarray, shape (n_hours,)
        Simulated system (Horns Rev) 10 m wind speed, m/s -- the generator's
        ``wind_speed_ms``.
    index : pandas.DatetimeIndex
        Local-time calendar of ``system_wind_10m``.
    site_model : dict
        Fitted site model (`generator.io.load_site_model`).
    hub_height_m : float
        Hub height at the site, m.
    rng : numpy.random.Generator
        Draws the site's own deviations (one normal per hour, plus one to start).

    Returns
    -------
    numpy.ndarray, shape (n_hours,)
        Wind speed at the site and hub height, m/s.

    Notes
    -----
    The site keeps its own climatological distribution (per month x hour), moves
    together with the system wind as much as the real weather does, including the
    delay with which fronts reach it, and its local deviations persist from hour to
    hour (AR(1) coefficient ``phi``). ERA5's ~0.25 deg grid point stands in for the site.
    """
    from scipy.special import ndtr, ndtri   # standard normal CDF and its inverse

    probs = np.asarray(site_model["probs"])
    q_x, q_y = np.asarray(site_model["q_system_10m"]), np.asarray(site_model["q_site_100m"])
    per_day = q_x.shape[1]                 # 24 (month x hour) or 4 (older 6-hour-block files)
    cells = _hour_block_cells(index, per_day)
    months = np.asarray(index.month) - 1

    u = np.empty(len(system_wind_10m))
    for c in np.unique(cells):
        m, b = divmod(c, per_day)
        mask = cells == c
        u[mask] = ndtri(np.interp(system_wind_10m[mask], q_x[m, b], probs))

    if "beta" in site_model:               # lagged regression on the system scores
        lags = site_model["lags"]
        beta = np.asarray(site_model["beta"])[months]          # (n, n_lags)
        signal = (_lagged(u, lags) * beta).sum(axis=1)
    else:                                  # older files: same-hour correlation only
        signal = np.asarray(site_model["rho"])[months] * u

    phi = float(site_model["phi"])
    e_std = np.asarray(site_model["e_std"])[months]
    innov = rng.standard_normal(len(u)) * e_std * np.sqrt(1.0 - phi**2)
    e = np.empty(len(u))
    e[0] = rng.standard_normal() * e_std[0]
    for t in range(1, len(u)):
        e[t] = phi * e[t - 1] + innov[t]
    w = signal + e

    v100 = np.empty(len(u))
    p = ndtr(w)
    for c in np.unique(cells):
        m, b = divmod(c, per_day)
        mask = cells == c
        v100[mask] = np.interp(p[mask], probs, q_y[m, b])

    edges = np.asarray(site_model["shear_v100_edges"], dtype=float)
    band = np.clip(np.searchsorted(edges, v100, side="right") - 1, 0, len(edges) - 2)
    alpha = np.asarray(site_model["shear_alpha"])[months, band]
    return v100 * (hub_height_m / 100.0) ** alpha
