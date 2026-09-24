"""
Wind stack: Transformer (wind speed, m/s) + capacity-factor ARMA (onshore/offshore,
generation MW). Extracted from the thesis repo's models/wind/wind_transformer.py and
models/wind/wind_capacity_factor_arma.py -- training/tuning/comparison code stripped,
simulation-only classes and functions kept verbatim.

wind_speed_ms feeds the price model's merit-order channel directly and unscaled --
WIND_SCALE never touches it (scaling wind speed has no physical meaning as a capacity
proxy). wind_generation_MW is a separate, independently-scaled stream: simulated
capacity factor (onshore/offshore ARMA + quantile mapper) x target installed capacity,
where WIND_SCALE multiplies the configured onshore/offshore capacity baseline.
"""

import math

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn

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
    """Free-running autoregressive simulation, burn_in steps discarded. buf is the
    AR lag buffer (already normalised), most-recent-last; mutated in place.

    index is the simulation's own hourly Europe/Copenhagen index (build_sim_index),
    so wind calendar features and quantile-mapper bins line up hour-for-hour with
    solar, load and price. The Transformer and mapper were trained on Copenhagen
    local-time hour/month, so DST is handled by construction: the index steps in
    absolute hours and ts.hour is the local wall-clock hour, exactly as in training.
    Leap days need no special case -- calendar features use only month and hour."""
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
    """Mean model conditions on wind speed (in addition to calendar) -- ties the
    CF simulation to the already-simulated wind_speed_ms path instead of drawing
    an independent, uncorrelated process."""
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
    """scale: either a single float (unified wind scale, onshore + offshore) or a
    {"onshore": x, "offshore": y} dict for independent per-component scaling.

    Returns (gen_onshore_MW, gen_offshore_MW, gen_total_MW, cf_onshore, cf_offshore).
    The raw (unscaled) capacity factors are also returned so callers can build a
    scale-invariant wind_gen_ratio by re-weighting with the *baseline* capacity
    split rather than the scenario's scaled one."""
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
