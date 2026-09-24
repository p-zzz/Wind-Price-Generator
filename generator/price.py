"""Price MDN v11: DK1 day-ahead price as a conditional Gaussian mixture.

Inputs per hour: calendar, 10 m wind speed, ``solar_load_ratio`` and
``wind_load_ratio`` (scaled MW / load, so ``wind.scale``/``solar.scale`` move
price), load, net position and gas price. The shipped model is a 10-seed ensemble
(`MDNEnsemble`) trained on 2015-2025 with signed net position.

In-domain range: only up to ~1.25x wind.scale/solar.scale (at 2x, 12-13% of hours
have a solar/wind load ratio beyond the 2015-2025 training data). Beyond that,
aggregate statistics are usable with caution but individual hours are extrapolated.
Beyond ~3x the price response saturates and capture rates stop falling -- an
extrapolation artifact. Joint scaling behaves like the more extrapolated knob. See
README "Known limitations" (scale sweep on the 10-seed ensemble, 2026-09).
"""

import numpy as np
import torch
import torch.nn as nn

LOG_STD_MIN = -4.0
LOG_STD_MAX = 6.0

#: Column order the fitted model expects. simulate_price_mdn() passes its inputs
#: positionally in this order, and the wind x solar interaction term is built from
#: columns 0 and 1 -- load_fitted_objects() asserts the pkl's "feature_columns"
#: match exactly, so a reordered pkl fails loudly instead of mislabelling inputs.
PRICE_FEATURE_COLS = [
    "horns_rev_wind_speed_10m_ms",
    "solar_load_ratio",
    "actual_load_MW",
    "net_position_MW",
    "ttf_gas_price_eur_mwh",
    "wind_load_ratio",
]


class MDN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, K: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, K * 3)
        self.K = K

    def forward(self, x):
        h = self.backbone(x)
        out = self.head(h)
        K = self.K
        logits = out[:, :K]
        means = out[:, K : 2 * K]
        log_std = out[:, 2 * K :].clamp(LOG_STD_MIN, LOG_STD_MAX)
        pi = torch.softmax(logits, dim=-1)
        return pi, means, log_std


class MDNEnsemble(nn.Module):
    """Equal-weight ensemble of MDNs trained on the same data with different seeds.

    Pools every member's K components into one mixture of M*K components with
    weights pi/M -- sampling from it is exactly "pick a member uniformly, then sample
    from that member". Same (pi, means, log_std) interface as MDN, so
    simulate_price_mdn() takes either. Averages out the seed-to-seed spread of
    single models (see README "Known limitations").
    """

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        outs = [m(x) for m in self.members]
        n = len(outs)
        pi = torch.cat([o[0] for o in outs], dim=-1) / n
        means = torch.cat([o[1] for o in outs], dim=-1)
        log_std = torch.cat([o[2] for o in outs], dim=-1)
        return pi, means, log_std


def _calendar_features(index) -> np.ndarray:
    hour = index.hour
    month = index.month
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
        ]
    ).astype(np.float32)


def simulate_price_mdn(
    idx,
    wind_speed: np.ndarray,
    solar_load_ratio: np.ndarray,
    load: np.ndarray,
    net_position: np.ndarray,
    gas_price: float | np.ndarray,
    wind_load_ratio: np.ndarray,
    mdn: nn.Module,
    norm_stats: dict,
    rng: np.random.Generator,
    feature_cols: list[str],
    device: torch.device,
) -> np.ndarray:
    """Draw one day-ahead price per hour from the model's conditional distribution.

    Parameters
    ----------
    idx : pandas.DatetimeIndex
        Local-time calendar (hour, month feed the model).
    wind_speed : numpy.ndarray, shape (n_hours,)
        10 m wind speed at Horns Rev, m/s.
    solar_load_ratio, wind_load_ratio : numpy.ndarray, shape (n_hours,)
        Solar / wind generation at scaled capacity divided by load (dimensionless).
    load, net_position : numpy.ndarray, shape (n_hours,)
        Load and net position (export-positive), MW.
    gas_price : float or numpy.ndarray of shape (n_hours,)
        TTF gas price, EUR/MWh.
    mdn : torch.nn.Module
        `MDN` or `MDNEnsemble` (`generator.io.load_fitted_objects` ``["mdn"]``).
    norm_stats : dict
        Per-input training mean/std, from the price model file.
    rng : numpy.random.Generator
        Draws the mixture component and the Gaussian sample (two draws per hour).
    feature_cols : list of str
        Input order; must equal `PRICE_FEATURE_COLS` (checked at load time).
    device : torch.device

    Returns
    -------
    numpy.ndarray of float32, shape (n_hours,)
        Day-ahead price, EUR/MWh. Not clipped: can fall outside the market's
        harmonised limits.

    Notes
    -----
    Inputs are standardised with the training statistics and never clipped, so
    values outside the 2015-2025 range are extrapolated. Hours are sampled
    independently given their inputs; price autocorrelation comes from the
    autocorrelated inputs.

    Warnings
    --------
    In-domain only up to ~1.25x on the scale knobs; see the module docstring.
    """
    gas_arr = (
        np.full(len(idx), gas_price, dtype=np.float32)
        if np.isscalar(gas_price)
        else np.asarray(gas_price, dtype=np.float32)
    )
    values = [wind_speed, solar_load_ratio, load, net_position, gas_arr, wind_load_ratio]
    raw = dict(zip(feature_cols, values))
    cal = _calendar_features(idx)
    cont = np.column_stack(
        [
            (raw[col] - norm_stats[col]["mean"]) / norm_stats[col]["std"]
            for col in feature_cols
        ]
    ).astype(np.float32)
    interaction = (cont[:, 0] * cont[:, 1]).reshape(-1, 1)
    X = np.concatenate([cal, cont, interaction], axis=1)

    with torch.no_grad():
        pi_t, means_t, log_std_t = mdn(torch.from_numpy(X).to(device))
        pi_np = pi_t.cpu().numpy()
        means_np = means_t.cpu().numpy()
        std_np = log_std_t.exp().cpu().numpy()

    n, n_components = pi_np.shape  # K for one MDN, M*K for an MDNEnsemble
    u = rng.uniform(0.0, 1.0, size=n)
    cumsum = np.cumsum(pi_np, axis=1)
    k_idx = np.clip((u[:, None] >= cumsum).sum(axis=1), 0, n_components - 1)
    sel_means = means_np[np.arange(n), k_idx]
    sel_stds = std_np[np.arange(n), k_idx]
    return rng.normal(sel_means, sel_stds).astype(np.float32)
