"""
Price MDN v11 -- the only price model shipped in this repo (see README "Known
limitations" for why v10 was not: it's scale-invariant by construction, so
WIND_SCALE/SOLAR_SCALE cannot move price at all under it, defeating the point of a
cannibalisation-risk tool). v11 replaces solar_cf/wind_gen_ratio with
solar_load_ratio = solar_MW/actual_load_MW and wind_load_ratio =
wind_generation_MW/actual_load_MW -- built from the already-scaled MW streams, so
the scale knobs flow through to price.

CONFIRMED FAILURE MODE: v11's price response to solar reverses sign at
SOLAR_SCALE~=3.0x (isolated-feature probe + full correlated-generator confirmation
in the thesis repo, both cited in README). Treat any run with solar.scale >= ~2.5x
as outside the validated region.
"""

import numpy as np
import torch
import torch.nn as nn

LOG_STD_MIN = -4.0
LOG_STD_MAX = 6.0

# Column order the fitted model expects: wind speed, solar-thing, load,
# net position, gas, wind-thing -- read from the pkl's "feature_columns" at load
# time (solar_load_ratio/wind_load_ratio for v11), not hardcoded here.


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
    K: int,
    rng: np.random.Generator,
    feature_cols: list[str],
    device: torch.device,
) -> np.ndarray:
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

    n = len(X)
    u = rng.uniform(0.0, 1.0, size=n)
    cumsum = np.cumsum(pi_np, axis=1)
    k_idx = np.clip((u[:, None] >= cumsum).sum(axis=1), 0, K - 1)
    sel_means = means_np[np.arange(n), k_idx]
    sel_stds = std_np[np.arange(n), k_idx]
    return rng.normal(sel_means, sel_stds).astype(np.float32)
