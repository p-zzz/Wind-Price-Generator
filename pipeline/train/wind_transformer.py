"""
Causal Transformer + MDN for Horns Rev wind speed simulation.
Post-hoc seasonal Weibull quantile mapping for marginal calibration.

Pipeline
--------
1. Load ERA5 Horns Rev 10m wind speed (2014-2025); normalise
2. Build features: 72 AR lag tokens + 4 calendar features (kept separate)
3. Train WindTransformerMDN minimising Gaussian mixture CRPS
4. Fit WindQuantileMapper on observed data (24 seasonal x diurnal bins)
5. Simulate: free-running autoregressive, 200h burn-in, apply quantile mapper
6. Comparison table: Observed / ARMA / MDN r4b / MDN r6 / Transformer
7. Validation plots saved to analysis/outputs/DK1/wind_model/ (suffix _transformer)
8. Fitted objects: models/wind/fitted/wind_transformer.pkl / wind_transformer_best.pt
"""

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.stattools import acf as sm_acf

plt.style.use("seaborn-v0_8")

ERA5_PATH  = Path("DATA/processed/era5_sites_wind_hourly.parquet")
OUTDIR     = Path("analysis/outputs/DK1/wind_model")
FITTED_DIR = Path("models/wind/fitted")
OUTDIR.mkdir(parents=True, exist_ok=True)
FITTED_DIR.mkdir(parents=True, exist_ok=True)

WIND_COL     = "horns_rev_wind_speed_10m_ms"
TRAIN_START  = "2014-12-01"
TRAIN_END    = "2025-12-31"

K_LAGS       = 72
D_MODEL      = 64
N_HEADS      = 4
N_LAYERS     = 2
DIM_FF       = 256
DROPOUT      = 0.1
K_COMPONENTS = 8
LOG_STD_MIN  = -4.0
LOG_STD_MAX  = 6.0

BATCH_SIZE   = 512
EPOCHS       = 200
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 15
RANDOM_SEED  = 42

N_SIM_PATHS  = 25
SIM_HOURS    = 43800
BURN_IN      = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ------ Patsy patch ------

def _apply_patsy_patch():
    import patsy.eval
    _orig = patsy.eval.EvalEnvironment.capture.__func__

    @classmethod
    def _patched(cls, eval_env=0, reference=0):
        try:
            return _orig(cls, eval_env, reference)
        except Exception:
            return cls({}, {})

    patsy.eval.EvalEnvironment.capture = _patched


# ------ Data loading & normalisation ------

def load_and_normalise(path: Path):
    df = pd.read_parquet(path)
    df = df.tz_convert("Europe/Copenhagen").sort_index()
    df = df.loc[TRAIN_START:TRAIN_END, [WIND_COL]].dropna()
    mean_v = float(df[WIND_COL].mean())
    std_v  = float(df[WIND_COL].std())
    df[WIND_COL + "_norm"] = (df[WIND_COL] - mean_v) / std_v
    norm_stats = {"mean": mean_v, "std": std_v}
    print(f"Loaded {len(df)} obs  ({df.index[0]} .. {df.index[-1]})")
    print(f"  {WIND_COL}: mean={mean_v:.4f}  std={std_v:.4f}")
    return df, norm_stats


# ------ Calendar features ------

def _calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    return np.column_stack([
        np.sin(2 * np.pi * index.hour  / 24),
        np.cos(2 * np.pi * index.hour  / 24),
        np.sin(2 * np.pi * index.month / 12),
        np.cos(2 * np.pi * index.month / 12),
    ]).astype(np.float32)


# ------ Feature construction ------

def build_features(df: pd.DataFrame, k_lags: int):
    vals    = df[WIND_COL + "_norm"].values.astype(np.float32)
    n       = len(vals)
    cal     = _calendar_features(df.index)
    lag_mat = np.zeros((n, k_lags), dtype=np.float32)
    for k in range(1, k_lags + 1):
        lag_mat[k_lags:, k - 1] = vals[k_lags - k : n - k]
    X_lags = lag_mat[k_lags:]        # (N, k_lags)  position 0 = t-1 (most recent)
    X_cal  = cal[k_lags:]            # (N, 4)
    y      = vals[k_lags:].reshape(-1, 1)
    print(f"Features: X_lags={X_lags.shape}  X_cal={X_cal.shape}  y={y.shape}")
    return X_lags, X_cal, y


# ------ CRPS loss — closed form for Gaussian mixture ------

_SQRT2   = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _phi(z: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * z * z) / _SQRT2PI


def _Phi(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / _SQRT2))


def crps_gaussian_mixture(pi: torch.Tensor, mu: torch.Tensor,
                           sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # pi, mu, sigma: (batch, K)   y: (batch,)
    z     = (y.unsqueeze(1) - mu) / sigma                          # (batch, K)
    term1 = (pi * sigma * (2.0 * _phi(z) + z * (2.0 * _Phi(z) - 1.0))).sum(dim=1)

    mu_j  = mu.unsqueeze(2);    mu_k  = mu.unsqueeze(1)            # (batch, K, 1/1, K)
    sg_j  = sigma.unsqueeze(2); sg_k  = sigma.unsqueeze(1)
    pi_j  = pi.unsqueeze(2);    pi_k  = pi.unsqueeze(1)

    sigma_jk = (sg_j ** 2 + sg_k ** 2).sqrt()                     # (batch, K, K)
    delta_jk = (mu_j - mu_k) / sigma_jk
    pair     = 2.0 * _phi(delta_jk) + delta_jk * (2.0 * _Phi(delta_jk) - 1.0)
    term2    = 0.5 * (pi_j * pi_k * sigma_jk * pair).sum(dim=(1, 2))

    return (term1 - term2).mean()


# ------ Sinusoidal positional encoding ------

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ------ Transformer MDN model ------

class WindTransformerMDN(nn.Module):
    """
    Causal Transformer encoder over 72 AR lag tokens, followed by MDN head.

    Lag ordering: position 0 = t-1 (most recent), position 71 = t-72 (oldest).
    Causal mask allows position i to attend to positions i..71 (itself and all
    older lags). Mask blocks j < i (more recent than i) via tril(ones, -1).
    The output token at position 0 therefore has full context over all 72 lags.
    """

    def __init__(
        self,
        k_lags: int          = K_LAGS,
        d_model: int         = D_MODEL,
        n_heads: int         = N_HEADS,
        n_layers: int        = N_LAYERS,
        dim_feedforward: int = DIM_FF,
        dropout: float       = DROPOUT,
        K: int               = K_COMPONENTS,
        log_std_min: float   = LOG_STD_MIN,
        log_std_max: float   = LOG_STD_MAX,
    ):
        super().__init__()
        self.K           = K
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.lag_embed = nn.Linear(1, d_model)
        self.pos_enc   = SinusoidalPE(d_model, max_len=k_lags + 4)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Position 0 (most recent) must attend to all lag positions.
        # Block j < i so position i cannot attend to more-recent lags.
        mask = torch.tril(torch.ones(k_lags, k_lags, dtype=torch.bool), diagonal=-1)
        self.register_buffer("causal_mask", mask)

        mlp_in = d_model + 4
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64),    nn.GELU(), nn.Dropout(dropout),
        )
        self.mdn_head = nn.Linear(64, K * 3)

    def forward(self, lags: torch.Tensor, cal: torch.Tensor):
        # lags: (batch, k_lags)  position 0 = t-1
        # cal:  (batch, 4)
        x  = self.lag_embed(lags.unsqueeze(-1))          # (batch, k_lags, d_model)
        x  = self.pos_enc(x)
        x  = self.transformer(x, mask=self.causal_mask)  # (batch, k_lags, d_model)
        x0 = x[:, 0, :]                                  # (batch, d_model)
        h  = self.mlp(torch.cat([x0, cal], dim=-1))      # (batch, 64)
        out     = self.mdn_head(h)
        K       = self.K
        logits  = out[:, :K]
        mu      = out[:, K : 2 * K]
        log_std = out[:, 2 * K :].clamp(self.log_std_min, self.log_std_max)
        return torch.softmax(logits, dim=-1), mu, log_std.exp()


# ------ Wind quantile mapper ------

class WindQuantileMapper:
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
        self.params_ = {}
        for season, months in self._SEASONS.items():
            for block, hours in self._HOUR_BLOCKS.items():
                mask = (
                    observed.index.month.isin(months)
                    & observed.index.hour.isin(hours)
                )
                vals = np.maximum(observed[mask].values, 1e-6)
                c, _, scale = scipy.stats.weibull_min.fit(vals, floc=0)
                self.params_[(season, block)] = (float(c), float(scale))
                print(f"  Weibull ({season:6s}, {block:9s}): "
                      f"k={c:.3f}  lambda={scale:.3f}  n={mask.sum()}")
        return self

    def transform(self, sim: pd.Series) -> pd.Series:
        out = sim.values.copy().astype(np.float64)
        for season, months in self._SEASONS.items():
            for block, hours in self._HOUR_BLOCKS.items():
                mask = (
                    sim.index.month.isin(months)
                    & sim.index.hour.isin(hours)
                )
                if not mask.any():
                    continue
                vals      = out[mask]
                n         = len(vals)
                ranks     = scipy.stats.rankdata(vals).astype(np.float64)
                probs     = np.clip(ranks / n, 1e-6, 1.0 - 1e-6)
                c, scale  = self.params_[(season, block)]
                corrected = scipy.stats.weibull_min.ppf(probs, c=c, scale=scale)
                out[mask] = np.maximum(corrected, 0.0)
        return pd.Series(out, index=sim.index, name=sim.name)


# ------ DataLoaders ------

def make_loaders(X_lags: np.ndarray, X_cal: np.ndarray, y: np.ndarray):
    n_train = int(len(X_lags) * 0.8)
    kw = dict(num_workers=4, pin_memory=True) if device.type == "cuda" else dict(num_workers=0)
    tr_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_lags[:n_train]),
            torch.from_numpy(X_cal[:n_train]),
            torch.from_numpy(y[:n_train]),
        ),
        batch_size=BATCH_SIZE, shuffle=True, **kw,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_lags[n_train:]),
            torch.from_numpy(X_cal[n_train:]),
            torch.from_numpy(y[n_train:]),
        ),
        batch_size=BATCH_SIZE, shuffle=False, **kw,
    )
    print(f"Train: {n_train}  Val: {len(X_lags) - n_train}")
    return tr_loader, val_loader


# ------ Training ------

def train_model(model: WindTransformerMDN, tr_loader, val_loader):
    opt          = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_path    = FITTED_DIR / "wind_transformer_best.pt"
    best_val     = float("inf")
    best_epoch   = 0
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for lags_b, cal_b, y_b in tr_loader:
            lags_b = lags_b.to(device)
            cal_b  = cal_b.to(device)
            y_b    = y_b.squeeze(-1).to(device)
            opt.zero_grad()
            pi, mu, sigma = model(lags_b, cal_b)
            loss = crps_gaussian_mixture(pi, mu, sigma, y_b)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(lags_b)
        tr_loss /= len(tr_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lags_b, cal_b, y_b in val_loader:
                lags_b = lags_b.to(device)
                cal_b  = cal_b.to(device)
                y_b    = y_b.squeeze(-1).to(device)
                pi, mu, sigma = model(lags_b, cal_b)
                val_loss += crps_gaussian_mixture(pi, mu, sigma, y_b).item() * len(lags_b)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:3d}/{EPOCHS}  train_crps={tr_loss:.4f}  val_crps={val_loss:.4f}")

        if val_loss < best_val:
            best_val     = val_loss
            best_epoch   = epoch
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stopping at epoch {epoch}  (patience={PATIENCE})")
                break

    print(f"\nBest val CRPS={best_val:.4f} at epoch {best_epoch}  (saved to {best_path})")
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, best_val, best_epoch


# ------ Transformer simulation ------

def _run_transformer_simulation(
    model, norm_stats, k_lags, K, mapper, buf, n_hours, burn_in, wind_scale, rng
):
    model.eval()
    norm_floor = float(-norm_stats["mean"] / norm_stats["std"])

    def _step(ts):
        cal = np.array([
            np.sin(2 * np.pi * ts.hour  / 24),
            np.cos(2 * np.pi * ts.hour  / 24),
            np.sin(2 * np.pi * ts.month / 12),
            np.cos(2 * np.pi * ts.month / 12),
        ], dtype=np.float32)
        lags_arr = np.array(list(reversed(buf)), dtype=np.float32)
        lags_t   = torch.from_numpy(lags_arr).unsqueeze(0).to(device)
        cal_t    = torch.from_numpy(cal).unsqueeze(0).to(device)
        pi, mu, sigma = model(lags_t, cal_t)
        pi_np  = pi.squeeze(0).cpu().numpy()
        mu_np  = mu.squeeze(0).cpu().numpy()
        sg_np  = sigma.squeeze(0).cpu().numpy()
        k      = int(rng.choice(K, p=pi_np))
        sample = float(rng.normal(mu_np[k], sg_np[k]))
        return max(sample, norm_floor)

    burn_idx = pd.date_range(
        pd.Timestamp("2029-01-01", tz="Europe/Copenhagen"), periods=burn_in, freq="h"
    )
    synth_index = pd.date_range(
        pd.Timestamp("2030-01-01", tz="Europe/Copenhagen"), periods=n_hours, freq="h"
    )

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

    wind_raw = (
        np.maximum(
            np.array(results, dtype=np.float32) * norm_stats["std"] + norm_stats["mean"], 0.0
        ) * wind_scale
    )
    sim_df = pd.DataFrame({WIND_COL: wind_raw}, index=synth_index)
    sim_df[WIND_COL] = mapper.transform(sim_df[WIND_COL])
    return sim_df


def run_transformer_paths(model, norm_stats, k_lags, K, mapper, pool, n_paths, n_hours):
    rng   = np.random.default_rng(RANDOM_SEED)
    paths = []
    for i in range(n_paths):
        max_start = max(len(pool) - k_lags - 1, 1)
        start_idx = int(rng.integers(0, max_start))
        buf       = list(pool[start_idx : start_idx + k_lags])
        sim       = _run_transformer_simulation(
            model, norm_stats, k_lags, K, mapper, buf, n_hours, BURN_IN, 1.0, rng
        )
        col = sim[WIND_COL]
        print(f"  Path {i:2d}  mean={col.mean():.3f} m/s  std={col.std():.3f} m/s")
        paths.append(sim)
    return paths


def simulate(n_hours: int = SIM_HOURS, burn_in: int = BURN_IN,
             wind_scale: float = 1.0, seed=None):
    pkl_path = FITTED_DIR / "wind_transformer.pkl"
    with open(pkl_path, "rb") as f:
        fitted = pickle.load(f)
    norm_stats = fitted["norm_stats"]
    mapper     = fitted["quantile_mapper"]
    hp         = fitted["hparams"]

    model = WindTransformerMDN(**{k: hp[k] for k in [
        "k_lags", "d_model", "n_heads", "n_layers",
        "dim_feedforward", "dropout", "K", "log_std_min", "log_std_max",
    ]}).to(device)
    best_path = Path(fitted["best_model_path"])
    if not best_path.exists():
        best_path = FITTED_DIR / best_path.name
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    df   = pd.read_parquet(ERA5_PATH).tz_convert("Europe/Copenhagen").sort_index()
    df   = df.loc[TRAIN_START:TRAIN_END, [WIND_COL]].dropna()
    pool = ((df[WIND_COL].values - norm_stats["mean"]) / norm_stats["std"]).astype(np.float32)

    k_lags    = hp["k_lags"]
    rng       = np.random.default_rng(seed)
    max_start = max(len(pool) - k_lags - 1, 1)
    start_idx = int(rng.integers(0, max_start))
    buf       = list(pool[start_idx : start_idx + k_lags])

    return _run_transformer_simulation(
        model, norm_stats, k_lags, hp["K"], mapper, buf, n_hours, burn_in, wind_scale, rng
    )


# ------ Metrics ------

def compute_metrics(series: pd.Series) -> dict:
    vals  = series.values
    acf_v = sm_acf(vals, nlags=48, fft=True)
    shape, _, _ = scipy.stats.weibull_min.fit(np.clip(vals, 1e-3, None), floc=0)
    monthly_mean = series.groupby(series.index.month).mean()
    rho, _ = scipy.stats.spearmanr(monthly_mean.index, monthly_mean.values)
    return {
        "mean (m/s)":       float(np.mean(vals)),
        "std (m/s)":        float(np.std(vals)),
        "Weibull k":        float(shape),
        "ACF lag-24h":      float(acf_v[24]),
        "ACF lag-48h":      float(acf_v[48]),
        "% above 12 m/s":  float(100 * np.mean(vals > 12.0)),
        "Monthly Spearman": float(rho),
    }


def _agg_paths(paths: list) -> dict:
    all_m = [compute_metrics(p[WIND_COL]) for p in paths]
    return {k: float(np.mean([m[k] for m in all_m])) for k in all_m[0]}


# ------ ARMA comparison helpers ------

def _arma_unstandardise(std_resid: pd.Series, monthly_std: dict) -> pd.Series:
    return std_resid * std_resid.index.month.map(monthly_std).values


def _run_arma_paths(n_paths: int, n_hours: int) -> list:
    _apply_patsy_patch()
    with open(FITTED_DIR / "horns_rev_arma.pkl", "rb") as f:
        fitted = pickle.load(f)
    mean_model  = fitted["mean_model"]
    monthly_std = fitted["monthly_std"]
    arma_model  = fitted["arma_model"]

    rng_a     = np.random.default_rng(RANDOM_SEED)
    synth_idx = pd.date_range(
        "2030-01-01", periods=n_hours, freq="h", tz="Europe/Copenhagen"
    )
    synth_df = pd.DataFrame(
        {WIND_COL: 0.0, "month": synth_idx.month, "hour": synth_idx.hour},
        index=synth_idx,
    )
    seasonal_mean = mean_model.predict(synth_df)

    paths = []
    for i in range(n_paths):
        sim_resid = arma_model.simulate(
            nsimulations=n_hours + BURN_IN,
            random_state=int(rng_a.integers(int(1e9))),
        )[BURN_IN:]
        resid    = _arma_unstandardise(
            pd.Series(sim_resid, index=synth_df.index), monthly_std
        )
        wind_sim = np.abs(seasonal_mean.values + resid.values)
        paths.append(pd.DataFrame({WIND_COL: wind_sim}, index=synth_idx))
        print(f"  ARMA path {i:2d}  mean={wind_sim.mean():.3f} m/s  std={wind_sim.std():.3f} m/s")
    return paths


# ------ Legacy MDN comparison helpers ------

_ACT_MAP = {"relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU}


class _LegacyMDN(nn.Module):
    def __init__(self, input_dim, hidden_dims, K, activation="relu", dropout=0.0):
        super().__init__()
        layers, in_d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), _ACT_MAP[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = h
        self.backbone = nn.Sequential(*layers)
        self.head     = nn.Linear(in_d, K * 3)
        self.K        = K

    def forward(self, x):
        h   = self.backbone(x)
        out = self.head(h)
        K       = self.K
        logits  = out[:, :K]
        mu      = out[:, K : 2 * K].unsqueeze(-1)
        log_std = out[:, 2 * K :].unsqueeze(-1).clamp(-6.0, 2.0)
        return torch.softmax(logits, -1), mu, log_std


def _load_legacy_mdn(pkl_name: str):
    with open(FITTED_DIR / pkl_name, "rb") as f:
        fitted = pickle.load(f)
    hp    = fitted["hparams"]
    model = _LegacyMDN(
        hp["input_dim"], hp["hidden"], hp["K"],
        activation=hp.get("activation", "relu"),
        dropout=hp.get("dropout", 0.0),
    ).to(device)
    stored  = Path(fitted["best_model_path"])
    pt_path = stored if stored.exists() else FITTED_DIR / stored.name
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()
    nll_key = "val_nll_best" if "val_nll_best" in fitted else "crps_val_best"
    print(f"  {pkl_name}: K={hp['K']}, hidden={hp['hidden']}, "
          f"k_lags={hp['k_lags']}, val={fitted.get(nll_key, float('nan')):.4f}")
    return model, fitted["norm_stats"], hp["k_lags"], hp["K"]


def _run_mdn_paths(model, norm_stats, pool, k_lags, K, n_paths, n_hours) -> list:
    rng        = np.random.default_rng(RANDOM_SEED)
    norm_floor = float(-norm_stats["mean"] / norm_stats["std"])
    synth_idx  = pd.date_range(
        "2030-01-01", periods=n_hours, freq="h", tz="Europe/Copenhagen"
    )
    paths = []
    model.eval()

    for i in range(n_paths):
        max_start = max(len(pool) - k_lags - 1, 1)
        start_idx = int(rng.integers(0, max_start))
        buf       = list(pool[start_idx : start_idx + k_lags])
        results   = []

        with torch.no_grad():
            for ts in synth_idx:
                cal = np.array([
                    np.sin(2 * np.pi * ts.hour  / 24),
                    np.cos(2 * np.pi * ts.hour  / 24),
                    np.sin(2 * np.pi * ts.month / 12),
                    np.cos(2 * np.pi * ts.month / 12),
                ], dtype=np.float32)
                lags_arr = np.array(list(reversed(buf)), dtype=np.float32)
                x_t      = torch.from_numpy(
                    np.concatenate([cal, lags_arr])
                ).unsqueeze(0).to(device)
                pi, mu, log_std = model(x_t)
                pi_np  = pi.squeeze(0).cpu().numpy()
                mu_np  = mu.squeeze(0).cpu().numpy()           # (K, 1)
                std_np = log_std.squeeze(0).exp().cpu().numpy()  # (K, 1)
                k      = int(rng.choice(K, p=pi_np))
                s      = float(rng.normal(mu_np[k, 0], std_np[k, 0]))
                s      = max(s, norm_floor)
                buf.append(s)
                buf.pop(0)
                results.append(s)

        wind = np.maximum(
            np.array(results, dtype=np.float32) * norm_stats["std"] + norm_stats["mean"], 0.0
        )
        paths.append(pd.DataFrame({WIND_COL: wind}, index=synth_idx))
        print(f"  MDN path {i:2d}  mean={wind.mean():.3f} m/s  std={wind.std():.3f} m/s")

    return paths


# ------ Comparison table ------

def print_comparison_table(obs, arma_paths, r4b_paths, r6_paths, transformer_paths):
    rows = [
        ("Observed",    compute_metrics(obs)),
        ("ARMA",        _agg_paths(arma_paths)),
        ("MDN r4b",     _agg_paths(r4b_paths)),
        ("MDN r6",      _agg_paths(r6_paths)),
        ("Transformer", _agg_paths(transformer_paths)),
    ]
    keys   = list(rows[0][1].keys())
    col_w  = 13
    header = f"{'Metric':<22}" + "".join(f"{label:>{col_w}}" for label, _ in rows)
    print(header)
    print("-" * len(header))
    for key in keys:
        row = f"{key:<22}" + "".join(
            f"{metrics[key]:>{col_w}.4f}" for _, metrics in rows
        )
        print(row)


# ------ Validation plots ------

def plot_kde(observed: pd.Series, sim_paths: list, outdir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    observed.plot.kde(ax=ax, color="black", linewidth=2, label="Observed")
    for sim in sim_paths:
        sim[WIND_COL].plot.kde(ax=ax, color="steelblue", alpha=0.4,
                               linewidth=0.9, label="_nolegend_")
    ax.plot([], [], color="steelblue", alpha=0.6, linewidth=0.9,
            label=f"Simulated (n={len(sim_paths)})")
    ax.set_xlim(left=-1)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_title("KDE — Horns Rev wind speed: observed vs Transformer simulated")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = outdir / "kde_wind_speeds_transformer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_acf_comparison(observed: pd.Series, sim_paths: list, outdir: Path):
    nlags   = 48
    lags    = np.arange(nlags + 1)
    obs_acf = sm_acf(observed.values, nlags=nlags, fft=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, sim in enumerate(sim_paths):
        sim_acf = sm_acf(sim[WIND_COL].values, nlags=nlags, fft=True)
        ax.plot(lags, sim_acf, color="steelblue", alpha=0.5, linewidth=0.9,
                label="Simulated" if i == 0 else None)
    ax.plot(lags, obs_acf, color="black", linewidth=2, label="Observed")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ACF")
    ax.set_title("ACF — Horns Rev wind speed: observed vs Transformer simulated (0-48 h)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = outdir / "acf_wind_speed_transformer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_diurnal(observed: pd.Series, sim_paths: list, outdir: Path):
    hours   = np.arange(24)
    seasons = [
        ("Annual", None,       "black",     "dimgray"),
        ("Winter", [12, 1, 2], "steelblue", "steelblue"),
        ("Summer", [6, 7, 8],  "tomato",    "tomato"),
    ]

    def diurnal(s, months=None):
        if months:
            s = s[s.index.month.isin(months)]
        return s.groupby(s.index.hour).mean().values

    fig, ax = plt.subplots(figsize=(9, 4))
    for label, months, obs_col, sim_col in seasons:
        obs_vals = diurnal(observed, months)
        sim_mat  = np.stack([diurnal(s[WIND_COL], months) for s in sim_paths])
        ax.plot(hours, obs_vals,        color=obs_col, lw=2,   ls="-",
                label=f"Obs {label}")
        ax.plot(hours, sim_mat.mean(0), color=sim_col, lw=1.5, ls="--",
                label=f"Sim {label}")
    ax.set_xlabel("Hour of day (Copenhagen time)")
    ax.set_ylabel("m/s")
    ax.set_title("Diurnal cycle — Horns Rev: observed vs Transformer simulated")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "diurnal_wind_transformer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ------ Main ------

def main():
    print("=" * 60)
    print("Wind Transformer MDN -- Horns Rev 10m wind speed")
    print("Causal Transformer + CRPS + Weibull quantile mapping")
    print("=" * 60)

    print("\n------ Data loading & normalisation ------")
    df, norm_stats = load_and_normalise(ERA5_PATH)
    pool = df[WIND_COL + "_norm"].values.astype(np.float32)

    print("\n------ Fitting WindQuantileMapper (24 bins) ------")
    mapper = WindQuantileMapper().fit(df[WIND_COL])

    print("\n------ Feature construction ------")
    X_lags, X_cal, y = build_features(df, K_LAGS)

    print("\n------ DataLoaders ------")
    tr_loader, val_loader = make_loaders(X_lags, X_cal, y)

    print("\n------ Model initialisation ------")
    model    = WindTransformerMDN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Architecture: k_lags={K_LAGS}, d_model={D_MODEL}, n_heads={N_HEADS}, "
          f"n_layers={N_LAYERS}, ff={DIM_FF}, dropout={DROPOUT}, K={K_COMPONENTS}")

    print("\n------ Training (CRPS loss) ------")
    model, best_val, best_epoch = train_model(model, tr_loader, val_loader)

    print(f"\n------ Transformer simulation ({N_SIM_PATHS} paths x {SIM_HOURS} h) ------")
    transformer_paths = run_transformer_paths(
        model, norm_stats, K_LAGS, K_COMPONENTS, mapper, pool, N_SIM_PATHS, SIM_HOURS
    )

    print("\n------ Saving fitted objects ------")
    best_model_path = str((FITTED_DIR / "wind_transformer_best.pt").resolve())
    fitted_out = {
        "norm_stats": norm_stats,
        "hparams": {
            "k_lags":          K_LAGS,
            "d_model":         D_MODEL,
            "n_heads":         N_HEADS,
            "n_layers":        N_LAYERS,
            "dim_feedforward": DIM_FF,
            "dropout":         DROPOUT,
            "K":               K_COMPONENTS,
            "log_std_min":     LOG_STD_MIN,
            "log_std_max":     LOG_STD_MAX,
        },
        "quantile_mapper": mapper,
        "best_model_path": best_model_path,
        "training_period": f"{TRAIN_START} to {TRAIN_END}",
        "val_crps_best":   best_val,
        "feature_columns": [WIND_COL],
    }
    pkl_path = FITTED_DIR / "wind_transformer.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(fitted_out, f)
    print(f"Saved: {pkl_path}")
    print(f"Saved: {best_model_path}")

    print("\n------ Validation plots ------")
    plot_kde(df[WIND_COL], transformer_paths, OUTDIR)
    plot_acf_comparison(df[WIND_COL], transformer_paths, OUTDIR)
    plot_diurnal(df[WIND_COL], transformer_paths, OUTDIR)

    print(f"\n------ ARMA simulation ({N_SIM_PATHS} paths x {SIM_HOURS} h) ------")
    arma_paths = _run_arma_paths(N_SIM_PATHS, SIM_HOURS)

    print(f"\n------ MDN r4b simulation ({N_SIM_PATHS} paths x {SIM_HOURS} h) ------")
    r4b_model, r4b_norm, r4b_klags, r4b_K = _load_legacy_mdn("wind_mdn_tuned_r4b.pkl")
    r4b_pool  = (
        (df[WIND_COL].values - r4b_norm["mean"]) / r4b_norm["std"]
    ).astype(np.float32)
    r4b_paths = _run_mdn_paths(
        r4b_model, r4b_norm, r4b_pool, r4b_klags, r4b_K, N_SIM_PATHS, SIM_HOURS
    )

    print(f"\n------ MDN r6 simulation ({N_SIM_PATHS} paths x {SIM_HOURS} h) ------")
    r6_model, r6_norm, r6_klags, r6_K = _load_legacy_mdn("wind_mdn_tuned_r6.pkl")
    r6_pool   = (
        (df[WIND_COL].values - r6_norm["mean"]) / r6_norm["std"]
    ).astype(np.float32)
    r6_paths  = _run_mdn_paths(
        r6_model, r6_norm, r6_pool, r6_klags, r6_K, N_SIM_PATHS, SIM_HOURS
    )

    print("\n------ Comparison table ------")
    print_comparison_table(df[WIND_COL], arma_paths, r4b_paths, r6_paths, transformer_paths)

    print("\nDone.")


if __name__ == "__main__":
    main()
