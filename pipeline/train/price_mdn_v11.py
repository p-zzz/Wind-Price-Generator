"""
Price MDN v11: ratio-to-load features replace solar_cf / wind_gen_ratio.

Base: price_mdn_v10.py (wind_gen_ratio fix). Single further change:
solar_cf and wind_gen_ratio (both installed-capacity-normalised, and by
construction completely invariant to SOLAR_SCALE/WIND_SCALE) are replaced by
solar_load_ratio = solar_MW / actual_load_MW and
wind_load_ratio  = wind_MW  / actual_load_MW, where solar_MW / wind_MW are
the same clean, gap-imputed generation series used to build v10's capacity
factors (solar_cf_final x solar_capacity, and the onshore/offshore capacity-
weighted wind blend), just left in MW rather than normalised by capacity.

Motivation (generator_validation.md sec 11.9-11.10): v10's pure
capacity-factor reframing fixed the z~40 extrapolation bug, but it also made
solar_cf/wind_gen_ratio completely scale-invariant BY CONSTRUCTION -- neither
SOLAR_SCALE nor WIND_SCALE can move price at all under v10, which defeats
the generator's stated purpose (cannibalisation-risk analysis under
different renewable-penetration futures). load is a much more stable-scale
denominator than installed capacity (which is exactly what the scenario
knobs vary), so solar_MW/load and wind_MW/load should respond to the scale
knobs without reintroducing the raw-MW zero-fill contamination that broke
net_load_MW (this still uses the same clean, gap-imputed generation numerator
as v10's capacity factors, not the contaminated dk1_features_hourly columns).

Validated numerically prior to this training run (generator_validation.md
sec 11.9): historical (2015-2025) solar_MW/load mean=0.0631 std=0.0989
max=0.9871 (max z=9.34); wind_MW/load mean=0.6107 std=0.3867 max=3.3073
(max z=6.97). Scenario z-scores against that historical distribution: 1x
scale sits at z=7.29 (solar) / z=5.04 (wind) -- within the historical range;
2x reaches z=15.21 / z=11.67 and target reaches z=28.69 / z=12.55 -- both
extrapolate well beyond anything ever observed. This is a partial fix, not
a full one: only the 1x (today's actual capacity) scenario is genuinely
in-domain.

Everything else (architecture, oversampling, regime-stratified split,
training procedure) is identical to v10.

Fitted objects:
  models/price/fitted/price_mdn_v11.pkl
  models/price/fitted/price_mdn_v11_best.pt
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

plt.style.use("seaborn-v0_8")

# ------ Patsy unpickling patch (needed to load wind_cf_*_arma.pkl / solar CF OLS objects) ------

import patsy.eval as _pe

_orig_capture = _pe.EvalEnvironment.capture.__func__


@classmethod
def _patched_capture(cls, eval_env=0, reference=0):
    try:
        return _orig_capture(cls, eval_env, reference)
    except Exception:
        return cls({}, {})


_pe.EvalEnvironment.capture = _patched_capture

FEATURES_PATH        = Path("DATA/processed/dk1_features_hourly.parquet")
ERA5_PATH            = Path("DATA/processed/era5_sites_wind_hourly.parquet")
GAS_PATH             = Path("DATA/raw/gas/Dutch TTF Natural Gas Futures Historical Data.csv")
RAW_GEN_PATH         = Path("DATA/raw/generation/dk1_generation_by_type_wide.parquet")
CAPACITY_PATH        = Path("DATA/raw/capacity_per_municipality/dk1_wind_capacity_monthly.parquet")
SOLAR_CF_PKL_PATH    = Path("models/exog/fitted/solar_model_capacity_factor.pkl")
WIND_CF_ONSHORE_PKL  = Path("models/wind/fitted/wind_cf_onshore_arma.pkl")
WIND_CF_OFFSHORE_PKL = Path("models/wind/fitted/wind_cf_offshore_arma.pkl")
OUTDIR               = Path("analysis/outputs/DK1/price_model")
FITTED_DIR           = Path("models/price/fitted")
OUTDIR.mkdir(parents=True, exist_ok=True)
FITTED_DIR.mkdir(parents=True, exist_ok=True)

WIND_COL           = "horns_rev_wind_speed_10m_ms"
PRICE_COL          = "day_ahead_price"
GAS_COL            = "ttf_gas_price_eur_mwh"
SOLAR_LOAD_COL     = "solar_load_ratio"
WIND_LOAD_COL      = "wind_load_ratio"
CONT_COLS    = [WIND_COL, SOLAR_LOAD_COL, "actual_load_MW", "net_position_MW",
                GAS_COL, WIND_LOAD_COL]

TRAIN_START     = "2015-01-01"
TRAIN_END       = "2025-12-31"
GAS_FILL_BEFORE = pd.Timestamp("2018-03-01")
GAS_FILL_VALUE  = 20.0
LOAD_FLOOR_MW   = 10.0

K_COMPONENTS = 6
HIDDEN_DIMS  = [64, 64, 32]
INPUT_DIM    = 11
LOG_STD_MIN  = -4.0
LOG_STD_MAX  = 6.0
BATCH_SIZE   = 512
EPOCHS       = 200
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 20
OVERSAMPLE_WEIGHT = 3.5
N_SIM_PATHS  = 10
# Seeds weight init + the oversampling sampler (the train/val split is chronological,
# so already deterministic). Val NLL varies ~0.1 across seeds, so the shipped model
# was chosen as the best val NLL of seeds {1, 2, 3, 4, 42} on signed net position;
# the default reproduces it. Override with PRICE_MDN_SEED=<int> to try others.
SEED = int(os.environ.get("PRICE_MDN_SEED", 2))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

LOG_2PI = float(np.log(2 * np.pi))


# ------ Gas price loading ------

def load_gas_prices(start: str, end: str) -> pd.Series:
    df = pd.read_csv(GAS_PATH, usecols=["Date", "Price"])
    df["Date"]  = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"]).sort_values("Date").set_index("Date")

    daily_idx = pd.date_range(start, end, freq="D")
    s = df["Price"].reindex(daily_idx).ffill().fillna(GAS_FILL_VALUE)
    s.loc[s.index < GAS_FILL_BEFORE] = GAS_FILL_VALUE

    hourly_idx = pd.date_range(start, end, freq="h")
    s_hourly   = s.resample("h").ffill().reindex(hourly_idx).ffill()
    s_hourly   = s_hourly.tz_localize("UTC").tz_convert("Europe/Copenhagen")
    s_hourly.name = GAS_COL
    print(f"Gas prices: {len(s_hourly)} hourly obs  min={s_hourly.min():.1f}  "
          f"max={s_hourly.max():.1f}  mean={s_hourly.mean():.1f} EUR/MWh")
    return s_hourly


# ------ Solar generation MW (gap-aware, shape-imputed, capacity-rescaled) ------

def load_hourly_capacity_interp(col: str, index: pd.DatetimeIndex) -> pd.Series:
    cap = pd.read_parquet(CAPACITY_PATH)[col]
    cap.index = cap.index.tz_localize("Europe/Copenhagen")
    union_idx = index.union(cap.index)
    cap_hourly = cap.reindex(union_idx).sort_index().interpolate(method="time")
    return cap_hourly.reindex(index).bfill()


def build_solar_MW(target_index: pd.DatetimeIndex) -> pd.Series:
    gen = pd.read_parquet(RAW_GEN_PATH)
    gen = gen.tz_convert("Europe/Copenhagen").sort_index()
    solar_h = gen[["gen_solar_MW"]].resample("h").mean()["gen_solar_MW"]
    full_idx = pd.date_range(solar_h.index.min(), solar_h.index.max(),
                              freq="h", tz="Europe/Copenhagen")
    solar_h = solar_h.reindex(full_idx)
    cap_hourly = load_hourly_capacity_interp("SolarPowerCapacity", full_idx)

    with open(SOLAR_CF_PKL_PATH, "rb") as f:
        cf_pkl = pickle.load(f)
    day_pairs  = cf_pkl["day_pairs"]
    mean_model = cf_pkl["mean_model"]
    day_mask = np.array([(m, h) in day_pairs for m, h in zip(full_idx.month, full_idx.hour)])

    solar_cf = np.zeros(len(full_idx), dtype=np.float64)
    solar_cf[day_mask] = solar_h.values[day_mask] / cap_hourly.values[day_mask]

    missing = day_mask & np.isnan(solar_cf)
    pred_df = pd.DataFrame({"month": full_idx.month[missing], "hour": full_idx.hour[missing]})
    solar_cf[missing] = np.maximum(mean_model.predict(pred_df).values, 0.0)

    n_missing = int(missing.sum())
    print(f"solar_MW: {n_missing} / {len(full_idx)} hours "
          f"({100 * n_missing / len(full_idx):.1f}%) gap-imputed from CF shape model; "
          f"rest genuinely reported or night(=0)")

    solar_mw = solar_cf * cap_hourly.values
    s = pd.Series(solar_mw, index=full_idx, name="solar_MW")
    return s.reindex(target_index)


# ------ Wind generation MW (gap-aware, onshore/offshore independent, capacity-rescaled) ------

def build_wind_MW(target_index: pd.DatetimeIndex) -> pd.Series:
    sys.path.insert(0, str(Path("models/wind")))
    from wind_capacity_factor_arma import WindCFQuantileMapper
    sys.modules["__main__"].WindCFQuantileMapper = WindCFQuantileMapper

    gen = pd.read_parquet(RAW_GEN_PATH)
    gen = gen.tz_convert("Europe/Copenhagen").sort_index()
    onshore_h  = gen["gen_wind_onshore_MW"].resample("h").mean()
    offshore_h = gen["gen_wind_offshore_MW"].resample("h").mean()
    full_idx = pd.date_range(min(onshore_h.index.min(), offshore_h.index.min()),
                              max(onshore_h.index.max(), offshore_h.index.max()),
                              freq="h", tz="Europe/Copenhagen")
    onshore_h  = onshore_h.reindex(full_idx)
    offshore_h = offshore_h.reindex(full_idx)

    onshore_cap  = load_hourly_capacity_interp("OnshoreWindCapacity", full_idx)
    offshore_cap = load_hourly_capacity_interp("OffshoreWindCapacity", full_idx)

    onshore_cf  = (onshore_h / onshore_cap).clip(lower=0.0, upper=1.0)
    offshore_cf = (offshore_h / offshore_cap).clip(lower=0.0, upper=1.0)

    era5 = pd.read_parquet(ERA5_PATH, columns=[WIND_COL])
    era5 = era5.tz_convert("Europe/Copenhagen").sort_index()
    wind_speed_full = era5[WIND_COL].reindex(full_idx)
    pred_df = pd.DataFrame({"month": full_idx.month, "hour": full_idx.hour,
                             WIND_COL: wind_speed_full.values}, index=full_idx)

    with open(WIND_CF_ONSHORE_PKL, "rb") as f:
        onshore_pkl = pickle.load(f)
    with open(WIND_CF_OFFSHORE_PKL, "rb") as f:
        offshore_pkl = pickle.load(f)

    onshore_pred  = np.clip(onshore_pkl["mean_model"].predict(pred_df).values, 0.0, 1.0)
    offshore_pred = np.clip(offshore_pkl["mean_model"].predict(pred_df).values, 0.0, 1.0)

    onshore_cf_final = onshore_cf.copy()
    missing_o = onshore_cf_final.isna()
    onshore_cf_final[missing_o] = onshore_pred[missing_o.values]

    offshore_cf_final = offshore_cf.copy()
    missing_off = offshore_cf_final.isna()
    offshore_cf_final[missing_off] = offshore_pred[missing_off.values]

    n_imputed = int(missing_o.sum() + missing_off.sum())
    print(f"wind_MW: {n_imputed} onshore/offshore gap-hours gap-imputed from "
          f"existing wind CF ARMA shape models (mean_model component)")

    wind_mw = onshore_cf_final * onshore_cap + offshore_cf_final * offshore_cap
    wind_mw.name = "wind_MW"
    return wind_mw.reindex(target_index)


# ------ Data loading & normalisation ------

def load_and_merge() -> pd.DataFrame:
    feat = pd.read_parquet(FEATURES_PATH)
    feat = feat.tz_convert("Europe/Copenhagen").sort_index()

    era5 = pd.read_parquet(ERA5_PATH)
    era5 = era5.tz_convert("Europe/Copenhagen").sort_index()[[WIND_COL]]

    gas = load_gas_prices(TRAIN_START, TRAIN_END)

    df = feat.join(era5, how="inner").join(gas, how="left")
    df = df.loc[TRAIN_START:TRAIN_END]

    df["solar_MW"] = build_solar_MW(df.index)
    df["wind_MW"]  = build_wind_MW(df.index)

    n_degenerate_load = int((df["actual_load_MW"] < LOAD_FLOOR_MW).sum())
    print(f"Dropping {n_degenerate_load} hour(s) with actual_load_MW < {LOAD_FLOOR_MW} MW "
          f"(reporting artifact -- DK1 demand is never genuinely near-zero) before "
          f"computing ratio-to-load features, to avoid inf/NaN propagation")
    df = df[df["actual_load_MW"] >= LOAD_FLOOR_MW]

    df[SOLAR_LOAD_COL] = df["solar_MW"] / df["actual_load_MW"]
    df[WIND_LOAD_COL]  = df["wind_MW"] / df["actual_load_MW"]

    keep = [PRICE_COL] + CONT_COLS
    df   = df[keep].dropna()
    print(f"Merged dataset: {len(df)} obs  ({df.index[0]} .. {df.index[-1]})")
    return df


def normalise(df: pd.DataFrame):
    norm_stats = {}
    for col in CONT_COLS:
        mean_v = float(df[col].mean())
        std_v  = float(df[col].std())
        df[col + "_norm"] = (df[col] - mean_v) / std_v
        norm_stats[col]   = {"mean": mean_v, "std": std_v}
        print(f"  {col}: mean={mean_v:.4f}  std={std_v:.4f}")
    return df, norm_stats


# ------ Feature construction ------

def _calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    hour  = index.hour
    month = index.month
    return np.column_stack([
        np.sin(2 * np.pi * hour  / 24),
        np.cos(2 * np.pi * hour  / 24),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
    ]).astype(np.float32)


def build_features(df: pd.DataFrame):
    cal         = _calendar_features(df.index)
    cont        = np.column_stack([df[c + "_norm"].values for c in CONT_COLS]).astype(np.float32)
    interaction = (cont[:, 0] * cont[:, 1]).reshape(-1, 1)
    X           = np.concatenate([cal, cont, interaction], axis=1)
    y           = df[PRICE_COL].values.astype(np.float32)
    print(f"Feature matrix: X={X.shape}  y={y.shape}")
    return X, y


# ------ MDN model ------

class MDN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, K: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim  = h
        self.backbone = nn.Sequential(*layers)
        self.head     = nn.Linear(in_dim, K * 3)
        self.K = K

    def forward(self, x):
        h   = self.backbone(x)
        out = self.head(h)
        K       = self.K
        logits  = out[:, :K]
        means   = out[:, K : 2 * K]
        log_std = out[:, 2 * K :].clamp(LOG_STD_MIN, LOG_STD_MAX)
        pi      = torch.softmax(logits, dim=-1)
        return pi, means, log_std


# ------ NLL loss ------

def mdn_nll(pi, means, log_std, y):
    log_pi  = torch.log(pi + 1e-8)
    log_p   = (-0.5 * LOG_2PI - log_std
               - 0.5 * ((y.unsqueeze(1) - means) / log_std.exp()) ** 2)
    log_mix = torch.logsumexp(log_pi + log_p, dim=-1)
    return -log_mix.mean()


# ------ Training ------

def _regime(year: int) -> str:
    if year <= 2021:
        return "pre_crisis"
    if year == 2022:
        return "crisis"
    return "post_crisis"


def make_loaders(X, y, index: pd.DatetimeIndex):
    # Regime-stratified 80/20 split -- identical to price_mdn_v10.make_loaders(),
    # see that file's docstring/comment for the crisis-year rationale.
    regimes = np.array([_regime(yr) for yr in index.year])
    train_mask = np.zeros(len(X), dtype=bool)
    val_mask   = np.zeros(len(X), dtype=bool)
    for reg in np.unique(regimes):
        reg_idx = np.where(regimes == reg)[0]
        n_reg_train = int(len(reg_idx) * 0.8)
        train_mask[reg_idx[:n_reg_train]] = True
        val_mask[reg_idx[n_reg_train:]]   = True

    X_tr, X_val = X[train_mask], X[val_mask]
    y_tr, y_val = y[train_mask], y[val_mask]
    train_idx   = index[train_mask]
    val_idx     = index[val_mask]

    weights      = np.where(train_idx.year >= 2023, OVERSAMPLE_WEIGHT, 1.0).astype(np.float32)
    sampler      = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )
    kw = dict(num_workers=4, pin_memory=True) if device.type == "cuda" else dict(num_workers=0)
    tr_loader  = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=BATCH_SIZE, sampler=sampler, **kw,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE, shuffle=False, **kw,
    )
    n_postcris = int((train_idx.year >= 2023).sum())
    print(f"Train: {len(X_tr)}  Val: {len(X_val)}  "
          f"post-crisis rows: {n_postcris} (weight={OVERSAMPLE_WEIGHT}), "
          f"rest: {len(X_tr)-n_postcris} (weight=1.0)")
    for reg in ["pre_crisis", "crisis", "post_crisis"]:
        n_tr  = int((np.array([_regime(yr) for yr in train_idx.year]) == reg).sum())
        n_val = int((np.array([_regime(yr) for yr in val_idx.year]) == reg).sum())
        print(f"  regime {reg:12s}  train={n_tr:6d}  val={n_val:6d}")
    return tr_loader, val_loader


def train_model(model, tr_loader, val_loader):
    opt            = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_path      = FITTED_DIR / "price_mdn_v11_best.pt"
    best_val       = float("inf")
    best_epoch     = 0
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = mdn_nll(*model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(tr_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += mdn_nll(*model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:3d}/{EPOCHS}  train_nll={tr_loss:.4f}  val_nll={val_loss:.4f}")

        if val_loss < best_val:
            best_val       = val_loss
            best_epoch     = epoch
            patience_count = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch}  (patience={PATIENCE})")
                break

    print(f"\nBest val NLL={best_val:.4f} at epoch {best_epoch}  (saved to {best_path})")
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, best_val, best_epoch


# ------ Simulation ------

def simulate(inputs_df: pd.DataFrame, gas_price=35.0, seed=None) -> pd.Series:
    pkl_path = FITTED_DIR / "price_mdn_v11.pkl"
    with open(pkl_path, "rb") as f:
        fitted = pickle.load(f)
    norm_stats      = fitted["norm_stats"]
    hp              = fitted["hparams"]
    feature_columns = fitted["feature_columns"]

    model = MDN(hp["input_dim"], hp["hidden"], hp["K"])
    model.load_state_dict(torch.load(fitted["best_model_path"], map_location=device))
    model.to(device)
    model.eval()

    df = inputs_df.copy()
    if isinstance(gas_price, pd.Series):
        df[GAS_COL] = gas_price.reindex(df.index)
    else:
        df[GAS_COL] = float(gas_price)

    cal  = _calendar_features(df.index)
    cont = np.column_stack([
        (df[col].values - norm_stats[col]["mean"]) / norm_stats[col]["std"]
        for col in feature_columns
    ]).astype(np.float32)
    interaction = (cont[:, 0] * cont[:, 1]).reshape(-1, 1)
    X = np.concatenate([cal, cont, interaction], axis=1)

    with torch.no_grad():
        pi, means, log_std = model(torch.from_numpy(X).to(device))
        pi_np    = pi.cpu().numpy()
        means_np = means.cpu().numpy()
        std_np   = log_std.exp().cpu().numpy()

    rng       = np.random.default_rng(seed)
    n         = len(X)
    k_choices = np.array([int(rng.choice(K_COMPONENTS, p=pi_np[i])) for i in range(n)])
    sel_means = means_np[np.arange(n), k_choices]
    sel_stds  = std_np[np.arange(n), k_choices]
    samples   = rng.normal(sel_means, sel_stds).astype(np.float32)

    return pd.Series(samples, index=inputs_df.index, name=PRICE_COL)


# ------ Main ------

def main():
    print("=" * 60)
    print("Price MDN v11 -- ratio-to-load (solar_MW/load, wind_MW/load), K=6, post-crisis 3.5x oversample")
    print("=" * 60)

    print("\n------ Data loading & merging ------")
    df = load_and_merge()

    print("\n------ Normalisation ------")
    df, norm_stats = normalise(df)

    print("\n------ Feature construction ------")
    X, y = build_features(df)

    print("\n------ Model initialisation ------")
    torch.manual_seed(SEED)
    print(f"Seed: {SEED}")
    model    = MDN(INPUT_DIM, HIDDEN_DIMS, K_COMPONENTS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Architecture: {INPUT_DIM} -> {' -> '.join(str(h) for h in HIDDEN_DIMS)} -> MDN(K={K_COMPONENTS})")

    print("\n------ Training ------")
    tr_loader, val_loader = make_loaders(X, y, df.index)
    model, best_val, best_epoch = train_model(model, tr_loader, val_loader)

    print("\n------ Saving fitted objects ------")
    best_model_path = str((FITTED_DIR / "price_mdn_v11_best.pt").resolve())
    fitted = {
        "norm_stats":      norm_stats,
        "hparams": {
            "K":                K_COMPONENTS,
            "hidden":           HIDDEN_DIMS,
            "input_dim":        INPUT_DIM,
            "interaction":      True,
            "solar_load_ratio": True,
            "wind_load_ratio":  True,
            "oversample_weight": OVERSAMPLE_WEIGHT,
            "seed":              SEED,
            "distribution":     "gaussian",
        },
        "best_model_path":  best_model_path,
        "training_period":  f"{TRAIN_START} to {TRAIN_END}",
        "val_nll_best":     best_val,
        "feature_columns":  CONT_COLS,
        "gas_price_regimes": {
            "source":     str(GAS_PATH),
            "pre_cutoff": str(GAS_FILL_BEFORE.date()),
            "fill_value": GAS_FILL_VALUE,
        },
    }
    pkl_path = FITTED_DIR / "price_mdn_v11.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(fitted, f)
    print(f"Saved: {pkl_path}")
    print(f"Saved: {best_model_path}")

    print(f"\n------ Simulation (N={N_SIM_PATHS} paths, observed inputs) ------")
    obs_inputs = df[[WIND_COL, SOLAR_LOAD_COL, "actual_load_MW",
                     "net_position_MW", WIND_LOAD_COL]].copy()
    obs_gas    = df[GAS_COL]
    sim_prices = []
    for i in range(N_SIM_PATHS):
        s = simulate(obs_inputs, gas_price=obs_gas, seed=i)
        neg_pct = 100 * (s < 0).mean()
        print(f"  Path {i:2d}  mean={s.mean():.1f} EUR/MWh  std={s.std():.1f}  "
              f"neg={neg_pct:.1f}%")
        sim_prices.append(s)

    print("\nDone.")


if __name__ == "__main__":
    main()
