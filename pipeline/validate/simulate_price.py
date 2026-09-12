"""
Generate synthetic price paths for Price MDN v11 -- the only price model this repo
ships. Writes output to DATA/synthetic/price_mdn_v11/price_paths.parquet, read by
validate_price.py.

Trimmed from the thesis repo's analysis/simulate_price.py, which is a multi-model
comparison harness spanning price_flow/v2/v3 and price_mdn v8/v10/v11 -- kept only
the v11 path (this repo's production model) and dropped the price_flow import
(a rejected model architecture) in favour of the handful of plain constants it was
only used for. See ../../.claude notes in the thesis repo (generator_extraction_plan.md)
for why this file exists as a trim rather than a straight copy.

Run from this repo's root, after pipeline/train/price_mdn_v11.py has produced
models/price/fitted/price_mdn_v11.pkl (or point FITTED_DIR at wherever you trained it).
"""

import pickle
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "train"))
import price_mdn_v11 as _v11  # noqa: E402


# ------ Config ------

MODEL_NAME = "price_mdn_v11"
N_PATHS = 25
N_DAYS = 4200  # ~11.5 years of days
SEED = 42

FITTED_DIR = Path("models/price/fitted")
FEATURES_PATH = Path("DATA/processed/dk1_features_hourly.parquet")
ERA5_PATH = Path("DATA/processed/era5_sites_wind_hourly.parquet")
GAS_PATH = Path("DATA/raw/gas/Dutch TTF Natural Gas Futures Historical Data.csv")

OUTPUT_DIR = Path("DATA/synthetic") / MODEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "price_paths.parquet"

_SIM_START = pd.Timestamp("2015-01-01", tz="Europe/Copenhagen")
_DATA_START = "2015-01-01"
_DATA_END = "2026-06-30"

WIND_COL = "horns_rev_wind_speed_10m_ms"
PRICE_COL = "day_ahead_price"
GAS_COL = "ttf_gas_price_eur_mwh"
GAS_FILL_BEFORE = pd.Timestamp("2018-03-01")
GAS_FILL_VALUE = 20.0
LOG_STD_MIN = -4.0
LOG_STD_MAX = 6.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------ Patsy patch ------

import patsy.eval

_orig_capture = patsy.eval.EvalEnvironment.capture.__func__


@classmethod
def _patched_capture(cls, eval_env=0, reference=0):
    try:
        return _orig_capture(cls, eval_env, reference)
    except Exception:
        return cls({}, {})


patsy.eval.EvalEnvironment.capture = _patched_capture


# ------ Simulator base ------


class PriceSimulator(ABC):
    def __init__(self, fitted_dir, n_paths, n_days, seed):
        self.fitted_dir = Path(fitted_dir)
        self.n_paths = n_paths
        self.n_days = n_days
        self.seed = seed
        self._result = None

    @abstractmethod
    def simulate(self) -> pd.DataFrame: ...

    def save(self, path: Path):
        if self._result is None:
            raise RuntimeError("Call simulate() before save()")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._result.to_parquet(path)


# ------ MDN architecture (mirrors generator/price.py) ------


class _MDN(nn.Module):
    def __init__(self, input_dim, hidden_dims, K):
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
        return torch.softmax(logits, dim=-1), means, log_std


def _mdn_calendar(index: pd.DatetimeIndex) -> np.ndarray:
    h = index.hour
    m = index.month
    return np.column_stack(
        [
            np.sin(2 * np.pi * h / 24),
            np.cos(2 * np.pi * h / 24),
            np.sin(2 * np.pi * m / 12),
            np.cos(2 * np.pi * m / 12),
        ]
    ).astype(np.float32)


def _load_data_v11(start: str, end: str) -> pd.DataFrame:
    feat = pd.read_parquet(FEATURES_PATH)
    feat = feat.tz_convert("Europe/Copenhagen").sort_index()

    era5 = pd.read_parquet(ERA5_PATH)
    era5 = era5.tz_convert("Europe/Copenhagen").sort_index()[[WIND_COL]]

    gas_df = pd.read_csv(GAS_PATH, usecols=["Date", "Price"])
    gas_df["Date"] = pd.to_datetime(gas_df["Date"], format="%m/%d/%Y")
    gas_df["Price"] = pd.to_numeric(gas_df["Price"], errors="coerce")
    gas_df = gas_df.dropna(subset=["Price"]).sort_values("Date").set_index("Date")
    daily_idx = pd.date_range(start, end, freq="D")
    gas_s = gas_df["Price"].reindex(daily_idx).ffill().fillna(GAS_FILL_VALUE)
    gas_s.loc[gas_s.index < GAS_FILL_BEFORE] = GAS_FILL_VALUE
    hourly_idx = pd.date_range(start, end, freq="h")
    gas_h = gas_s.resample("h").ffill().reindex(hourly_idx).ffill()
    gas_h = gas_h.tz_localize("UTC").tz_convert("Europe/Copenhagen")
    gas_h.name = GAS_COL

    df = feat.join(era5, how="inner").join(gas_h, how="left")
    df = df.loc[start:end]

    df["solar_MW"] = _v11.build_solar_MW(df.index)
    df["wind_MW"] = _v11.build_wind_MW(df.index)
    df = df[df["actual_load_MW"] >= _v11.LOAD_FLOOR_MW]
    df[_v11.SOLAR_LOAD_COL] = df["solar_MW"] / df["actual_load_MW"]
    df[_v11.WIND_LOAD_COL] = df["wind_MW"] / df["actual_load_MW"]

    df = df[[PRICE_COL] + _v11.CONT_COLS].dropna()
    return df


class MdnV11Simulator(PriceSimulator):
    def __init__(self, fitted_dir, n_paths, n_days, seed):
        super().__init__(fitted_dir, n_paths, n_days, seed)

        pkl_path = self.fitted_dir / f"{MODEL_NAME}.pkl"
        with open(pkl_path, "rb") as f:
            fitted = pickle.load(f)

        self._norm = fitted["norm_stats"]
        self._feat_cols = fitted["feature_columns"]
        hp = fitted["hparams"]

        self._model = _MDN(hp["input_dim"], hp["hidden"], hp["K"]).to(device)
        stored = Path(fitted["best_model_path"])
        pt_path = stored if stored.exists() else self.fitted_dir / stored.name
        self._model.load_state_dict(torch.load(pt_path, map_location=device))
        self._model.eval()
        self._K = hp["K"]

    def simulate(self) -> pd.DataFrame:
        print("  Loading data ...")
        df = _load_data_v11(_DATA_START, _DATA_END).sort_index()

        df = df[df.index >= _SIM_START]
        n_hours = self.n_days * 24
        if len(df) > n_hours:
            df = df.iloc[:n_hours]
        elif len(df) < n_hours:
            print(f"  Warning: {len(df)} hours available, requested {n_hours}")

        cal = _mdn_calendar(df.index)
        cont = np.column_stack(
            [
                (df[col].values - self._norm[col]["mean"]) / self._norm[col]["std"]
                for col in self._feat_cols
            ]
        ).astype(np.float32)
        interaction = (cont[:, 0] * cont[:, 1]).reshape(-1, 1)
        X = np.concatenate([cal, cont, interaction], axis=1)

        X_t = torch.from_numpy(X).to(device)
        with torch.no_grad():
            pi, means, log_std = self._model(X_t)
            pi_np = pi.cpu().numpy()
            means_np = means.cpu().numpy()
            std_np = log_std.exp().cpu().numpy()

        K = self._K
        n = len(X)
        cumpi = np.cumsum(pi_np, axis=1)

        cols = {}
        for i in range(self.n_paths):
            rng = np.random.default_rng(self.seed + i)
            u = rng.uniform(size=(n, 1))
            k_choices = (u > cumpi).sum(axis=1).clip(0, K - 1)
            sel_means = means_np[np.arange(n), k_choices]
            sel_stds = std_np[np.arange(n), k_choices]
            v = rng.normal(sel_means, sel_stds).astype(np.float32)
            cols[f"sim_{i}"] = v
            print(f"  Path {i:2d}  mean={v.mean():.1f} EUR/MWh  "
                  f"std={v.std():.1f}  neg={100 * (v < 0).mean():.1f}%")

        self._result = pd.DataFrame(cols, index=df.index)
        return self._result


def main():
    print("=" * 60)
    print(f"simulate_price — MODEL_NAME={MODEL_NAME}")
    print(f"N_PATHS={N_PATHS}, N_DAYS={N_DAYS}, SEED={SEED}")
    print("=" * 60)

    simulator = MdnV11Simulator(FITTED_DIR, N_PATHS, N_DAYS, SEED)

    print(f"\n------ Simulating {N_PATHS} paths x {N_DAYS} days ------")
    simulator.simulate()

    print("\n------ Saving ------")
    simulator.save(OUTPUT_PATH)
    print(f"Saved {N_PATHS} paths x {N_DAYS} days to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
