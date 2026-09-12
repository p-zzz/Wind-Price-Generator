"""
Generate synthetic wind speed paths for a specified wind model.
Write output to DATA/synthetic/{MODEL_NAME}/wind_paths.parquet.
No plotting. No validation.
"""
import pickle
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "wind"))

from wind_transformer import (
    BURN_IN, K_COMPONENTS, LOG_STD_MIN, LOG_STD_MAX,
    device,
    WindTransformerMDN, WindQuantileMapper,
    _apply_patsy_patch,
    _run_transformer_simulation,
)

import __main__
__main__.WindQuantileMapper = WindQuantileMapper


# ------ Config ------

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "transformer_r3"
N_PATHS    = 25
N_HOURS    = 43800
SEED       = 42

FITTED_DIR = Path("models/wind/fitted")
ERA5_PATH  = Path("DATA/processed/era5_sites_wind_hourly.parquet")
WIND_COL   = "horns_rev_wind_speed_10m_ms"

OUTPUT_DIR  = Path("DATA/synthetic") / MODEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "wind_paths.parquet"

_PKL_MAP = {
    "transformer_base": "wind_transformer",
    "transformer_r1":   "wind_transformer_tuned_r1",
    "transformer_r2":   "wind_transformer_tuned_r2",
    "transformer_r3":   "wind_transformer_tuned_r3",
    "transformer_r4":   "wind_transformer_tuned_r4",
}

# Standard-architecture Wind MDN family (hyperparameter tuning rounds r1-r6).
# All share the same MLP-backbone + Gaussian-mixture-head architecture; only
# k_lags / K / hidden_dims / activation / dropout differ per round.
_MDN_PKL_MAP = {
    "mdn_base":   "wind_mdn",
    "mdn_r1":     "wind_mdn_tuned_r1",
    "mdn_r3":     "wind_mdn_tuned_r3",
    "mdn_r3_ext": "wind_mdn_tuned_r3_extended",
    "mdn_r4a":    "wind_mdn_tuned_r4a",
    "mdn_r4b":    "wind_mdn_tuned_r4b",
    "mdn_r5":     "wind_mdn_tuned_r5",
    "mdn_r6":     "wind_mdn_tuned_r6",
}

_MDN_ACT_MAP     = {"relu": nn.ReLU, "gelu": nn.GELU}
_MDN_LOG_STD_MIN = -6.0
_MDN_LOG_STD_MAX = 2.0

_SIM_START = pd.Timestamp("2020-01-01", tz="Europe/Copenhagen")


# ------ Abstract base ------

class WindSimulator(ABC):
    def __init__(self, fitted_dir, wind_col, n_paths, n_hours, seed):
        self.fitted_dir = Path(fitted_dir)
        self.wind_col   = wind_col
        self.n_paths    = n_paths
        self.n_hours    = n_hours
        self.seed       = seed
        self._result    = None

    @abstractmethod
    def simulate(self) -> pd.DataFrame:
        ...

    def save(self, path: Path):
        if self._result is None:
            raise RuntimeError("Call simulate() before save()")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._result.to_parquet(path)


# ------ Transformer simulator ------

class TransformerSimulator(WindSimulator):
    def __init__(self, model_name, fitted_dir, wind_col, n_paths, n_hours, seed, era5_path):
        super().__init__(fitted_dir, wind_col, n_paths, n_hours, seed)
        self._era5_path = Path(era5_path)

        _apply_patsy_patch()

        pkl_stem = _PKL_MAP[model_name]
        pkl_path = self.fitted_dir / f"{pkl_stem}.pkl"
        with open(pkl_path, "rb") as f:
            fitted = pickle.load(f)

        hp           = fitted["hparams"]
        self._norm   = fitted["norm_stats"]
        self._mapper = fitted["quantile_mapper"]
        self._k_lags = hp["k_lags"]
        self._K      = hp["K"]

        self._model = WindTransformerMDN(**{k: hp[k] for k in [
            "k_lags", "d_model", "n_heads", "n_layers",
            "dim_feedforward", "dropout", "K", "log_std_min", "log_std_max",
        ]}).to(device)

        best_path = Path(fitted["best_model_path"])
        if not best_path.exists():
            best_path = self.fitted_dir / best_path.name
        self._model.load_state_dict(torch.load(best_path, map_location=device))
        self._model.eval()

    def simulate(self) -> pd.DataFrame:
        df_obs = pd.read_parquet(self._era5_path, columns=[self.wind_col])
        df_obs = df_obs.tz_convert("Europe/Copenhagen")
        pool   = (
            (df_obs[self.wind_col].values - self._norm["mean"]) / self._norm["std"]
        ).astype(np.float32)

        sim_index = pd.date_range(_SIM_START, periods=self.n_hours, freq="h")
        rng  = np.random.default_rng(self.seed)
        cols = {}

        for i in range(self.n_paths):
            max_start = max(len(pool) - self._k_lags - 1, 1)
            start_idx = int(rng.integers(0, max_start))
            buf       = list(pool[start_idx : start_idx + self._k_lags])

            sim  = _run_transformer_simulation(
                self._model, self._norm, self._k_lags, self._K,
                self._mapper, buf, self.n_hours, BURN_IN, 1.0, rng,
            )
            vals = sim[self.wind_col].values
            cols[f"sim_{i}"] = vals
            print(f"  Path {i:2d}  mean={vals.mean():.3f} m/s  std={vals.std():.3f} m/s")

        self._result = pd.DataFrame(cols, index=sim_index)
        return self._result


# ------ ARMA simulator (Horns Rev, production + SAR(24) ablation) ------

class ARMASimulator(WindSimulator):
    """
    Unlike TransformerSimulator/MDNSimulator, the ARMA process does not need
    a real historical lag buffer -- model.simulate() draws its own innovations
    from the fitted process. Simulation is over a synthetic calendar index
    (matching the burn-in/seasonal-mean procedure in
    models/wind/arma_horns_rev.py's plot_arma_acf_comparison, the source of
    the currently-documented ARMA validation numbers) so this reproduces them.
    """

    _PKL_STEM = {
        "arma":         "horns_rev_arma",
        "arma_no_sar":  "horns_rev_arma_no_sar",
    }
    _BURN_IN = 200

    def __init__(self, model_name, fitted_dir, wind_col, n_paths, n_hours, seed):
        super().__init__(fitted_dir, wind_col, n_paths, n_hours, seed)

        import patsy.eval as _pe
        _orig_capture = _pe.EvalEnvironment.capture.__func__

        @classmethod
        def _patched_capture(cls, eval_env=0, reference=0):
            try:
                return _orig_capture(cls, eval_env, reference)
            except Exception:
                return cls({}, {})

        _pe.EvalEnvironment.capture = _patched_capture

        pkl_path = self.fitted_dir / f"{self._PKL_STEM[model_name]}.pkl"
        with open(pkl_path, "rb") as f:
            fitted = pickle.load(f)
        self._mean_model  = fitted["mean_model"]
        self._monthly_std = fitted["monthly_std"]
        self._arma_model  = fitted["arma_model"]

    def _unstandardise(self, std_resid: pd.Series) -> pd.Series:
        multiplier = std_resid.index.month.map(self._monthly_std)
        return std_resid * multiplier.values

    def simulate(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        sim_index = pd.date_range(_SIM_START, periods=self.n_hours, freq="h")
        synth_df  = pd.DataFrame(
            {self.wind_col: 0.0, "month": sim_index.month, "hour": sim_index.hour},
            index=sim_index,
        )
        seasonal_mean = self._mean_model.predict(synth_df)
        cols = {}

        for i in range(self.n_paths):
            sim_resid = self._arma_model.simulate(
                nsimulations=self.n_hours + self._BURN_IN,
                random_state=int(rng.integers(int(1e9))),
            )[self._BURN_IN:]
            resid    = self._unstandardise(pd.Series(sim_resid, index=sim_index))
            wind_sim = np.abs(seasonal_mean.values + resid.values)
            cols[f"sim_{i}"] = wind_sim
            print(f"  Path {i:2d}  mean={wind_sim.mean():.3f} m/s  std={wind_sim.std():.3f} m/s")

        self._result = pd.DataFrame(cols, index=sim_index)
        return self._result


# ------ Wind MDN model (standard architecture family, rounds r1-r6) ------

class GaussianMDN(nn.Module):
    def __init__(self, input_dim, hidden_dims, K, activation="relu", dropout=0.0):
        super().__init__()
        act_cls = _MDN_ACT_MAP[activation]
        layers  = []
        in_dim  = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act_cls()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head     = nn.Linear(in_dim, K * 3)
        self.K = K

    def forward(self, x):
        h   = self.backbone(x)
        out = self.head(h)
        K       = self.K
        logits  = out[:, :K]
        means   = out[:, K : 2 * K].unsqueeze(-1)
        log_std = out[:, 2 * K :].unsqueeze(-1).clamp(_MDN_LOG_STD_MIN, _MDN_LOG_STD_MAX)
        return torch.softmax(logits, dim=-1), means, log_std


# ------ Wind MDN simulator ------

class MDNSimulator(WindSimulator):
    def __init__(self, model_name, fitted_dir, wind_col, n_paths, n_hours, seed, era5_path):
        super().__init__(fitted_dir, wind_col, n_paths, n_hours, seed)
        self._era5_path = Path(era5_path)

        pkl_stem = _MDN_PKL_MAP[model_name]
        pkl_path = self.fitted_dir / f"{pkl_stem}.pkl"
        with open(pkl_path, "rb") as f:
            fitted = pickle.load(f)

        hp         = fitted["hparams"]
        self._norm = fitted["norm_stats"]
        self._k_lags = hp["k_lags"]
        self._K      = hp["K"]

        self._model = GaussianMDN(
            input_dim=hp["input_dim"],
            hidden_dims=hp["hidden"],
            K=hp["K"],
            activation=hp.get("activation", "relu"),
            dropout=hp.get("dropout", 0.0),
        ).to(device)

        # best_model_path recorded in some pkls is stale (points at a different
        # round's checkpoint, e.g. r3_extended -> r3) — the f"{pkl_stem}_best.pt"
        # naming convention is followed by every round and takes precedence.
        best_path = self.fitted_dir / f"{pkl_stem}_best.pt"
        if not best_path.exists():
            best_path = Path(fitted["best_model_path"])
            if not best_path.exists():
                best_path = self.fitted_dir / best_path.name

        self._model.load_state_dict(torch.load(best_path, map_location=device))
        self._model.eval()

    def _run_mdn_path(self, buf, rng, norm_floor):
        results = []
        with torch.no_grad():
            for t in range(self.n_hours):
                ts  = _SIM_START + pd.Timedelta(hours=t)
                cal = np.array([
                    np.sin(2 * np.pi * ts.hour  / 24),
                    np.cos(2 * np.pi * ts.hour  / 24),
                    np.sin(2 * np.pi * ts.month / 12),
                    np.cos(2 * np.pi * ts.month / 12),
                ], dtype=np.float32)
                lags = np.array(list(reversed(buf)), dtype=np.float32)
                x_t  = torch.from_numpy(np.concatenate([cal, lags])).unsqueeze(0).to(device)

                pi, mu, log_std = self._model(x_t)
                pi_np  = pi.squeeze(0).cpu().numpy()
                mu_np  = mu.squeeze(0).cpu().numpy()
                std_np = log_std.squeeze(0).exp().cpu().numpy()

                k      = int(rng.choice(self._K, p=pi_np))
                sample = float(rng.normal(mu_np[k, 0], std_np[k, 0]))
                sample = max(sample, norm_floor)

                buf.append(sample)
                buf.pop(0)
                results.append(sample)

        return np.maximum(
            np.array(results, dtype=np.float32) * self._norm["std"] + self._norm["mean"], 0.0
        )

    def simulate(self) -> pd.DataFrame:
        df_obs = pd.read_parquet(self._era5_path, columns=[self.wind_col])
        df_obs = df_obs.tz_convert("Europe/Copenhagen")
        pool   = (
            (df_obs[self.wind_col].values - self._norm["mean"]) / self._norm["std"]
        ).astype(np.float32)

        sim_index  = pd.date_range(_SIM_START, periods=self.n_hours, freq="h")
        rng        = np.random.default_rng(self.seed)
        norm_floor = float(-self._norm["mean"] / self._norm["std"])
        cols       = {}

        for i in range(self.n_paths):
            max_start = max(len(pool) - self._k_lags - 1, 1)
            start_idx = int(rng.integers(0, max_start))
            buf       = list(pool[start_idx : start_idx + self._k_lags])

            vals = self._run_mdn_path(buf, rng, norm_floor)
            cols[f"sim_{i}"] = vals
            print(f"  Path {i:2d}  mean={vals.mean():.3f} m/s  std={vals.std():.3f} m/s")

        self._result = pd.DataFrame(cols, index=sim_index)
        return self._result


# ------ Main ------

def main():
    print("=" * 60)
    print(f"simulate_wind — MODEL_NAME={MODEL_NAME}")
    print(f"N_PATHS={N_PATHS}, N_HOURS={N_HOURS}, SEED={SEED}")
    print("=" * 60)

    if MODEL_NAME in _PKL_MAP:
        simulator = TransformerSimulator(
            model_name=MODEL_NAME,
            fitted_dir=FITTED_DIR,
            wind_col=WIND_COL,
            n_paths=N_PATHS,
            n_hours=N_HOURS,
            seed=SEED,
            era5_path=ERA5_PATH,
        )
    elif MODEL_NAME in _MDN_PKL_MAP:
        simulator = MDNSimulator(
            model_name=MODEL_NAME,
            fitted_dir=FITTED_DIR,
            wind_col=WIND_COL,
            n_paths=N_PATHS,
            n_hours=N_HOURS,
            seed=SEED,
            era5_path=ERA5_PATH,
        )
    elif MODEL_NAME in ARMASimulator._PKL_STEM:
        simulator = ARMASimulator(
            model_name=MODEL_NAME,
            fitted_dir=FITTED_DIR,
            wind_col=WIND_COL,
            n_paths=N_PATHS,
            n_hours=N_HOURS,
            seed=SEED,
        )
    else:
        raise ValueError(
            f"Unknown MODEL_NAME={MODEL_NAME!r}; expected one of "
            f"{list(_PKL_MAP) + list(_MDN_PKL_MAP) + list(ARMASimulator._PKL_STEM)}"
        )

    print(f"\n------ Simulating {N_PATHS} paths x {N_HOURS} h ------")
    simulator.simulate()

    print(f"\n------ Saving ------")
    simulator.save(OUTPUT_PATH)
    print(f"Saved {N_PATHS} paths x {N_HOURS}h to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
