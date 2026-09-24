"""
Loading fitted objects and the small historical data slices. Deliberately does
NOT load load_model.pkl/net_position_model.pkl -- those exist in the thesis repo
but were dead code even there (superseded by the paired block bootstrap below,
which draws directly from data/bootstrap_pool.parquet).
"""

import inspect
import numbers
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import wind as wind_mod
from .price import MDN, PRICE_FEATURE_COLS

# ------ Patsy unpickling patch ------
#
# statsmodels OLS objects (used inside the solar/wind-CF pkls' mean_model) fail to
# unpickle with patsy >=1.0.2 due to a frame-capture regression in
# EvalEnvironment.__setstate__. Patch applied before any pickle.load below.

import patsy.eval as _pe

_ALL_FUTURE_FLAGS = _pe._ALL_FUTURE_FLAGS


@classmethod
def _safe_capture(cls, eval_env=0, reference=0):
    if isinstance(eval_env, cls):
        return eval_env
    if isinstance(eval_env, numbers.Integral):
        depth = eval_env + reference
    else:
        raise TypeError(
            "Parameter 'eval_env' must be either an integer "
            "or an instance of patsy.EvalEnvironment."
        )
    frame = inspect.currentframe()
    try:
        for _ in range(depth + 1):
            if frame is None:
                return cls([{}], 0)
            frame = frame.f_back
        if frame is None:
            return cls([{}], 0)
        return cls(
            [frame.f_locals, frame.f_globals],
            frame.f_code.co_flags & _ALL_FUTURE_FLAGS,
        )
    finally:
        del frame


_pe.EvalEnvironment.capture = _safe_capture


def _register_unpickle_classes() -> None:
    """The thesis repo's fitted pkls reference their quantile mapper classes by
    the original module they were fitted in -- WindQuantileMapper as
    __module__ == "wind_transformer", WindCFQuantileMapper as __module__ ==
    "wind_capacity_factor_arma" (checked empirically against the actual pickle
    bytes, not assumed). Register both module names as aliases for
    generator.wind, which holds identical copies of both classes, so unpickling
    resolves without needing the original thesis-repo scripts present."""
    sys.modules["wind_transformer"] = wind_mod
    sys.modules["wind_capacity_factor_arma"] = wind_mod


def load_fitted_objects(models_dir: Path, device: torch.device) -> dict:
    _register_unpickle_classes()

    with open(models_dir / "wind_transformer_r3.pkl", "rb") as f:
        wind_pkl = pickle.load(f)
    with open(models_dir / "wind_cf_onshore_arma.pkl", "rb") as f:
        wind_cf_onshore_pkl = pickle.load(f)
    with open(models_dir / "wind_cf_offshore_arma.pkl", "rb") as f:
        wind_cf_offshore_pkl = pickle.load(f)
    with open(models_dir / "solar_cf.pkl", "rb") as f:
        solar_pkl = pickle.load(f)
    with open(models_dir / "price_mdn_v11.pkl", "rb") as f:
        price_pkl = pickle.load(f)

    if list(price_pkl["feature_columns"]) != PRICE_FEATURE_COLS:
        raise ValueError(
            f"price_mdn_v11.pkl feature_columns {price_pkl['feature_columns']} do not "
            f"match the order simulate_price_mdn() feeds: {PRICE_FEATURE_COLS}"
        )

    hp = price_pkl["hparams"]
    mdn = MDN(hp["input_dim"], hp["hidden"], hp["K"]).to(device)
    stored = Path(price_pkl["best_model_path"])
    pt_path = stored if stored.exists() else models_dir / stored.name
    mdn.load_state_dict(torch.load(pt_path, map_location=device))
    mdn.eval()

    wind_cfg = wind_pkl["hparams"]
    transformer = wind_mod.WindTransformerMDN(
        k_lags=wind_cfg["k_lags"],
        d_model=wind_cfg["d_model"],
        n_heads=wind_cfg["n_heads"],
        n_layers=wind_cfg["n_layers"],
        dim_feedforward=wind_cfg.get("dim_feedforward", 256),
        dropout=wind_cfg["dropout"],
        K=wind_cfg["K"],
        log_std_min=wind_cfg.get("log_std_min", -4.0),
        log_std_max=wind_cfg.get("log_std_max", 6.0),
    ).to(device)
    wind_stored = Path(wind_pkl["best_model_path"])
    wind_pt_path = wind_stored if wind_stored.exists() else models_dir / wind_stored.name
    transformer.load_state_dict(torch.load(wind_pt_path, map_location=device))
    transformer.eval()

    return {
        "transformer": transformer,
        "mapper": wind_pkl["quantile_mapper"],
        "wind_pkl": wind_pkl,
        "wind_cf_onshore": wind_cf_onshore_pkl,
        "wind_cf_offshore": wind_cf_offshore_pkl,
        "solar": solar_pkl,
        "price": price_pkl,
        "price_feature_cols": price_pkl["feature_columns"],
        "mdn": mdn,
    }


def load_wind_speed_seed(data_dir: Path, k_lags: int, norm_mean: float, norm_std: float) -> list:
    seed = pd.read_parquet(data_dir / "wind_speed_seed.parquet")
    obs_pool = seed[wind_mod.WIND_COL].dropna().values
    return list((obs_pool[-k_lags:] - norm_mean) / norm_std)


def load_bootstrap_source(data_dir: Path) -> pd.DataFrame:
    """Reindexed onto a full hourly index so positional blocks are calendar-true;
    the pool's missing hours become NaN rows, which paired_block_bootstrap never
    draws across."""
    df = pd.read_parquet(data_dir / "bootstrap_pool.parquet").sort_index()
    full_idx = pd.date_range(df.index[0], df.index[-1], freq="h")
    return df[["actual_load_MW", "net_position_MW"]].reindex(full_idx)
