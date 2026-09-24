"""Load the fitted models (``models/``) and the historical data slices (``data/``).

Also patches two unpickling issues the shipped pickles need: a patsy >= 1.0.2
frame-capture regression, and quantile-mapper classes pickled under their thesis-repo
module names. Deliberately does not load the thesis repo's load/net-position models,
which were superseded by the paired block bootstrap.
"""

import inspect
import json
import numbers
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import wind as wind_mod
from .price import MDN, MDNEnsemble, PRICE_FEATURE_COLS

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
    """Load every fitted model the generator needs.

    Parameters
    ----------
    models_dir : Path
        Directory with the wind Transformer (+ quantile mapper), onshore/offshore
        capacity-factor ARMA, solar CF and price model files (the repo's ``models/``).
    device : torch.device
        Device for the two neural networks (see `generator.run.get_device`).

    Returns
    -------
    dict
        Keys ``transformer``, ``mapper``, ``wind_pkl``, ``wind_cf_onshore``,
        ``wind_cf_offshore``, ``solar``, ``price``, ``price_feature_cols`` and
        ``mdn`` -- the price model, an `generator.price.MDN` or, for a seed
        ensemble (``price_mdn_v11.pkl`` with a ``members`` list), an
        `generator.price.MDNEnsemble`. Pass it unchanged to `generator.run.generate`.

    Raises
    ------
    ValueError
        If the price model's ``feature_columns`` differ from
        `generator.price.PRICE_FEATURE_COLS` (inputs would otherwise be fed in the
        wrong order without any error).
    """
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

    # Either one model ("best_model_path") or a seed ensemble ("members", written by
    # pipeline/train/assemble_price_ensemble.py) sharing norm_stats/hparams.
    hp = price_pkl["hparams"]
    members = price_pkl.get("members") or [{"best_model_path": price_pkl["best_model_path"]}]
    nets = []
    for member in members:
        net = MDN(hp["input_dim"], hp["hidden"], hp["K"]).to(device)
        stored = Path(member["best_model_path"])
        pt_path = stored if stored.exists() else models_dir / stored.name
        net.load_state_dict(torch.load(pt_path, map_location=device))
        nets.append(net)
    mdn = nets[0] if len(nets) == 1 else MDNEnsemble(nets).to(device)
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
    """Load the wind Transformer's initial autoregressive buffer.

    Parameters
    ----------
    data_dir : Path
        Directory with ``wind_speed_seed.parquet`` (recent Horns Rev 10 m wind).
    k_lags : int
        Number of lags the Transformer uses (buffer length).
    norm_mean, norm_std : float
        The Transformer's wind-speed normalisation, m/s.

    Returns
    -------
    list of float
        The last ``k_lags`` observed hours, normalised, oldest first. Every path
        starts from this same buffer; paths diverge through the random draws.
    """
    seed = pd.read_parquet(data_dir / "wind_speed_seed.parquet")
    obs_pool = seed[wind_mod.WIND_COL].dropna().values
    return list((obs_pool[-k_lags:] - norm_mean) / norm_std)


def load_shear_table(data_dir: Path) -> pd.DataFrame:
    """Wind-shear exponents for the hub-height wind output.

    Parameters
    ----------
    data_dir : Path
        Directory with ``horns_rev_shear.csv`` (built by ``scripts/build_shear_table.py``).

    Returns
    -------
    pandas.DataFrame
        One row per calendar month x 10 m wind-speed band: ``month``,
        ``v10_min_ms``, ``v10_max_ms``, ``alpha``, ``n_hours``. See
        `generator.wind.hub_height_wind_speed`.
    """
    return pd.read_csv(data_dir / "horns_rev_shear.csv")


def load_site_model(data_dir: Path, name: str) -> dict:
    """Load a fitted farm-site wind model.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``sites/<name>.json`` (the repo's ``data/``).
    name : str
        Site name as in ``config/sites.yaml``.

    Returns
    -------
    dict
        The model `generator.wind.site_wind_speed` takes (written by
        ``scripts/build_site_model.py``).

    Raises
    ------
    FileNotFoundError
        If the site has no fitted model yet.
    """
    path = data_dir / "sites" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No site model {path}. Add '{name}' to config/sites.yaml, then run "
            f"pipeline/build/era5_sites_builder.py and scripts/build_site_model.py --site {name}.")
    return json.loads(path.read_text())


def load_bootstrap_source(data_dir: Path) -> pd.DataFrame:
    """Historical load and net position the bootstrap resamples from.

    Parameters
    ----------
    data_dir : Path
        Directory with ``bootstrap_pool.parquet`` (DK1, 2023-2025, hourly).

    Returns
    -------
    pandas.DataFrame
        Columns ``actual_load_MW`` and ``net_position_MW`` (MW; net position is
        export-positive, import-negative) on a gap-free hourly Europe/Copenhagen
        index. Hours missing from the source are NaN rows, which
        `generator.bootstrap.paired_block_bootstrap` never draws across.
    """
    df = pd.read_parquet(data_dir / "bootstrap_pool.parquet").sort_index()
    full_idx = pd.date_range(df.index[0], df.index[-1], freq="h")
    return df[["actual_load_MW", "net_position_MW"]].reindex(full_idx)
