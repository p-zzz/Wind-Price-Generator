"""Solar: DK1 fleet capacity factor (deterministic seasonal-mean shape).

``solar_generation_MW = simulate_solar_cf(...) x capacity_mw x solar.scale``. The
price model sees solar through ``solar_load_ratio`` (that scaled MW / load), so
``solar.scale`` moves price. The fitted model (``models/solar_cf.pkl``) is a plain
statsmodels OLS of capacity factor on hour x month plus the set of (month, hour)
pairs that count as daytime.
"""

import numpy as np
import pandas as pd


def _make_day_mask(index: pd.DatetimeIndex, day_pairs: set) -> np.ndarray:
    return np.array(
        [(m, h) in day_pairs for m, h in zip(index.month, index.hour)],
        dtype=bool,
    )


def simulate_solar_cf(df: pd.DataFrame, solar_pkl: dict) -> np.ndarray:
    """Solar capacity factor for every hour of the simulation calendar.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation calendar with ``hour`` and ``month`` columns
        (`generator.run.build_sim_index`).
    solar_pkl : dict
        Fitted solar model (`generator.io.load_fitted_objects`).

    Returns
    -------
    numpy.ndarray, shape (n_hours,)
        Capacity factor in [0, 1] (clipped at 0); exactly 0 at night.

    Warnings
    --------
    Deterministic: the same for every path and every seed, with no cloud-driven
    day-to-day variability. Solar output is the seasonal-mean shape only; see README
    "Known limitations".
    """
    mean_model = solar_pkl["mean_model"]
    day_pairs = solar_pkl["day_pairs"]
    n = len(df)
    day_mask = _make_day_mask(df.index, day_pairs)
    cf = np.zeros(n)
    if day_mask.any():
        cf[day_mask] = np.maximum(mean_model.predict(df[day_mask]).values, 0.0)
    return cf
