"""
Solar: capacity-factor shape model (deterministic seasonal mean, no stochastic
residual -- a documented limitation, see README "Known limitations"). Extracted
from joint_generator.py; the fitted pkl (models/solar_cf.pkl) needs no custom
classes to unpickle (mean_model is a plain statsmodels OLS, day_pairs a plain set).

solar_cf feeds the price model directly and unscaled -- SOLAR_SCALE never touches
it, for the same reason WIND_SCALE never touches wind_speed_ms (see generator/wind.py).
solar_MW = solar_cf x target installed solar capacity x SOLAR_SCALE is the separate,
independently-scaled stream.
"""

import numpy as np
import pandas as pd


def _make_day_mask(index: pd.DatetimeIndex, day_pairs: set) -> np.ndarray:
    return np.array(
        [(m, h) in day_pairs for m, h in zip(index.month, index.hour)],
        dtype=bool,
    )


def simulate_solar_cf(df: pd.DataFrame, solar_pkl: dict) -> np.ndarray:
    mean_model = solar_pkl["mean_model"]
    day_pairs = solar_pkl["day_pairs"]
    n = len(df)
    day_mask = _make_day_mask(df.index, day_pairs)
    cf = np.zeros(n)
    if day_mask.any():
        cf[day_mask] = np.maximum(mean_model.predict(df[day_mask]).values, 0.0)
    return cf
