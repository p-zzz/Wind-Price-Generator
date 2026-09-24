"""
Paired block bootstrap for load and net position -- drawn directly from real
historical data (data/bootstrap_pool.parquet), not a fitted parametric model.
"Paired" means both series are resampled from the same randomly chosen blocks, so
their real historical co-movement (e.g. cold snaps driving both load and imports up)
is preserved.

Calendar-conditioned: each block is drawn from pool start hours with the same local
hour-of-day and a day-of-year within +/- window_days of the simulated timestamp it
fills, so load seasonality and the daily cycle stay aligned with the simulation
calendar (and with solar/price, which condition on the same calendar).

DST: a block never spans a UTC-offset change, in the pool or in the simulation. A
block that crossed one on only one side would shift every later hour in it by one
relative to the local clock, so blocks are cut at simulated DST transitions and only
drawn from pool stretches with a constant offset.
"""

import numpy as np
import pandas as pd


def _seasonal_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Day-of-year on a fixed 365-day basis: in leap years, days after Feb 28 are
    shifted back by one so the same calendar date maps to the same value in every
    year (Feb 29 shares its value with Mar 1)."""
    doy = index.dayofyear.to_numpy()
    return doy - (index.is_leap_year & (index.month > 2)).astype(int)


def _utc_offset_hours(index: pd.DatetimeIndex) -> np.ndarray:
    """Local wall-clock minus UTC, in hours (1 in CET, 2 in CEST)."""
    local = index.tz_localize(None)
    utc = index.tz_convert("UTC").tz_localize(None)
    return ((local - utc) / pd.Timedelta(hours=1)).to_numpy()


def paired_block_bootstrap(
    s1: np.ndarray,
    s2: np.ndarray,
    pool_index: pd.DatetimeIndex,
    sim_index: pd.DatetimeIndex,
    block_size: int,
    window_days: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """s1/s2 are aligned to pool_index, a gap-free hourly index (missing hours as
    NaN, see load_bootstrap_source). Blocks that would cross a NaN are never drawn,
    so no gap-filling is needed."""
    n_pool = len(pool_index)
    n_hours = len(sim_index)
    if block_size > n_pool:
        raise ValueError(f"block_size={block_size} exceeds bootstrap pool length {n_pool}")

    # Prefix counts, so "does [s, s + take) contain a NaN / an offset change" is an
    # O(1) difference for every candidate start at once.
    missing = np.isnan(s1) | np.isnan(s2)
    n_missing = np.concatenate([[0], np.cumsum(missing)])
    pool_offset = _utc_offset_hours(pool_index)
    n_pool_shift = np.concatenate([[0], np.cumsum(pool_offset[1:] != pool_offset[:-1])])

    pool_day = _seasonal_day(pool_index)
    pool_hour = pool_index.hour.to_numpy()
    sim_day = _seasonal_day(sim_index)
    sim_hour = sim_index.hour.to_numpy()
    sim_offset = _utc_offset_hours(sim_index)
    # Positions where the simulated UTC offset changes (DST transitions).
    sim_shift_at = np.flatnonzero(sim_offset[1:] != sim_offset[:-1]) + 1

    r1, r2 = np.empty(n_hours), np.empty(n_hours)
    filled = 0
    while filled < n_hours:
        next_shift = sim_shift_at[np.searchsorted(sim_shift_at, filled, side="right"):]
        take = min(block_size, n_hours - filled)
        if next_shift.size:
            take = min(take, int(next_shift[0]) - filled)

        starts = np.arange(n_pool - take + 1)
        clean = (n_missing[starts + take] - n_missing[starts]) == 0
        same_offset = (n_pool_shift[starts + take - 1] - n_pool_shift[starts]) == 0
        day_dist = np.abs(pool_day[: starts.size] - sim_day[filled])
        day_dist = np.minimum(day_dist, 365 - day_dist)  # Dec 31 and Jan 1 are neighbours
        candidates = np.flatnonzero(
            clean
            & same_offset
            & (pool_hour[: starts.size] == sim_hour[filled])
            & (day_dist <= window_days)
        )
        if candidates.size == 0:
            raise ValueError(
                f"No gap-free bootstrap block starts within +/-{window_days} days of "
                f"{sim_index[filled]} at hour {sim_hour[filled]} -- widen "
                f"bootstrap_window_days or shrink block_size."
            )
        start = int(candidates[rng.integers(candidates.size)])
        r1[filled : filled + take] = s1[start : start + take]
        r2[filled : filled + take] = s2[start : start + take]
        filled += take
    return r1, r2
