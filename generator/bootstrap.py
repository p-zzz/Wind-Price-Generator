"""
Paired block bootstrap for load and net position -- drawn directly from real
historical data (data/bootstrap_pool.parquet), not a fitted parametric model.
"Paired" means both series are resampled from the same randomly chosen blocks, so
their real historical co-movement (e.g. cold snaps driving both load and imports up)
is preserved.
"""

import numpy as np


def paired_block_bootstrap(
    s1: np.ndarray,
    s2: np.ndarray,
    n_hours: int,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_source = len(s1)
    r1, r2 = np.empty(n_hours), np.empty(n_hours)
    filled = 0
    while filled < n_hours:
        start = int(rng.integers(0, n_source - block_size + 1))
        take = min(block_size, n_hours - filled)
        r1[filled : filled + take] = s1[start : start + take]
        r2[filled : filled + take] = s2[start : start + take]
        filled += take
    return r1, r2
