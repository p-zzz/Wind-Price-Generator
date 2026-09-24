"""Gas price schedule (TTF, EUR/MWh): a flat level or a multi-regime trajectory.

No fitted model: gas doesn't respond to anything in the simulation. See README
"Known limitations" for how trajectory noise weakens the wind-price correlation.
"""

import numpy as np

#: TTF gas price level per historical regime, EUR/MWh (regime means, rounded).
GAS_LEVELS = {
    "pre_crisis": 35.0,
    "crisis": 150.0,
    "post_crisis": 45.0,
}


def _walk_segments(n_hours: int, segments: list[dict]) -> tuple[list, int]:
    """(start, end, regime) per segment in order, clipped to n_hours, plus the hour
    the schedule itself ends (>= n_hours when the segments cover the horizon)."""
    walked = []
    hour = 0
    for seg in segments:
        seg_hours = int(seg["years"] * 8760)
        walked.append((hour, min(hour + seg_hours, n_hours), seg["regime"]))
        hour += seg_hours
        if hour >= n_hours:
            break
    return walked, hour


def build_gas_trajectory(
    n_hours: int,
    segments: list[dict],
    ramp_hours: int,
    noise_std: float,
    noise_ar: float,
    noise_floor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Multi-regime gas price path: regime levels, linear ramps, AR(1) noise.

    Parameters
    ----------
    n_hours : int
        Length of the path.
    segments : list of dict
        ``{"regime": str, "years": float}`` in order (1 year = 8760 h; regimes from
        `GAS_LEVELS`). Trimmed to ``n_hours``; a shorter schedule holds its last
        level. Must not be empty.
    ramp_hours : int
        Linear ramp from the previous level at the start of each later segment.
    noise_std : float
        Stationary standard deviation of the AR(1) noise, EUR/MWh (innovations are
        scaled by ``sqrt(1 - noise_ar**2)``). The noise starts at 0.
    noise_ar : float
        AR(1) coefficient.
    noise_floor : float
        Lower bound on the result, EUR/MWh.
    rng : numpy.random.Generator
        Draws the noise (``n_hours - 1`` normals).

    Returns
    -------
    numpy.ndarray of float32, shape (n_hours,)
        Gas price, EUR/MWh.
    """
    levels = np.zeros(n_hours)
    walked, hour = _walk_segments(n_hours, segments)
    regime_segments = [(start, end, GAS_LEVELS[regime]) for start, end, regime in walked]

    for i, (start, end, level) in enumerate(regime_segments):
        if i == 0:
            levels[start:end] = level
        else:
            prev_level = regime_segments[i - 1][2]
            ramp_end = min(start + ramp_hours, end)
            ramp = np.linspace(prev_level, level, ramp_end - start)
            levels[start:ramp_end] = ramp
            levels[ramp_end:end] = level

    if hour < n_hours:
        levels[hour:] = GAS_LEVELS[segments[-1]["regime"]]

    noise = np.zeros(n_hours)
    for t in range(1, n_hours):
        noise[t] = noise_ar * noise[t - 1] + rng.normal(
            0, noise_std * np.sqrt(1 - noise_ar**2)
        )

    trajectory = np.clip(levels + noise, noise_floor, None)
    return trajectory.astype(np.float32)


def gas_regime_labels(n_hours: int, segments: list[dict], ramp_hours: int) -> np.ndarray:
    """Regime name of every hour of a `build_gas_trajectory` schedule.

    Parameters
    ----------
    n_hours, segments, ramp_hours
        As in `build_gas_trajectory`.

    Returns
    -------
    numpy.ndarray of str, shape (n_hours,)
        Regime per hour; ramp hours between two *different* regimes are
        ``"transition"``, so they don't blur either regime's statistics.
    """
    walked, hour = _walk_segments(n_hours, segments)
    labels = np.empty(n_hours, dtype=object)
    for i, (start, end, regime) in enumerate(walked):
        labels[start:end] = regime
        if i > 0 and walked[i - 1][2] != regime:
            labels[start : min(start + ramp_hours, end)] = "transition"
    if hour < n_hours:
        labels[hour:] = walked[-1][2]
    return labels


def flat_gas_price(regime: str, custom_eur_mwh: float | None = None) -> float:
    """Constant gas price for ``gas.mode = "flat"``.

    Parameters
    ----------
    regime : {"pre_crisis", "crisis", "post_crisis", "custom"}
    custom_eur_mwh : float, optional
        Required when ``regime == "custom"``.

    Returns
    -------
    float
        Gas price, EUR/MWh.
    """
    if regime == "custom":
        if custom_eur_mwh is None:
            raise ValueError("gas.flat.custom_eur_mwh must be set when regime='custom'")
        return float(custom_eur_mwh)
    return GAS_LEVELS[regime]
