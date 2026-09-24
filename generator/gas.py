"""
Gas price: either a flat level (a real historical regime mean, or a custom value)
or a multi-regime trajectory (linear ramps + AR(1) noise) over the simulation
horizon. No fitted model -- this is a config-driven schedule, not a market model
of gas itself (see README "Known limitations").
"""

import numpy as np

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
    """segments: list of {"regime": str, "years": float}, walked in order and
    padded/trimmed to n_hours. Linear ramp of ramp_hours between each segment's
    level, AR(1) noise on top, clipped at noise_floor."""
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
    """Per-hour regime name for a trajectory, matching build_gas_trajectory's
    schedule. Ramp hours between two different regimes are labelled "transition"
    so they don't blur either regime's statistics."""
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
    if regime == "custom":
        if custom_eur_mwh is None:
            raise ValueError("gas.flat.custom_eur_mwh must be set when regime='custom'")
        return float(custom_eur_mwh)
    return GAS_LEVELS[regime]
