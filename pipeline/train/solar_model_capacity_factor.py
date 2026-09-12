"""
Solar generation model -- capacity factor, multi-year shape fit.

Motivation: solar_model_shape25_level26.py's raw-MW approach requires a
calib_ratio hack (shape fit on 2025, level calibrated to 2026) because raw
gen_solar_MW is not comparable across years of different installed capacity.
Feeding that raw MW into Price MDN also means any SOLAR_SCALE knob multiplies
the MW value directly, pushing the price model's z-scored solar input tens of
standard deviations beyond its 2015-2025 training range (see static sweep
wrong-signed solar-price finding, 2026-07-05).

Fix: normalise gen_solar_MW by DK1 installed solar capacity (Energinet
CapacityPerMunicipality, already downloaded -- see
DATA/raw/capacity_per_municipality/dk1_wind_capacity_monthly.parquet,
SolarPowerCapacity column, monthly since 2016-01) to get a bounded capacity
factor solar_cf in [0, ~0.86]. This is scale-invariant to future capacity
growth, so the shape can be fit on the full 2015-2025 history at once (no
single-year restriction, no calib_ratio) and it is safe to feed directly into
Price MDN regardless of any future SOLAR_SCALE value.

Fit quality (daylight hours, gap-dropped, not zero-filled; physical
solar-elevation-angle day mask -- see "Day/night mask" note below;
re-verified 2026-07-26, analysis/solar_cf_shape_window_comparison.py):
  2025 only:    n=1396  R2=0.6404
  2020-2025:    n=3866  R2=0.6570
  2015-2025:    n=6341  R2=0.6609
Multi-year fit matches single-year R2 with less than half the daylight
obs count of the earlier (now-superseded) empirical-threshold mask's
figures (n=2420/5919/9101 at that mask) -- shape is stable across years
once capacity-normalised, as expected physically (capacity factor depends
on irradiance/panel siting, not on how many MW are installed).

Pre-2016 capacity gap: capacity series starts 2016-01; earlier hours use the
first available capacity value held flat (negligible -- DK1 solar was tiny
pre-2016 and only ~624 raw obs exist for all of 2015).

Day/night mask (updated 2026-07-23): now a physical solar-elevation-angle
mask (astral, DK1 representative coordinate 56.0N/9.5E), replacing an
earlier empirical group-mean(MW) > 1.0 threshold. The empirical threshold
was found to misclassify hour=1-3am as "daylight" in 11 of 12 months,
traced to isolated stale/carried-forward nonzero readings in the raw
ENTSO-E solar series (e.g. 2024-05-03 02:00 local reports 39.04 MW -- not a
real reading). The physical mask is immune to this class of bug by
construction. See build_day_mask()'s docstring for detail.

Fitted objects:
  models/exog/fitted/solar_model_capacity_factor.pkl
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from astral import LocationInfo
from astral.sun import elevation as _solar_elevation

plt.style.use("seaborn-v0_8")

RAW_GEN_PATH  = Path("DATA/raw/generation/dk1_generation_by_type_wide.parquet")
CAPACITY_PATH = Path("DATA/raw/capacity_per_municipality/dk1_wind_capacity_monthly.parquet")
OUTDIR        = Path("analysis/outputs/DK1/exog_models/solar_capacity_factor")
FITTED_DIR    = Path("models/exog/fitted")
OUTDIR.mkdir(parents=True, exist_ok=True)
FITTED_DIR.mkdir(parents=True, exist_ok=True)

VARIABLE      = "gen_solar_MW"
TRAIN_START   = "2015-01-01"
TRAIN_END     = "2025-12-31"

# DK1 representative coordinate (central Jutland/Funen); DK1 spans ~54.9-57.3N
# in this project's ERA5 sites, but sunrise/sunset timing differs by only a
# few minutes across that range -- negligible at hourly resolution.
DK1_LATITUDE           = 56.0
DK1_LONGITUDE          = 9.5
ELEVATION_THRESHOLD_DEG = 0.0  # sun above horizon
_DK1_OBSERVER = LocationInfo(
    "DK1", "Denmark", "Europe/Copenhagen", DK1_LATITUDE, DK1_LONGITUDE
).observer


# ------ Data loading (raw, NaN-aware -- no fillna) ------

def load_hourly_solar() -> pd.Series:
    gen = pd.read_parquet(RAW_GEN_PATH)
    gen = gen.tz_convert("Europe/Copenhagen").sort_index()
    solar_h = gen[[VARIABLE]].resample("h").mean()
    full_idx = pd.date_range(solar_h.index.min(), solar_h.index.max(),
                              freq="h", tz="Europe/Copenhagen")
    return solar_h.reindex(full_idx)[VARIABLE]


def load_hourly_capacity(index: pd.DatetimeIndex) -> pd.Series:
    cap = pd.read_parquet(CAPACITY_PATH)["SolarPowerCapacity"]
    cap.index = cap.index.tz_localize("Europe/Copenhagen")
    union_idx = index.union(cap.index)
    cap_hourly = cap.reindex(union_idx).sort_index().interpolate(method="time")
    cap_hourly = cap_hourly.reindex(index).bfill()
    return cap_hourly


# ------ Day mask ------
#
# Physical daylight mask (solar elevation angle), replacing an earlier
# empirical group-mean(MW) > threshold approach. That approach was found to
# be contaminated by spurious nonzero nighttime readings in the raw ENTSO-E
# solar series -- e.g. 2024-05-03 02:00 local reports 39.04 MW, a stale/
# carried-forward meter value, not a real reading (confirmed against the raw
# long-form parquet directly: 01:00 and 02:00 both report the same 39.09/
# 39.04 MW). This single bad reading alone was enough to push several
# (month, hour) group means above the old 1.0 MW threshold, misclassifying
# hour=1-3am as "daylight" in 11 of 12 months. A physical mask is immune to
# this entire class of bug by construction, since it depends only on
# date/time/location, never on the (possibly corrupted) generation reading
# itself.

def _solar_elevation_deg(index: pd.DatetimeIndex) -> np.ndarray:
    return np.array([_solar_elevation(_DK1_OBSERVER, ts.to_pydatetime()) for ts in index])


def build_day_mask(observed: pd.Series) -> tuple[pd.Series, set]:
    elev = _solar_elevation_deg(observed.index)
    mask = pd.Series(elev > ELEVATION_THRESHOLD_DEG, index=observed.index)

    # day_pairs: (month, hour) combinations ever daylight, computed from solar
    # elevation on a representative mid-month date/time (2023-{month}-15,
    # {hour}:30 local) rather than from which hours happened to be reported
    # in the observed training window -- so it generalises correctly to any
    # future generator date, not just the training sample.
    day_pairs = set()
    for month in range(1, 13):
        for hour in range(24):
            ts = pd.Timestamp(2023, month, 15, hour, 30, tz="Europe/Copenhagen")
            if _solar_elevation(_DK1_OBSERVER, ts.to_pydatetime()) > ELEVATION_THRESHOLD_DEG:
                day_pairs.add((month, hour))

    print(f"Daylight (month, hour) pairs: {len(day_pairs)}")
    print(f"Daylight hours fraction:      {mask.mean():.3f}  ({mask.sum()} / {len(mask)} obs)")
    return mask, day_pairs


def make_day_mask_from_pairs(index: pd.DatetimeIndex, day_pairs: set) -> np.ndarray:
    return np.array(
        [(m, h) in day_pairs for m, h in zip(index.month, index.hour)],
        dtype=bool,
    )


# ------ Main ------

def main():
    print("=" * 60)
    print("Exogenous model: solar capacity factor (multi-year shape)")
    print("=" * 60)

    solar = load_hourly_solar()
    capacity_hourly = load_hourly_capacity(solar.index)
    solar_cf = (solar / capacity_hourly).rename("solar_cf")

    print(f"\nInstalled capacity series: {capacity_hourly.index.min()} .. "
          f"{capacity_hourly.index.max()}  "
          f"({capacity_hourly.iloc[0]:.1f} -> {capacity_hourly.iloc[-1]:.1f} MW)")

    day_mask_full, day_pairs = build_day_mask(solar.loc[TRAIN_START:TRAIN_END])

    train_cf = solar_cf.loc[TRAIN_START:TRAIN_END]
    train_mask = day_mask_full.reindex(train_cf.index)
    train_day = train_cf[train_mask].dropna()
    print(f"\nTraining window: {TRAIN_START} .. {TRAIN_END}  "
          f"({len(train_day)} valid daylight obs, gaps dropped not zero-filled)")

    valid_all = solar_cf.dropna()
    print(f"solar_cf range check (all obs): min={valid_all.min():.4f}  "
          f"max={valid_all.max():.4f}  frac>1.0={float((valid_all > 1.0).mean()):.4f}")

    df = train_day.to_frame("solar_cf")
    df["month"] = df.index.month
    df["hour"]  = df.index.hour

    print("\n-- Fitting capacity-factor shape (OLS: hour + month dummies), 2015-2025 --")
    mean_model = smf.ols("solar_cf ~ C(hour) + C(month)", data=df).fit()
    print(f"Shape R2 (daylight hours, 2015-2025): {mean_model.rsquared:.4f}")

    print("\n-- Calibrated full-year monthly means (predicted capacity factor) --")
    months = np.arange(1, 13)
    full_year_idx = pd.DataFrame({"month": np.repeat(months, 24), "hour": np.tile(np.arange(24), 12)})
    full_year_idx["is_day"] = [(m, h) in day_pairs for m, h in zip(full_year_idx["month"], full_year_idx["hour"])]
    day_rows = full_year_idx[full_year_idx["is_day"]].copy()
    day_rows["predicted_cf"] = mean_model.predict(day_rows[["month", "hour"]]).values

    predicted_monthly = day_rows.groupby("month")["predicted_cf"].mean().reindex(months)
    observed_monthly  = df.groupby("month")["solar_cf"].mean().reindex(months)

    print(f"{'Month':>6} {'Predicted CF':>14} {'Observed CF (2015-2025)':>24}")
    for m in months:
        obs_str = f"{observed_monthly[m]:24.4f}" if not np.isnan(observed_monthly[m]) else f"{'--':>24}"
        print(f"{m:6d} {predicted_monthly[m]:14.4f} {obs_str}")

    latest_capacity_MW = float(capacity_hourly.iloc[-1])
    latest_capacity_asof = capacity_hourly.index[-1]

    print("\n-- Validation plot: predicted CF vs observed monthly CF --")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(months, predicted_monthly.values, color="steelblue", lw=2, marker="o", label="Predicted (OLS shape, 2015-2025)")
    ax.plot(months, observed_monthly.values, color="black", lw=1.5, ls="--", marker="s", label="Observed (2015-2025, gaps dropped)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Capacity factor (daylight hours mean)")
    ax.set_title("Solar capacity factor: seasonal shape, 2015-2025")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUTDIR / "solar_cf_shape.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    fitted = {
        "mean_model":           mean_model,
        "day_pairs":            day_pairs,
        "variable":             "solar_cf",
        "training_period":      f"{TRAIN_START} to {TRAIN_END}",
        "latest_capacity_MW":   latest_capacity_MW,
        "latest_capacity_asof": str(latest_capacity_asof),
    }
    pkl_path = FITTED_DIR / "solar_model_capacity_factor.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(fitted, f)
    print(f"\nFitted objects saved to: {pkl_path}")
    print("Note: mean_model.predict(...) gives the capacity factor directly --")
    print("      scale-invariant, no calib_ratio or SOLAR_SCALE multiplication needed.")
    print(f"      latest_capacity_MW={latest_capacity_MW:.1f} (as of {latest_capacity_asof}) is")
    print("      the SOLAR_SCALE=1.0 anchor for converting predicted CF back to MW.")
    print("Done.")


if __name__ == "__main__":
    main()
