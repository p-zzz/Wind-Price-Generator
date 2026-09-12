# era5_sites_wind_download.py, optimised for speed
# Single bounding box covering all DK1 sites
# ThreadPoolExecutor submits all months concurrently
# Site wind speeds extracted by nearest-grid-point after download

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

# ------ CONFIG ------

OUT_RAW       = Path("DATA/raw/era5")
OUT_PROCESSED = Path("DATA/processed")
OUT_RAW.mkdir(parents=True, exist_ok=True)
OUT_PROCESSED.mkdir(parents=True, exist_ok=True)

SITES = {
    "horns_rev":   (55.55,   7.91  ),
    "esbjerg":     (55.465,  8.551 ),
    "aabenraa":    (54.9169, 9.3587),
    "bronderslev": (57.3169, 9.9593),
}

START_LOCAL = pd.Timestamp("1990-01-01", tz="Europe/Copenhagen")
END_LOCAL   = pd.Timestamp.now(tz="Europe/Copenhagen").normalize()

DATASET     = "reanalysis-era5-single-levels"
MAX_WORKERS = 8    # Keep as it is
RETRY_WAIT  = 30   # seconds to wait before retrying a failed request
MAX_RETRIES = 3


# ------ Bounding box covering ALL sites in one request ------

def dk1_area() -> list[float]:
    """Single N/W/S/E box that contains all four sites with a 0.25° margin."""
    lats = [lat for lat, _ in SITES.values()]
    lons = [lon for _, lon in SITES.values()]
    return [
        max(lats) + 0.25,   # N
        min(lons) - 0.25,   # W
        min(lats) - 0.25,   # S
        max(lons) + 0.25,   # E
    ]


# ------ Monthly request helpers ------

def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    s = start.tz_convert("UTC").to_period("M").to_timestamp().tz_localize("UTC")
    e = end.tz_convert("UTC").to_period("M").to_timestamp().tz_localize("UTC")
    return pd.date_range(s, e, freq="MS", tz="UTC")


def target_path(year: int, month: int) -> Path:
    return OUT_RAW / f"era5_dk1_{year:04d}{month:02d}.nc"


def retrieve_one_month(year: int, month: int, area: list[float]) -> Path:
    target = target_path(year, month)

    if target.exists() and target.stat().st_size > 10_000:
        print(f"  skip {target.name} (already exists)")
        return target

    req = {
        "product_type": "reanalysis",
        "format":       "netcdf",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
            "mean_sea_level_pressure",
        ],
        "year":         f"{year:04d}",
        "month":        f"{month:02d}",
        "day":          [f"{d:02d}" for d in range(1, 32)],
        "time":         [f"{h:02d}:00" for h in range(24)],
        "area":         area,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Each thread needs its own client instance
            c = cdsapi.Client(quiet=True)
            c.retrieve(DATASET, req, str(target))
            print(f"  done  {target.name}")
            return target
        except Exception as exc:
            print(f"  ERROR {target.name} attempt {attempt}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT * attempt)
            else:
                raise

    return target


# ------ Concurrent download ------

def download_all(area: list[float]) -> list[Path]:
    ms = month_starts(START_LOCAL, END_LOCAL)
    if len(ms) == 0:
        raise SystemExit("Empty date range — check START_LOCAL / END_LOCAL")

    jobs = [(int(dt.year), int(dt.month)) for dt in ms]
    print(f"Submitting {len(jobs)} monthly requests "
          f"({MAX_WORKERS} concurrent) ...")

    completed: list[Path] = []
    failed: list[tuple[int, int]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(retrieve_one_month, y, m, area): (y, m)
            for y, m in jobs
        }
        for fut in as_completed(futures):
            y, m = futures[fut]
            try:
                completed.append(fut.result())
            except Exception as exc:
                print(f"  FAILED {y:04d}-{m:02d} permanently: {exc}")
                failed.append((y, m))

    if failed:
        print(f"\nWARNING: {len(failed)} months failed and were skipped:")
        for y, m in sorted(failed):
            print(f"  {y:04d}-{m:02d}")

    return sorted(completed)


# ------ Extract per-site series from downloaded NetCDFs ------

def extract_site_series(nc_files: list[Path]) -> pd.DataFrame:
    """
    Open all monthly files, extract u10/v10 at the nearest grid point to each
    site, compute wind speed, return a UTC-indexed DataFrame.
    """
    valid = [p for p in nc_files if p.exists() and p.stat().st_size > 10_000]
    if not valid:
        raise FileNotFoundError("No valid NetCDF files to process.")

    print(f"Opening {len(valid)} NetCDF files ...")
    ds = xr.open_mfdataset([str(p) for p in valid], combine="by_coords")

    # Handle both old and new files transparently
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    print(f"  detected time dimension: '{time_dim}'")
    if time_dim == "valid_time":
        ds = ds.rename({"valid_time": "time"})

    start_utc = START_LOCAL.tz_convert("UTC").tz_localize(None)
    end_utc   = END_LOCAL.tz_convert("UTC").tz_localize(None)
    ds = ds.sel(time=slice(start_utc, end_utc))

    series = {}
    for site, (lat, lon) in SITES.items():
        point = ds.sel(latitude=lat, longitude=lon, method="nearest")
        
        u10  = point["u10"].values
        v10  = point["v10"].values
        u100 = point["u100"].values
        v100 = point["v100"].values
        msl  = point["msl"].values

        idx = pd.DatetimeIndex(point["time"].values).tz_localize("UTC")

        series[f"{site}_wind_speed_10m_ms"]  = pd.Series(np.sqrt(u10**2  + v10**2),  index=idx)
        series[f"{site}_wind_speed_100m_ms"] = pd.Series(np.sqrt(u100**2 + v100**2), index=idx)
        series[f"{site}_mslp_pa"]            = pd.Series(msl, index=idx)

    df = pd.DataFrame(series).sort_index()
    df.index.name = "time_utc"
    return df


# ------ Main ------

def main() -> None:
    area = dk1_area()
    print(f"Bounding box (N/W/S/E): {area}")

    nc_files = download_all(area)

    print("\nExtracting per-site wind speed series ...")
    df = extract_site_series(nc_files)

    print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Range:   {df.index.min()} → {df.index.max()}")
    print("\nMissing values:")
    print(df.isna().sum())
    print(df.head())
    print(df.tail())

    out_parquet = OUT_PROCESSED / "era5_sites_wind_hourly.parquet"
    out_csv     = OUT_PROCESSED / "era5_sites_wind_hourly.csv"
    df.to_parquet(out_parquet)
    df.to_csv(out_csv)
    print(f"\nSaved:\n  {out_parquet}\n  {out_csv}")


if __name__ == "__main__":
    main()