# era5_sites_wind_download.py, optimised for speed
# Single bounding box covering all sites in config/sites.yaml (+0.25 deg margin).
# ThreadPoolExecutor submits all months concurrently. Cached months are reused only
# if they cover every site -- otherwise re-downloaded (a stale smaller box would make
# nearest-grid-point extraction silently snap to its edge).
# Per-site extraction is done by pipeline/build/era5_sites_builder.py afterwards.

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sys

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "build"))
import era5_sites_builder as builder  # noqa: E402  (shared site list + extraction)

# ------ CONFIG ------

OUT_RAW       = Path("DATA/raw/era5")
OUT_PROCESSED = Path("DATA/processed")
OUT_RAW.mkdir(parents=True, exist_ok=True)
OUT_PROCESSED.mkdir(parents=True, exist_ok=True)

SITES = builder.load_sites()   # {name: (lat, lon)} from config/sites.yaml

START_LOCAL = pd.Timestamp("1990-01-01", tz="Europe/Copenhagen")
END_LOCAL   = pd.Timestamp.now(tz="Europe/Copenhagen").normalize()

DATASET     = "reanalysis-era5-single-levels"
MAX_WORKERS = 8    # Keep as it is
RETRY_WAIT  = 30   # seconds to wait before retrying a failed request
MAX_RETRIES = 3


# ------ Bounding box covering ALL sites in one request ------

def dk1_area() -> list[float]:
    """Single N/W/S/E box that contains all sites in config/sites.yaml with a 0.25° margin."""
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
        with xr.open_dataset(target) as ds:
            uncovered = builder.file_covers(ds, SITES)
        if not uncovered:
            print(f"  skip {target.name} (already exists, covers all sites)")
            return target
        print(f"  re-download {target.name}: does not cover {uncovered}")

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


# ------ Main ------

def main() -> None:
    area = dk1_area()
    print(f"Bounding box (N/W/S/E): {area}")

    download_all(area)

    print("\nExtracting per-site wind speed series ...")
    df = builder.extract(OUT_RAW, SITES)

    print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Range:   {df.index.min()} → {df.index.max()}")
    print("\nMissing values:")
    print(df.isna().sum())
    print(df.head())
    print(df.tail())

    out_parquet = OUT_PROCESSED / "era5_sites_wind_hourly.parquet"
    df.to_parquet(out_parquet)
    print(f"\nSaved: {out_parquet}")


if __name__ == "__main__":
    main()