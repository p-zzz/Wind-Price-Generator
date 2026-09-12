# ERA5 multi-site dataset builder
# Reads monthly NetCDF files produced by era5_sites_wind_download.py
# and assembles a single parquet with one column per site.
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
 
# ------ CONFIG ------
 
RAW_DIR = Path("DATA/raw/era5")
OUT_DIR = Path("DATA/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
# Must match SITES keys in the download script exactly
SITES = [
    "horns_rev",
    "esbjerg",
    "aabenraa",
    "bronderslev",
]
 
START_LOCAL = pd.Timestamp("2014-12-12", tz="Europe/Copenhagen")
END_LOCAL   = pd.Timestamp("2026-02-27", tz="Europe/Copenhagen")
 
MIN_FILE_SIZE_KB = 10  # files smaller than this are treated as broken downloads
 
 
# ------ Per-site builder ------
 
def build_site_series(site: str, nc_files: list[Path]) -> pd.Series:
    """
    Open all monthly NetCDF files for one site, compute wind speed,
    and return a UTC-indexed Series trimmed to [START_LOCAL, END_LOCAL).
    """
    if not nc_files:
        raise FileNotFoundError(f"No valid NetCDF files found for site '{site}'")
 
    ds = xr.open_mfdataset([str(p) for p in nc_files], combine="by_coords")
 
    u10 = ds["u10"].mean(dim=("latitude", "longitude"))
    v10 = ds["v10"].mean(dim=("latitude", "longitude"))
    wspd = np.sqrt(u10 ** 2 + v10 ** 2).to_pandas()
 
    wspd.index = pd.DatetimeIndex(wspd.index, tz="UTC", name="time_utc")
    wspd = wspd.sort_index()
 
    start_utc = START_LOCAL.tz_convert("UTC")
    end_utc   = END_LOCAL.tz_convert("UTC")
    wspd = wspd.loc[(wspd.index >= start_utc) & (wspd.index < end_utc)]
 
    wspd.name = f"{site}_wind_speed_10m_ms"
    return wspd
 
 
# ------ Main ------
 
def main() -> None:
    series = {}
 
    for site in SITES:
        # Match files produced by the download script: era5_{site}_{YYYYMM}.nc
        all_files = sorted(RAW_DIR.glob(f"era5_{site}_*.nc"))
        valid_files = [p for p in all_files if p.stat().st_size > MIN_FILE_SIZE_KB * 1024]
        skipped = len(all_files) - len(valid_files)
 
        print(f"{site}: {len(valid_files)} valid files"
              + (f" ({skipped} skipped — too small)" if skipped else ""))
 
        if not valid_files:
            print(f"  WARNING: no files found for {site}, column will be NaN")
            series[site] = pd.Series(name=f"{site}_wind_speed_10m_ms", dtype=float)
            continue
 
        series[site] = build_site_series(site, valid_files)
 
    # Outer join so a missing site shows as NaN rather than silently dropping rows
    df = pd.concat(series.values(), axis=1, join="outer").sort_index()
 
    # Checks
    print(f"\nAssembled dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Time range: {df.index.min()} → {df.index.max()}")
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nHead:")
    print(df.head())
 
    out_parquet = OUT_DIR / "era5_sites_wind_hourly.parquet"
    out_csv     = OUT_DIR / "era5_sites_wind_hourly.csv"
    df.to_parquet(out_parquet)
    df.to_csv(out_csv)
 
    print(f"\nSaved:\n  {out_parquet}\n  {out_csv}")
 
 
if __name__ == "__main__":
    main()