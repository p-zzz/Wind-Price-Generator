"""
Extract per-site ERA5 wind from the monthly NetCDF files into one hourly parquet.

Reads the sites from config/sites.yaml and the monthly files written by
pipeline/download/era5_sites_wind_download.py (DATA/raw/era5/era5_dk1_YYYYMM.nc),
takes the nearest ERA5 grid point to each site, and writes
DATA/processed/era5_sites_wind_hourly.parquet with, per site,
<site>_wind_speed_10m_ms, <site>_wind_speed_100m_ms and <site>_mslp_pa (UTC index).

Adding a site that lies inside the downloaded box needs no new download -- just
rerun this script. A site outside the box (or a month file that doesn't cover it)
is an error, not a silent snap to the box edge.

Run from the repo root (paths are relative to it):
    python pipeline/build/era5_sites_builder.py [--raw-dir DATA/raw/era5] [--out DATA/processed]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

REPO = Path(__file__).resolve().parents[2]
SITES_YAML = REPO / "config" / "sites.yaml"
MIN_FILE_SIZE = 10_000          # bytes; smaller files are broken downloads
MAX_SNAP_DEG = 0.18             # nearest grid point must be within ~half an ERA5 cell


def load_sites(path: Path = SITES_YAML) -> dict[str, tuple[float, float]]:
    """{name: (lat, lon)} from config/sites.yaml."""
    cfg = yaml.safe_load(path.read_text())
    return {name: (float(s["lat"]), float(s["lon"])) for name, s in cfg["sites"].items()}


def file_covers(ds: xr.Dataset, sites: dict[str, tuple[float, float]]) -> list[str]:
    """Names of the sites whose nearest grid point in ds is too far away (i.e. outside)."""
    lats, lons = ds["latitude"].values, ds["longitude"].values
    return [n for n, (lat, lon) in sites.items()
            if np.abs(lats - lat).min() > MAX_SNAP_DEG or np.abs(lons - lon).min() > MAX_SNAP_DEG]


def extract(raw_dir: Path, sites: dict[str, tuple[float, float]]) -> pd.DataFrame:
    files = sorted(p for p in raw_dir.glob("era5_dk1_*.nc") if p.stat().st_size > MIN_FILE_SIZE)
    if not files:
        raise FileNotFoundError(f"No ERA5 files (era5_dk1_*.nc) in {raw_dir}")

    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            if "valid_time" in ds.dims:
                ds = ds.rename({"valid_time": "time"})
            missing = file_covers(ds, sites)
            if missing:
                raise SystemExit(
                    f"{f.name} does not cover site(s) {missing} (box lat "
                    f"{float(ds.latitude.min())}-{float(ds.latitude.max())}, lon "
                    f"{float(ds.longitude.min())}-{float(ds.longitude.max())}). Run "
                    f"pipeline/download/era5_sites_wind_download.py to re-download with a box "
                    f"that includes them (needs a Copernicus CDS key).")
            cols = {}
            for name, (lat, lon) in sites.items():
                p = ds.sel(latitude=lat, longitude=lon, method="nearest")
                cols[f"{name}_wind_speed_10m_ms"] = np.hypot(p["u10"].values, p["v10"].values)
                cols[f"{name}_wind_speed_100m_ms"] = np.hypot(p["u100"].values, p["v100"].values)
                cols[f"{name}_mslp_pa"] = p["msl"].values
            idx = pd.DatetimeIndex(ds["time"].values).tz_localize("UTC")
            parts.append(pd.DataFrame(cols, index=idx))

    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "time_utc"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("DATA/raw/era5"))
    parser.add_argument("--out", type=Path, default=Path("DATA/processed"))
    args = parser.parse_args()

    sites = load_sites()
    print(f"Sites ({SITES_YAML.relative_to(REPO)}): " + ", ".join(f"{n} {v}" for n, v in sites.items()))
    df = extract(args.raw_dir, sites)

    with xr.open_dataset(sorted(args.raw_dir.glob("era5_dk1_*.nc"))[-1]) as ds:
        for name, (lat, lon) in sites.items():
            p = ds.sel(latitude=lat, longitude=lon, method="nearest")
            print(f"  {name:12s} -> grid point {float(p.latitude):.2f}N {float(p.longitude):.2f}E")
    print(f"{len(df):,} hours ({df.index.min()} .. {df.index.max()}), "
          f"{df.isna().any(axis=1).sum()} with missing values")

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "era5_sites_wind_hourly.parquet"
    df.to_parquet(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
