"""
One-off extraction of the small historical data slices this repo's generator
actually needs, from the thesis repo's (Code/) full processed datasets. Run once,
from within this repo. Never modifies anything in Code/.

Two files come out of this:

- data/wind_speed_seed.parquet: last SEED_HOURS hours of Horns Rev 10m wind speed,
  used only to seed the wind Transformer's AR buffer before simulation starts.
  generator/wind.py only ever reads the last k_lags=72 hours of this regardless of
  how much is provided -- SEED_HOURS is padded well beyond that for margin.
- data/bootstrap_pool.parquet: actual_load_MW + net_position_MW for the
  BOOTSTRAP_START..BOOTSTRAP_END window, used by the paired block bootstrap that
  drives simulated load/net position. Matches joint_generator.py's window exactly.

Capacity baseline (onshore/offshore/solar installed MW) is not extracted as a data
file -- see print_capacity_baseline() below, which prints the three scalars to bake
into config/example.yaml directly (not worth a parquet for three numbers).

--source points at any directory with the thesis layout (DATA/processed/...,
DATA/raw/...), e.g. a workspace rebuilt via pipeline/build/; it defaults to the
thesis repo's Code/. The bootstrap pool must come from the same processed data the
shipped price model was trained on (see README "Known limitations").
"""

import argparse
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2] / "Code"  # default --source
THIS_DATA = Path(__file__).resolve().parents[1] / "data"
THIS_DATA.mkdir(parents=True, exist_ok=True)

SEED_HOURS = 500
BOOTSTRAP_START = "2023-01-01"
BOOTSTRAP_END = "2025-12-31"


def extract_wind_speed_seed() -> None:
    src = CODE_ROOT / "DATA/processed/era5_sites_wind_hourly.parquet"
    df = pd.read_parquet(src, columns=["horns_rev_wind_speed_10m_ms"])
    df = df.dropna().sort_index().tail(SEED_HOURS)
    dst = THIS_DATA / "wind_speed_seed.parquet"
    df.to_parquet(dst)
    print(f"wind_speed_seed.parquet: {len(df)} hours ({df.index[0]} .. {df.index[-1]})"
          f"  {dst.stat().st_size / 1e3:.1f} KB")


def extract_bootstrap_pool() -> None:
    src = CODE_ROOT / "DATA/processed/dk1_features_hourly.parquet"
    df = pd.read_parquet(src, columns=["actual_load_MW", "net_position_MW"])
    df = df.tz_convert("Europe/Copenhagen").sort_index()
    df = df.loc[BOOTSTRAP_START:BOOTSTRAP_END].dropna()
    dst = THIS_DATA / "bootstrap_pool.parquet"
    df.to_parquet(dst)
    print(f"bootstrap_pool.parquet: {len(df)} hours ({df.index[0].date()} .. "
          f"{df.index[-1].date()})  {dst.stat().st_size / 1e3:.1f} KB")


def print_capacity_baseline() -> None:
    src = CODE_ROOT / "DATA/raw/capacity_per_municipality/dk1_wind_capacity_monthly.parquet"
    cap = pd.read_parquet(src)
    latest = cap.iloc[-1]
    print(f"\nCapacity baseline as of {cap.index[-1].date()} "
          f"(source: Energinet CapacityPerMunicipality) -- bake into config/example.yaml:")
    print(f"  onshore_capacity_mw: {float(latest['OnshoreWindCapacity']):.1f}")
    print(f"  offshore_capacity_mw: {float(latest['OffshoreWindCapacity']):.1f}")
    print(f"  solar_capacity_mw: {float(latest['SolarPowerCapacity']):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=CODE_ROOT,
                        help="root with DATA/processed and DATA/raw (default: thesis Code/)")
    CODE_ROOT = parser.parse_args().source.resolve()
    print(f"Source: {CODE_ROOT}")
    extract_wind_speed_seed()
    extract_bootstrap_pool()
    print_capacity_baseline()
