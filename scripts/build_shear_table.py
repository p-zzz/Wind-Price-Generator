"""
Build data/horns_rev_shear.csv: the wind-shear exponent alpha at Horns Rev, per
calendar month x 10 m wind-speed band, measured from ERA5's own 10 m / 100 m pair
(power law v(z) = v10 * (z / 10) ** alpha).

The generator simulates 10 m wind (every model is trained on it); this table lets it
also output hub-height wind (wind.hub_height_m) without retraining anything. Offshore
shear is low and varies mostly with season (warm-sea summers ~0.07, spring ~0.12) and
wind speed, barely with time of day. Per cell, alpha = log(mean v100 / mean v10) /
log(10), which is unbiased on mean 100 m speed (a median-of-hourly-alpha table is not).
Over 1990-2026: MAE on 100 m speed 0.62 m/s vs 0.72 for one constant alpha (0.096).

--source: root with the thesis layout (DATA/processed/era5_sites_wind_hourly.parquet);
defaults to the thesis repo's Code/, same as extract_data_slices.py.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2] / "Code"
OUT = Path(__file__).resolve().parents[1] / "data" / "horns_rev_shear.csv"
SPEED_EDGES = [0, 3, 5, 7, 9, 11, 14, np.inf]  # 10 m wind-speed bands, m/s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=CODE_ROOT,
                        help="root with DATA/processed (default: thesis Code/)")
    src = parser.parse_args().source.resolve()

    era5 = pd.read_parquet(src / "DATA/processed/era5_sites_wind_hourly.parquet")
    d = era5[["horns_rev_wind_speed_10m_ms", "horns_rev_wind_speed_100m_ms"]].dropna()
    d.columns = ["v10", "v100"]
    d = d.tz_convert("Europe/Copenhagen")          # months on the generator's calendar
    band = pd.cut(d["v10"], SPEED_EDGES, right=False)
    g = d.groupby([d.index.month, band], observed=True)
    alpha = np.log(g["v100"].mean() / g["v10"].mean()) / np.log(10)

    rows = [{"month": m, "v10_min_ms": b.left, "v10_max_ms": b.right, "alpha": round(a, 4),
             "n_hours": int(g.size().loc[(m, b)])}
            for (m, b), a in alpha.items()]
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False)
    print(f"{OUT.name}: {len(table)} cells from {len(d):,} hours "
          f"({d.index.min().date()} .. {d.index.max().date()}), alpha {table.alpha.min():.3f}-{table.alpha.max():.3f}")


if __name__ == "__main__":
    main()
