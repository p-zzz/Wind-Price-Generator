import pandas as pd
from pathlib import Path

RAW = Path("DATA/raw")
OUT = Path("DATA/processed")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading raw datasets...")

prices = pd.read_parquet(RAW/"prices/dk1_day_ahead_prices_raw.parquet")
load   = pd.read_parquet(RAW/"load/dk1_actual_load_raw.parquet")
gen    = pd.read_parquet(RAW/"generation/dk1_generation_by_type_wide.parquet")
netpos = pd.read_parquet(RAW/"net_position/dk1_net_position_raw.parquet")


# Convert everything to Cph time
prices = prices.tz_convert("Europe/Copenhagen")
load   = load.tz_convert("Europe/Copenhagen")
gen    = gen.tz_convert("Europe/Copenhagen")
netpos = netpos.tz_convert("Europe/Copenhagen")

# Resample everything to hourly
print("Resampling to hourly...")

prices_h = prices[["day_ahead_price"]].resample("h").mean()
load_h   = load[["actual_load_MW"]].resample("h").mean()
gen_h    = gen.resample("h").mean()
netpos_h = netpos[["net_position_MW"]].resample("h").mean()

# fill structural zeros
gen_h = gen_h.fillna(0)

# Create thermal aggregate
thermal_cols = [
    "gen_fossil_gas_MW",
    "gen_fossil_hard_coal_MW",
    "gen_fossil_oil_MW",
    "gen_fossil_brown_coal_lignite_MW",
]

gen_h["gen_thermal_MW"] = gen_h[thermal_cols].sum(axis=1, min_count=1)

# Merge everything
print("Merging datasets...")

df = (
    prices_h
    .join(load_h)
    .join(gen_h)
    .join(netpos_h)
)

# Sort index
df = df.sort_index()

print("\nMissing values per column:")
print(df.isna().sum())

print("\nMissing values (%):")
print((df.isna().mean()*100).round(3))

print("\nTotal rows:", len(df))

print("\nTime step distribution:")
print(df.index.to_series().diff().value_counts().head())

print("\nFirst rows with NaNs:")
print(df[df.isna().any(axis=1)].head(20))

######## LINUX SHIT #######
# If using windows safely remove
import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

df["day_ahead_price"].isna().astype(int).plot()
plt.title("NaN locations in price")
plt.show()

# Interpolation and drop nans
df = df.interpolate(limit=2)
df = df.dropna()

# Convert back to UTC for storage
df = df.tz_convert("UTC")

# Save
df.to_parquet(OUT/"dk1_base_hourly.parquet")
df.to_csv(OUT/"dk1_base_hourly.csv")

print("Done.")
print(df.shape)
print(df.head())
print(df.tail())