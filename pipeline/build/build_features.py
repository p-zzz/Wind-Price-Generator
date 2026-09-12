import pandas as pd

df = pd.read_parquet("DATA/processed/dk1_base_hourly.parquet")

# generation aggregates
df["wind_total_MW"] = df["gen_wind_onshore_MW"] + df["gen_wind_offshore_MW"]

df["renewable_total_MW"] = (
    df["gen_wind_onshore_MW"]
    + df["gen_wind_offshore_MW"]
    + df["gen_solar_MW"]
    + df["gen_hydro_run_of_river_MW"]
)

df["thermal_share"] = df["gen_thermal_MW"] / df["actual_load_MW"]

df["renewable_share"] = df["renewable_total_MW"] / df["actual_load_MW"]

# time features
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month

df.to_parquet("DATA/processed/dk1_features_hourly.parquet")