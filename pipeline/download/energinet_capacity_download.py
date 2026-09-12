"""
Download DK1 installed wind capacity (onshore/offshore, MW) from Energinet's
free public Energi Data Service API (no token, no auth).

Source: https://api.energidataservice.dk/dataset/CapacityPerMunicipality
Monthly granularity per municipality since 2016-01. DK1/DK2 split is derived
from Danish administrative region (static, standard mapping):
  DK1 = Region Nordjylland + Region Midtjylland + Region Syddanmark
  DK2 = Region Hovedstaden + Region Sjaelland

Saves:
  DATA/raw/capacity_per_municipality/capacity_per_municipality_raw.parquet  (all municipalities, all months)
  DATA/raw/capacity_per_municipality/municipality_region_map.parquet       (MunicipalityCode -> RegionName)
  DATA/raw/capacity_per_municipality/dk1_wind_capacity_monthly.parquet     (DK1-aggregated onshore/offshore MW)
"""

import requests
import pandas as pd
from pathlib import Path

OUT_DIR = Path("DATA/raw/capacity_per_municipality")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DK1_REGIONS = {"Region Nordjylland", "Region Midtjylland", "Region Syddanmark"}
DK2_REGIONS = {"Region Hovedstaden", "Region Sjælland"}


# ------ Fetch municipality -> region mapping ------

def fetch_municipality_region_map() -> pd.DataFrame:
    url = "https://api.energidataservice.dk/dataset/PrivateConsumptionHeatingMonth"
    params = {"limit": 5000, "columns": "MunicipalityCode,Municipality,RegionName"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    records = r.json()["records"]
    df = pd.DataFrame(records).drop_duplicates(subset="MunicipalityCode")
    df = df.sort_values("MunicipalityCode").reset_index(drop=True)
    return df


# ------ Fetch full CapacityPerMunicipality history ------

def fetch_capacity_per_municipality() -> pd.DataFrame:
    url = "https://api.energidataservice.dk/dataset/CapacityPerMunicipality"
    params = {"limit": 20000}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()
    print(f"CapacityPerMunicipality: total={payload['total']}  fetched={len(payload['records'])}")
    df = pd.DataFrame(payload["records"])
    df["Month"] = pd.to_datetime(df["Month"])
    df["MunicipalityNo"] = df["MunicipalityNo"].astype(int)
    return df


# ------ Main ------

def main():
    print("------ Fetching municipality -> region map ------")
    muni_map = fetch_municipality_region_map()
    print(f"{len(muni_map)} municipalities mapped")
    print(muni_map["RegionName"].value_counts())
    muni_map_path = OUT_DIR / "municipality_region_map.parquet"
    muni_map.to_parquet(muni_map_path)
    print(f"Saved: {muni_map_path}")

    cap_path = OUT_DIR / "capacity_per_municipality_raw.parquet"
    if cap_path.exists():
        print(f"\n------ Reusing existing raw pull: {cap_path} ------")
        cap = pd.read_parquet(cap_path)
    else:
        print("\n------ Fetching CapacityPerMunicipality (full history) ------")
        cap = fetch_capacity_per_municipality()
        cap.to_parquet(cap_path)
        print(f"Saved: {cap_path}")

    print("\n------ Merging and aggregating to DK1 ------")
    merged = cap.merge(
        muni_map[["MunicipalityCode", "RegionName"]],
        left_on="MunicipalityNo", right_on="MunicipalityCode", how="left",
    )
    unmapped = merged[merged["RegionName"].isna()]["MunicipalityNo"].unique()
    if len(unmapped):
        print(f"WARNING: {len(unmapped)} municipality codes had no region match: {sorted(unmapped)}")
        print("MunicipalityNo=1 confirmed as Thor Offshore Wind Farm (72 turbines, 1058.4 MW,")
        print("22km off Thorsminde, west Jutland coast) -- DK1 territory, not attributable to")
        print("any single municipality. Assigning explicitly to DK1 rather than dropping.")
        merged.loc[merged["MunicipalityNo"] == 1, "RegionName"] = "Region Midtjylland"

    dk1 = merged[merged["RegionName"].isin(DK1_REGIONS)]
    dk1_monthly = (
        dk1.groupby("Month")[["OnshoreWindCapacity", "OffshoreWindCapacity", "SolarPowerCapacity"]]
        .sum(min_count=1)
        .sort_index()
    )
    dk1_monthly["wind_total_capacity_MW"] = (
        dk1_monthly["OnshoreWindCapacity"] + dk1_monthly["OffshoreWindCapacity"]
    )

    print(f"\nDK1 monthly capacity series: {len(dk1_monthly)} months "
          f"({dk1_monthly.index.min()} .. {dk1_monthly.index.max()})")
    print(dk1_monthly.tail(12))

    dk1_path = OUT_DIR / "dk1_wind_capacity_monthly.parquet"
    dk1_monthly.to_parquet(dk1_path)
    print(f"\nSaved: {dk1_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
