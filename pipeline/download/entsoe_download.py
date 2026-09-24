from __future__ import annotations

import os
import time as _time
import random
from pathlib import Path
import requests
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional
from lxml import etree

# Config
BASE_URL = "https://web-api.tp.entsoe.eu/api"
DK1_BZN = "10YDK-1--------W"


# Helpers: time + chunking
def to_entsoe_utc_string(ts: pd.Timestamp) -> str:
    if ts.tz is None:
        raise ValueError("Timestamp must be timezone-aware")
    ts_utc = ts.tz_convert("UTC")
    return ts_utc.strftime("%Y%m%d%H%M")


def yearly_chunks_local_midnight(
    start_local: pd.Timestamp,
    end_local: pd.Timestamp,
    local_tz: str = "Europe/Copenhagen",
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Yield [start, end) 1 year chunks with boundaries at local midnight.
    """
    if start_local.tz is None:
        start_local = start_local.tz_localize(local_tz)
    else:
        start_local = start_local.tz_convert(local_tz)

    if end_local.tz is None:
        end_local = end_local.tz_localize(local_tz)
    else:
        end_local = end_local.tz_convert(local_tz)

    # normalize to local midnight
    start_local = start_local.normalize()
    end_local = end_local.normalize()

    cur = start_local
    while cur < end_local:
        nxt = min(cur + pd.DateOffset(years=1), end_local)
        yield cur, nxt
        cur = nxt


def monthly_chunks_local_midnight(
    start_local: pd.Timestamp,
    end_local: pd.Timestamp,
    local_tz: str = "Europe/Copenhagen",
):
    """
    Yield [start, end) chunks aligned to local midnight at monthly boundaries.
    """

    # ensure timezone
    if start_local.tz is None:
        start_local = start_local.tz_localize(local_tz)
    else:
        start_local = start_local.tz_convert(local_tz)

    if end_local.tz is None:
        end_local = end_local.tz_localize(local_tz)
    else:
        end_local = end_local.tz_convert(local_tz)

    # normalize to midnight
    start_local = start_local.normalize()
    end_local = end_local.normalize()

    current = start_local

    while current < end_local:
        next_month = current + pd.DateOffset(months=1)
        next_month = next_month.normalize()

        chunk_end = min(next_month, end_local)

        yield current, chunk_end

        current = chunk_end


# HTTP with retries
@dataclass
class EntsoeClient:
    token: str
    session: Optional[requests.Session] = None
    timeout_s: int = 60
    max_retries: int = 6

    def __post_init__(self):
        if self.session is None:
            self.session = requests.Session()

    def get(self, params: Dict[str, str]) -> bytes:
        params = dict(params)
        params["securityToken"] = self.token

        last_err = None
        for attempt in range(self.max_retries):
            t0 = _time.time()
            try:
                r = self.session.get(BASE_URL, params=params, timeout=self.timeout_s)
                dt = _time.time() - t0

                if r.status_code == 200 and b"<Acknowledgement_MarketDocument" not in r.content:
                    return r.content

                # Retrying can't fix a rejected token -- fail now instead of backing off 12x.
                if r.status_code in (401, 403):
                    msg = _extract_entsoe_error(r.content) or r.reason
                    raise PermissionError(
                        f"ENTSO-E rejected the API token (HTTP {r.status_code}: {msg}). "
                        f"Check ENTSOE_TOKEN and that REST API access is enabled for the account."
                    )

                if b"<Acknowledgement_MarketDocument" in r.content:
                    msg = _extract_entsoe_error(r.content) or "ENTSO-E API returned an error document."

                    if "No matching data found" in msg:
                        return b""
                    
                    if "Mandatory parameter" in msg or "Input parameter does not exist" in msg:
                        raise ValueError(msg)
                    
                    raise RuntimeError(msg)

                if r.status_code in (429, 503, 502, 504):
                    retry_after = r.headers.get("Retry-After")
                    if retry_after is not None:
                        sleep_s = float(retry_after)
                    else:
                        sleep_s = min(120.0, (2 ** (attempt + 1)) + random.random() * 2)

                    print(f"[HTTP {r.status_code}] retry {attempt+1}/{self.max_retries} "
                        f"after {dt:.1f}s; sleeping {sleep_s:.1f}s")
                    _time.sleep(sleep_s)
                    continue

                r.raise_for_status()
                return r.content

            except (PermissionError, ValueError):
                raise  # bad token / bad request parameters: not transient
            except Exception as e:
                dt = _time.time() - t0
                last_err = e
                sleep_s = min(120.0, (2 ** (attempt + 1)) + random.random() * 2)
                print(f"[EXC] retry {attempt+1}/{self.max_retries} after {dt:.1f}s: {e} "
                    f"(sleep {sleep_s:.1f}s)")
                _time.sleep(sleep_s)

        raise RuntimeError(f"Failed after {self.max_retries} retries: {last_err}")


TOKEN_FILE = Path("DATA/token.txt")


def resolve_token() -> str:
    """ENTSOE_TOKEN env var if set (nothing on disk), else DATA/token.txt (relative
    to cwd, i.e. the repo root; gitignored via DATA/)."""
    token = os.environ.get("ENTSOE_TOKEN", "").strip()
    if token:
        return token
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        if token:
            return token
    raise SystemExit(
        "No ENTSO-E API token found. Either export ENTSOE_TOKEN='...' or save the token "
        f"as {TOKEN_FILE} (run from the repo root; DATA/ is gitignored)."
    )


def _extract_entsoe_error(xml_bytes: bytes) -> Optional[str]:
    try:
        root = etree.fromstring(xml_bytes)
        # namespaces vary; search by local-name
        reasons = root.xpath("//*[local-name()='Reason']//*[local-name()='text']/text()")
        if reasons:
            return " | ".join([r.strip() for r in reasons if r.strip()])
        return None
    except Exception:
        return None


# XML parsing
def parse_timeseries_points(xml_bytes: bytes, dedupe: bool = True) -> pd.DataFrame:
    if not xml_bytes:
        return pd.DataFrame(columns=["value"]).set_index(pd.DatetimeIndex([], name="time_utc"))

    root = etree.fromstring(xml_bytes)

    rows = []
    for ts in root.xpath("//*[local-name()='TimeSeries']"):
        # Some useful metadata
        business_type = _first_text(ts, ".//*[local-name()='businessType']")
        psr_type = _first_text(ts, ".//*[local-name()='MktPSRType']/*[local-name()='psrType']")
        # Flat elements (<in_Domain.mRID>), not nested -- the old nested xpath never
        # matched, which silently dropped the net-position flow direction.
        in_domain = _first_text(ts, ".//*[local-name()='in_Domain.mRID']")
        out_domain = _first_text(ts, ".//*[local-name()='out_Domain.mRID']")
        currency = _first_text(ts, ".//*[local-name()='currency_Unit.name']")
        measure_unit = _first_text(ts, ".//*[local-name()='measurement_Unit.name']")

        for period in ts.xpath(".//*[local-name()='Period']"):
            start = _first_text(period, ".//*[local-name()='timeInterval']/*[local-name()='start']")
            resolution = _first_text(period, ".//*[local-name()='resolution']")
            if start is None or resolution is None:
                continue

            start_ts = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
            step = _resolution_to_timedelta(resolution)

            for point in period.xpath(".//*[local-name()='Point']"):
                pos = _first_text(point, ".//*[local-name()='position']")
                qty = _first_text(point, ".//*[local-name()='price.amount']") or _first_text(point, ".//*[local-name()='quantity']")
                if pos is None or qty is None:
                    continue
                t = start_ts + (int(pos) - 1) * step
                rows.append(
                    {
                        "time_utc": t,
                        "value": float(qty),
                        "businessType": business_type,
                        "psrType": psr_type,
                        "in_Domain": in_domain,
                        "out_Domain": out_domain,
                        "currency": currency,
                        "unit": measure_unit,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["value"]).set_index(pd.DatetimeIndex([], name="time_utc"))

    df = pd.DataFrame(rows).set_index("time_utc").sort_index()

    # In case duplicates, keep the last. Callers whose TimeSeries legitimately share
    # timestamps (net position: one series per flow direction) pass dedupe=False.
    if dedupe:
        df = df[~df.index.duplicated(keep="last")]
    return df


def _first_text(node, xpath: str) -> Optional[str]:
    res = node.xpath(xpath + "/text()")
    if not res:
        return None
    s = str(res[0]).strip()
    return s if s else None


def _resolution_to_timedelta(res: str) -> pd.Timedelta:
    if res == "PT60M":
        return pd.Timedelta(hours=1)
    if res == "PT30M":
        return pd.Timedelta(minutes=30)
    if res == "PT15M":
        return pd.Timedelta(minutes=15)
    raise ValueError(f"Unsupported resolution: {res}")


# Day-ahead prices
def fetch_day_ahead_prices_dk1(
    client: EntsoeClient,
    start_local: pd.Timestamp,
    end_local: pd.Timestamp,
    local_tz: str = "Europe/Copenhagen",
) -> pd.DataFrame:
    all_parts = []
    # yearly -> fast, monthly -> slow
    for s_local, e_local in yearly_chunks_local_midnight(start_local, end_local, local_tz=local_tz):
        params = {
            "documentType": "A44",     # Price document
            "processType": "A01",      # Day-ahead
            "in_Domain": DK1_BZN,
            "out_Domain": DK1_BZN,
            "periodStart": to_entsoe_utc_string(s_local),
            "periodEnd": to_entsoe_utc_string(e_local),
        }
        xml_bytes = client.get(params)
        part = parse_timeseries_points(xml_bytes)
        all_parts.append(part)
        # Add little delay between yearly chunks just in case they mistake me for a dirty bot
        _time.sleep(0.3)

    if not all_parts:
        return pd.DataFrame(columns=["value"]).set_index(pd.DatetimeIndex([], name="time_utc"))

    df = pd.concat(all_parts).sort_index()
    # if duplicates, keep last
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "time_utc"

    df = df.rename(columns={"value": "day_ahead_price"})
    return df


# Fetch load function
def fetch_actual_load_dk1(
    client: EntsoeClient,
    start_local: pd.Timestamp,
    end_local: pd.Timestamp,
    local_tz: str = "Europe/Copenhagen",
) -> pd.DataFrame:

    all_parts = []
    # yearly -> fast, monthly -> slow but safer maybe ?
    for s_local, e_local in monthly_chunks_local_midnight(start_local, end_local, local_tz):

        params = {
            "documentType": "A65",   # System load
            "processType": "A16",    # Realised
            "outBiddingZone_Domain": DK1_BZN,
            "periodStart": to_entsoe_utc_string(s_local),
            "periodEnd": to_entsoe_utc_string(e_local),
        }

        xml_bytes = client.get(params)
        part = parse_timeseries_points(xml_bytes)

        all_parts.append(part)
        # chunk progress print
        print(f"Load chunk {s_local.date()} → {e_local.date()} ...")
        _time.sleep(0.3)

    df = pd.concat(all_parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df = df.rename(columns={"value": "actual_load_MW"})

    return df

def fetch_generation_by_type_dk1(
        client: EntsoeClient,
        start_local: pd.Timestamp,
        end_local: pd.Timestamp,
        local_tz: str = "Europe/Copenhagen",
        chunk: str = "yearly",   # "yearly" or "monthly"
    ) -> pd.DataFrame:
    """
    Aggregated generation per type for DK1.
    Returns LONG-form dataframe indexed by UTC with columns:
      value (MW), psrType, businessType, unit, etc. (as available)
    """
    chunks = yearly_chunks_local_midnight if chunk == "yearly" else monthly_chunks_local_midnight

    all_parts = []
    for s_local, e_local in chunks(start_local, end_local, local_tz=local_tz):
        print(f"Generation chunk {s_local.date()} → {e_local.date()}")

        params = {
            "documentType": "A75",                 # Generation per type
            "processType": "A16",                  # Realised
            "in_Domain": DK1_BZN,
            "periodStart": to_entsoe_utc_string(s_local),
            "periodEnd": to_entsoe_utc_string(e_local),
        }

        xml_bytes = client.get(params)
        part = parse_timeseries_points(xml_bytes)

        print(f"  got {len(part)} rows")
        all_parts.append(part)

        _time.sleep(0.3)

    if not all_parts:
        return pd.DataFrame(columns=["value"]).set_index(pd.DatetimeIndex([], name="time_utc"))

    df = pd.concat(all_parts).sort_index()
    # keep multiple series per timestamp (different psrType)
    # might still get perfect duplicates tho across chunk boundaries (same timestamp + psrType)
    if "psrType" in df.columns:
        df = (
            df.reset_index()
              .drop_duplicates(subset=["time_utc", "psrType"], keep="last")
              .set_index("time_utc")
              .sort_index()
        )
    else:
        df = df[~df.index.duplicated(keep="last")]

    df = df.rename(columns={"value": "gen_MW"})
    return df

PSRTYPE_NAME = {
    # Common ENTSO-E PSR type codes (courtesy of ChatGPT cause I'm not going to look through THAT ¡!¡!)
    # edit: works !!!
    "B01": "biomass",
    "B02": "fossil_brown_coal_lignite",
    "B03": "fossil_coal_derived_gas",
    "B04": "fossil_gas",
    "B05": "fossil_hard_coal",
    "B06": "fossil_oil",
    "B07": "fossil_oil_shale",
    "B08": "fossil_peat",
    "B09": "geothermal",
    "B10": "hydro_pumped_storage",
    "B11": "hydro_run_of_river",
    "B12": "hydro_reservoir",
    "B13": "marine",
    "B14": "nuclear",
    "B15": "other_renewable",
    "B16": "solar",
    "B17": "waste",
    "B18": "wind_offshore",
    "B19": "wind_onshore",
    "B20": "other",
    "B21": "hydro",
}

def pivot_generation_wide(gen_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-form generation to wide-form:
      index: time_utc
      columns: gen_<type>_MW
    """
    if gen_long.empty:
        return pd.DataFrame(index=gen_long.index)

    if "psrType" not in gen_long.columns:
        raise ValueError("Expected 'psrType' column in generation data.")

    df = gen_long.copy()
    df["type_name"] = df["psrType"].map(PSRTYPE_NAME).fillna(df["psrType"])
    wide = (
        df.reset_index()
          .pivot_table(index="time_utc", columns="type_name", values="gen_MW", aggfunc="last")
          .sort_index()
    )

    # prefix columns
    wide.columns = [f"gen_{c}_MW" for c in wide.columns]
    wide.index = pd.DatetimeIndex(wide.index, name="time_utc")
    return wide

def fetch_net_position_dk1(
        client: EntsoeClient,
        start_local: pd.Timestamp,
        end_local: pd.Timestamp,
        local_tz: str = "Europe/Copenhagen",
        chunk: str = "yearly",   # "yearly" or "monthly"
    ) -> pd.DataFrame:

    chunks = yearly_chunks_local_midnight if chunk == "yearly" else monthly_chunks_local_midnight

    all_parts = []
    for s_local, e_local in chunks(start_local, end_local, local_tz=local_tz):
        print(f"Net position chunk {s_local.date()} → {e_local.date()}")

        params = {
            "documentType": "A25",                 # Allocation result doc (contains net position)
            "businessType": "B09",                 # Net position
            "contract_MarketAgreement.type": "A01",# Day-ahead
            "in_Domain": DK1_BZN,
            "out_Domain": DK1_BZN,
            "periodStart": to_entsoe_utc_string(s_local),
            "periodEnd": to_entsoe_utc_string(e_local),
        }

        xml_bytes = client.get(params)
        part = parse_timeseries_points(xml_bytes, dedupe=False)

        # ENTSO-E reports net position as non-negative quantities in one TimeSeries
        # per flow direction: out_Domain = DK1 means energy flows out of DK1 (export,
        # +), in_Domain = DK1 means it flows in (import, -). Sum per timestamp so an
        # hour covered by both directions nets out instead of one being dropped.
        if len(part):
            is_export = part["out_Domain"] == DK1_BZN
            is_import = part["in_Domain"] == DK1_BZN
            ambiguous = is_export == is_import
            if ambiguous.any():
                raise ValueError(
                    f"{int(ambiguous.sum())} net-position points have no single DK1 flow "
                    f"direction (in/out_Domain: "
                    f"{part.loc[ambiguous, ['in_Domain', 'out_Domain']].drop_duplicates().values.tolist()})"
                )
            part["value"] = part["value"].where(is_export, -part["value"])
            part = part.groupby(level=0).agg({"value": "sum", "businessType": "first"})

        print(f"  got {len(part)} rows")
        all_parts.append(part)

        _time.sleep(0.3)

    if not all_parts:
        return pd.DataFrame(columns=["value"]).set_index(pd.DatetimeIndex([], name="time_utc"))

    df = pd.concat(all_parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.rename(columns={"value": "net_position_MW"})
    return df


if __name__ == "__main__":
    import os
    from pathlib import Path

    ###############################################
    # SELECT WHAT TO DOWNLOAD HERE
    DATASETS_TO_DOWNLOAD = [
        "prices",
        "load",
        "generation",
        "net_position",
    ]
    ############################################

    client = EntsoeClient(token=resolve_token(), max_retries=12, timeout_s=60)

    DEFAULT_START = pd.Timestamp("2014-01-01", tz="Europe/Copenhagen")
    END = pd.Timestamp("2026-07-01", tz="Europe/Copenhagen")

    BASE_DIR = Path("DATA/raw")

    # ------ Incremental fetch helpers ------
    def load_existing_raw(path: Path) -> Optional[pd.DataFrame]:
        return pd.read_parquet(path) if path.exists() else None

    def incremental_start(existing: Optional[pd.DataFrame]) -> pd.Timestamp:
        if existing is None or existing.empty:
            return DEFAULT_START
        return existing.index.max().tz_convert("Europe/Copenhagen").normalize()

    def merge_and_dedupe(existing: Optional[pd.DataFrame], new: pd.DataFrame, subset=None) -> pd.DataFrame:
        index_name = new.index.name or "time_utc"
        combined = new if existing is None or existing.empty else pd.concat([existing, new])
        if subset:
            combined = (
                combined.reset_index()
                .drop_duplicates(subset=subset, keep="last")
                .set_index(index_name)
                .sort_index()
            )
        else:
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined

    ############ PRICES ###########
    if "prices" in DATASETS_TO_DOWNLOAD:

        print("Downloading prices...")

        outdir = BASE_DIR / "prices"
        outdir.mkdir(parents=True, exist_ok=True)
        price_path = outdir / "dk1_day_ahead_prices_raw.parquet"

        existing = load_existing_raw(price_path)
        start = incremental_start(existing)

        new_df = fetch_day_ahead_prices_dk1(client, start, END)
        df = merge_and_dedupe(existing, new_df)

        df.to_parquet(price_path)
        df.to_csv(outdir / "dk1_day_ahead_prices_raw.csv")

        print("Prices done:", len(df), "rows total,", len(new_df), "fetched this run")


    ############ LOAD ############
    if "load" in DATASETS_TO_DOWNLOAD:

        print("Downloading load...")

        outdir = BASE_DIR / "load"
        outdir.mkdir(parents=True, exist_ok=True)
        load_path = outdir / "dk1_actual_load_raw.parquet"

        existing = load_existing_raw(load_path)
        start = incremental_start(existing)

        new_df = fetch_actual_load_dk1(client, start, END)
        df = merge_and_dedupe(existing, new_df)

        df.to_parquet(load_path)
        df.to_csv(outdir / "dk1_actual_load_raw.csv")

        print("Load done:", len(df), "rows total,", len(new_df), "fetched this run")


    ######### GENERATION #########
    if "generation" in DATASETS_TO_DOWNLOAD:
        print("Downloading generation...")

        outdir_raw = BASE_DIR / "generation"
        outdir_raw.mkdir(parents=True, exist_ok=True)
        gen_long_path = outdir_raw / "dk1_generation_by_type_raw.parquet"

        existing_long = load_existing_raw(gen_long_path)
        start = incremental_start(existing_long)

        new_long = fetch_generation_by_type_dk1(client, start, END, chunk="yearly")
        gen_long = merge_and_dedupe(existing_long, new_long, subset=["time_utc", "psrType"])
        gen_wide = pivot_generation_wide(gen_long)

        gen_long.to_parquet(gen_long_path)
        gen_long.to_csv(outdir_raw / "dk1_generation_by_type_raw.csv")

        gen_wide.to_parquet(outdir_raw / "dk1_generation_by_type_wide.parquet")
        gen_wide.to_csv(outdir_raw / "dk1_generation_by_type_wide.csv")

        print("Generation done:",
            len(gen_long), "raw rows total,", len(new_long), "fetched this run;",
            gen_wide.shape[0], "timestamps;",
            gen_wide.shape[1], "columns")

        print(gen_long["psrType"].value_counts().head(20))
        print(gen_wide.columns)
        print(gen_wide.index.to_series().diff().value_counts().head())


    ######## NET POSITION #########
    if "net_position" in DATASETS_TO_DOWNLOAD:

        print("Downloading net position...")

        outdir_raw = BASE_DIR / "net_position"
        outdir_raw.mkdir(parents=True, exist_ok=True)
        netpos_path = outdir_raw / "dk1_net_position_raw.parquet"

        existing = load_existing_raw(netpos_path)
        start = incremental_start(existing)

        new_netpos = fetch_net_position_dk1(client, start, END, chunk="yearly")
        netpos = merge_and_dedupe(existing, new_netpos)

        netpos.to_parquet(netpos_path)
        netpos.to_csv(outdir_raw / "dk1_net_position_raw.csv")

        print("Net position done:", len(netpos), "rows total,", len(new_netpos), "fetched this run")

        print(netpos.index.to_series().diff().value_counts().head())
        print(netpos.describe())


    print("All requested downloads finished.")