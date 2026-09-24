# Outputs

## Layout

`generator.run.generate` returns a pandas DataFrame in **long format**: one row per
hour per path, `n_paths x horizon_hours` rows in total. The index holds the hourly
timestamps, which **repeat once per path**; the `path` column (0-based) tells paths
apart. `generator.run.save_scenarios` writes the same table to
`<output_dir>/scenarios_<YYYYmmdd_HHMMSS>.csv`, with the timestamps as the first
column, `timestamp`.

## Time

- **Hourly, Europe/Copenhagen local time**, starting at `generator.start_date`.
- Steps are real elapsed hours. On the spring daylight-saving day the local hour
  02:00 is missing, and on the autumn day it appears twice. This matches the
  training data.
- In the CSV every timestamp carries its UTC offset (for example
  `2024-03-31 03:00:00+02:00`), so it converts to UTC unambiguously:
  `pd.to_datetime(df["timestamp"], utc=True)`.

## Columns

| Column | Unit | Meaning |
|---|---|---|
| `path` | - | Path number, 0 to `n_paths - 1`. |
| `wind_speed_ms` | m/s | Wind speed at **10 m** at the Horns Rev ERA5 point (Danish North Sea). The price model's wind input. |
| `wind_speed_hub_ms` | m/s | The same Horns Rev wind at `wind.hub_height_m` (default 150 m). Absent if `hub_height_m: null`. |
| `site_wind_speed_hub_ms` | m/s | Wind at the farm site `site.name` and its hub height. **Use this for a farm's power.** Only if `site` is set; see {doc}`sites`. |
| `wind_generation_MW` | MW | DK1 wind fleet output at scaled capacity (onshore + offshore). |
| `wind_generation_onshore_MW`, `wind_generation_offshore_MW` | MW | The two components. |
| `solar_generation_MW` | MW | DK1 solar fleet output at scaled capacity. Deterministic. |
| `solar_load_ratio`, `wind_load_ratio` | - | Generation divided by load: the price model's renewable inputs. |
| `actual_load_MW` | MW | DK1 load, bootstrapped from 2023-2025 history. |
| `net_position_MW` | MW | DK1 net position: **positive = export**, negative = import. Bootstrapped together with load. |
| `gas_price_eur_mwh` | EUR/MWh | TTF gas price (shared by all paths). |
| `day_ahead_price` | EUR/MWh | DK1 day-ahead price: one random draw per hour from the price model's conditional distribution. |

```{note}
`day_ahead_price` is **not clipped** to the market's price limits: rare draws can
fall well outside the range seen in practice. For expected values, average over
paths rather than relying on single extreme hours.
```

## Validation printout

The command line (and `generator.run.print_validation`) prints, as mean +/- std
across paths:

- **Wind speed:** mean, std, Weibull shape `k`, autocorrelation at 24 h and 48 h.
- **Price:** mean, std, % negative hours overall and in May-June.
- **Spearman(wind speed, price)**, over all hours and per gas regime.

Expect the Spearman correlation to be **negative**: the merit-order effect, where more
wind means lower prices. Its size depends on the inputs (gas mode and noise, scale
knobs, mix of regimes), so there is no single target value. Compare it within a gas
regime rather than across all hours.

Wind-speed statistics barely vary across paths. That's by design: each path is
mapped onto the long-run climatological wind distribution (see {doc}`../theory/wind`).
