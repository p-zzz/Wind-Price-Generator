# Configuration

A scenario is one YAML file (start from `config/example.yaml`), read by
`generator.config.Config.from_yaml`. The same settings can be built in Python with
the dataclasses in `generator.config` (see `examples/run_example.py`).

## `generator`

| Key | Default | Meaning |
|---|---|---|
| `horizon_hours` | required | Simulation length in hours (8760 = 1 year, 43800 = 5 years). |
| `n_paths` | required | Number of independent stochastic paths. |
| `random_seed` | required | Seed of the single random stream; see {doc}`reproducibility`. |
| `block_size` | required | Load/net-position bootstrap block length in hours (168 = one week). |
| `bootstrap_window_days` | `14` | Bootstrap blocks come from the same local hour of day and within this many days of the calendar date they fill. |
| `start_date` | `"2024-01-01"` | First hour of the output, local time (Europe/Copenhagen). Sets the calendar every model conditions on, so match it to your project's timeline. |

## `wind`

| Key | Default | Meaning |
|---|---|---|
| `scale` | required | Multiplier on installed onshore + offshore capacity. Changes wind generation and, through it, the price. Never changes wind speed. |
| `onshore_capacity_mw` | required | Installed onshore capacity baseline, MW (example: 4162.6, DK1 on 2026-06-01, Energinet). |
| `offshore_capacity_mw` | required | Installed offshore capacity baseline, MW (example: 2661.7). |
| `hub_height_m` | `150` | Height of the `wind_speed_hub_ms` output column, m. Set it to your turbine's hub height; `null` drops the column. |

Typical hub heights: onshore 4-6 MW about 100-130 m; offshore 8-11 MW about
110-140 m; 14-15 MW about 140-150 m; 20-22 MW about 160-170 m.

## `site` (optional)

Wind at a wind-farm site, for evaluating a farm there. See {doc}`sites`.

| Key | Default | Meaning |
|---|---|---|
| `name` | required | A site with a fitted model in `data/sites/<name>.json`. Shipped: `thor`, `ringkobing`, `herning`. |
| `hub_height_m` | `150` | Hub height at the site, m. |

Leave the block out to skip the `site_wind_speed_hub_ms` column.

## `solar`

| Key | Default | Meaning |
|---|---|---|
| `scale` | required | Multiplier on installed solar capacity. Changes solar generation and the price. |
| `capacity_mw` | required | Installed solar capacity baseline, MW (example: 3881.1). |

```{warning}
The price model is in-domain only up to **~1.25x** on `wind.scale` and
`solar.scale`, alone or together. The generator prints a warning above that. See
{doc}`scenarios`.
```

## `gas`

`mode` is `"flat"` or `"trajectory"`.

**Flat** (`gas.flat`): one constant price for the whole horizon.

| Key | Default | Meaning |
|---|---|---|
| `regime` | `"post_crisis"` | `pre_crisis` (35 EUR/MWh), `crisis` (150), `post_crisis` (45) or `custom`. |
| `custom_eur_mwh` | `40.0` | Price used when `regime: custom`, EUR/MWh. |

**Trajectory** (`gas.trajectory`): a sequence of regimes with ramps and noise.

| Key | Default | Meaning |
|---|---|---|
| `segments` | `[]` | List of `{regime, years}` walked in order (1 year = 8760 h). Segments past the horizon are ignored; a shorter schedule holds its last level. |
| `ramp_hours` | `720` | Linear ramp between two regime levels, hours. |
| `noise_std` | `3.0` | Stationary standard deviation of the AR(1) noise, EUR/MWh. |
| `noise_ar` | `0.99` | AR(1) coefficient (0.99 gives slow, week-scale wander). |
| `noise_floor` | `10.0` | Lower bound on the gas price, EUR/MWh. |

See {doc}`../theory/gas` for the trade-off between the two modes.

## `paths`

| Key | Meaning |
|---|---|
| `fitted_models_dir` | The fitted models (the repo's `models/`). |
| `data_dir` | The historical data slices (the repo's `data/`). |
| `output_dir` | Where scenario CSVs are written; created if missing. |

Relative paths are resolved against the directory you run from.
