# DK1 Joint Scenario Generator

A stochastic scenario generator for the DK1 (Denmark West) bidding zone: produces
realistic hourly time series of wind speed, wind generation, solar generation,
load, net position, and day-ahead electricity price, over a multi-year horizon.
Built for wind farm revenue and cannibalisation-risk analysis under different
renewable-penetration and gas-price scenarios.

Extracted from the wind/price joint-behavior modelling work of a DTU MSc thesis
(2026). This repo ships only the final, production model stack -- not the
research trail of rejected model variants, tuning runs, and diagnostics that
led to it.

## What this is / isn't

This is a **stochastic scenario generator**, not a forecaster. It does not
predict what wind and price will actually do on a given future date -- it draws
statistically realistic *paths* consistent with each model's own trained
distribution, conditioned on the scenario knobs you set (installed capacity
scale, gas price regime). Each run represents one fixed state of the world
(e.g. "2x today's solar capacity, crisis-level gas") applied uniformly over the
whole horizon, not a year-by-year build-out trajectory.

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .

python -m generator.run --config config/example.yaml
```

Output: a CSV under `outputs/` with hourly `wind_speed_ms` (10 m, see below),
`wind_speed_hub_ms` (at `wind.hub_height_m`, default 150 m), `wind_generation_MW`,
`solar_generation_MW`, `actual_load_MW`, `net_position_MW`, `gas_price_eur_mwh`,
and `day_ahead_price` columns, plus a printed validation table: marginal stats,
and Spearman(wind speed, price) over all hours and per gas regime. Expect it to
be negative -- the merit-order effect (more wind, lower price) is preserved --
but its strength depends on which inputs the price model sees (gas mode and
noise, scale knobs, regime mix), so there is no single target value; compare
per regime rather than the pooled figure.

Edit `config/example.yaml` to change the horizon, random seed, `wind.scale`/
`solar.scale` (multipliers on installed capacity), and the gas price schedule.
See the comments in that file for valid options.

The shipped Price MDN v11 was trained on data through end of 2025. If you want
scenarios anchored to current market conditions, pull fresh ENTSO-E/ERA5/gas
data and retrain via `pipeline/` first -- otherwise this reflects the pre-2026
regime the model actually learned.

## Model stack

```
Wind Transformer + quantile mapper     -> wind_speed_ms          (feeds price model, unscaled)
Wind CF ARMA (onshore + offshore)      -> wind_generation_MW     (x wind.scale)
Solar CF model                         -> solar_generation_MW    (x solar.scale)
Paired block bootstrap (real 2023-2025 data) -> actual_load_MW, net_position_MW
Gas price (flat level or multi-regime trajectory) -> gas_price_eur_mwh
                                                    |
                                                    v
                              Price MDN v11 -> day_ahead_price (conditional distribution draw)
```

`wind_speed_ms` feeds the price model's merit-order channel directly and
unscaled -- `wind.scale` never touches it, since scaling wind speed has no
physical meaning as a capacity proxy. `wind_generation_MW`/`solar_generation_MW`
are separate, independently-scaled streams: simulated capacity factor x target
installed capacity x scale. Price MDN v11 takes `solar_load_ratio`/
`wind_load_ratio` (the scaled MW streams divided by simulated load), so
`wind.scale`/`solar.scale` do move price -- see "Known limitations" for the
in-domain range this is trustworthy over.

## Known limitations

- **`wind_speed_ms` is 10 m wind at a single ERA5 point (Horns Rev), not hub
  height.** Every model is trained on it, and it drives the price model as a
  merit-order signal. For turbine power or energy use `wind_speed_hub_ms`
  (set `wind.hub_height_m` to your turbine's hub height; default 150 m, typical
  for a new ~15 MW offshore turbine), which extrapolates it with the wind-shear exponent
  measured from ERA5's own 10 m / 100 m pair at Horns Rev, per month and wind
  speed (`data/horns_rev_shear.csv`, built by `scripts/build_shear_table.py`;
  about 0.10 on average, 0.05-0.12 by cell). Don't apply a generic onshore
  exponent such as 0.2 to the 10 m wind: offshore it overstates hub-height wind
  by about a third (≈13.4 vs ≈10 m/s at 170 m) and turbine capacity factor by
  far more than any realistic site achieves. Both columns are offshore Horns Rev
  wind: fine for DK1 offshore farms, but not a substitute for measured wind at
  a specific onshore site.
- **The shipped price model is in-domain only up to ~1.25x scale.** Beyond
  that, aggregate statistics are usable with caution, but individual hours are
  extrapolated and should not be trusted. Joint multi-knob scenarios (e.g.
  `wind.scale` and `solar.scale` both raised) were not validated beyond 1.25x.
  The generator prints a warning when either knob is above 1.25x.
- **Solar scaling has a confirmed non-monotonic price response.** Confirmed both
  by an isolated single-feature probe and by the full correlated generator
  (stratified by season/day-night): mean price troughs near `solar.scale≈3x` in
  every season tested, then rises again at 4x/5x. Wind's equivalent effect is
  far smaller (a sub-1-EUR/MWh wobble in one seasonal context) and not treated
  as a practical concern.
- **Solar has no stochastic residual.** `solar_generation_MW` is a deterministic
  seasonal-mean capacity factor -- real cloud-driven day-to-day variance exists
  in the training data but is not reproduced in simulated paths. Wind does have
  this (an ARMA residual on top of the seasonal mean); solar does not.
- **Gas price is a schedule, not a market model.** The trajectory mode is linear
  ramps between regime levels plus AR(1) noise -- it does not respond to
  anything else in the simulation (wind, solar, price) and is not a model of
  the gas market itself. Internal testing found that layering AR(1)/deviation-
  bootstrap noise onto gas (the stochastic ingredient behind trajectory mode)
  measurably weakens the wind-price merit-order correlation relative to a flat
  gas level, most of all in the crisis regime, where it already falls short of
  the observed correlation.
  Prefer flat mode over trajectory mode when the wind-price correlation is
  what you actually care about getting right, at the cost of std -- flat gas
  undershoots the real crisis-regime price std (107 vs. a real 145) that the
  noisier modes get closer to.
- **Load and net position are drawn from real 2023-2025 history via a paired
  block bootstrap**, not a fitted parametric model -- this preserves their real
  joint distribution and autocorrelation structure but means simulated paths
  cannot express a future load/net-position regime that never occurred in that
  window.
- **The price model is a 10-seed ensemble, and some model noise remains.**
  Single training runs of Price MDN v11 (same data, different seed) agree on
  historical inputs but diverge on generated scenarios, which combine inputs
  (e.g. flat gas with 2026 installed capacity) the 2015-2025 data barely
  covers. The shipped model therefore pools 10 seeds with equal weight
  (`models/price_mdn_v11_seed{1..10}.pt`; rebuild with
  `pipeline/train/price_mdn_v11.py` per `PRICE_MDN_SEED` +
  `pipeline/train/assemble_price_ensemble.py`), using hyperparameters chosen
  for low seed-to-seed spread. Residual model noise is roughly ±0.4 EUR/MWh on
  a scenario's mean price (about ±20 M EUR NPV for the reference WinPACT
  case); treat smaller scenario differences as not meaningful.
- **Net position sign convention: export positive, import negative.** Earlier
  versions of this repo shipped a price model and `data/bootstrap_pool.parquet`
  built on *unsigned* net-position magnitudes, because the ENTSO-E download
  parser dropped the flow direction. Both have been rebuilt from the fixed
  parser; scenarios generated with older versions have non-negative
  `net_position_MW` and are not comparable on this column.

## Repository layout

- `generator/` -- the installable core package. Everything needed to run a
  scenario is here plus `models/` and `data/`.
- `models/` -- fitted model weights, re-exported from the source thesis repo
  (see `scripts/reexport_fitted_objects.py` for exactly how, including a
  verification step that the re-export is numerically identical to the
  original before being trusted).
- `data/` -- small historical data slices the generator needs at runtime (an
  AR-buffer seed for the wind Transformer, a load/net-position bootstrap pool).
  See "Data provenance & terms" below.
- `config/example.yaml` -- the scenario configuration surface.
- `examples/run_example.py` -- a runnable, short-horizon smoke test that
  doubles as a minimal Python API usage example.
- `pipeline/` -- optional advanced tier: the full download -> build -> train ->
  validate stack, for anyone who wants to retrain the models from scratch
  instead of using the fitted weights already shipped. Not needed to use the
  generator. See `pipeline/README.md`.

## Data provenance & terms

`data/wind_speed_seed.parquet` (last ~500 hours of Horns Rev 10m wind speed) and
`data/bootstrap_pool.parquet` (2023-2025 hourly load and net position) are small
slices of data ultimately sourced from:

- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) (load, net
  position, prices). Source: ENTSO-E Transparency Platform
  (transparency.entsoe.eu).
- [Copernicus Climate Data Store / ERA5 reanalysis](https://cds.climate.copernicus.eu/)
  (wind speed). Contains modified Copernicus Climate Change Service information
  2026. Neither the European Commission nor ECMWF is responsible for any use
  that may be made of the Copernicus information or data it contains.
- [Energinet Energi Data Service](https://www.energidataservice.dk/) (installed
  capacity baseline in `config/example.yaml`). Licensed under
  [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source: Energinet
  (www.energidataservice.dk).

This repo's code is MIT-licensed (see `LICENSE`), separately from the data
terms above.

## Citation

If you use this in academic work, please cite the originating thesis:

> Pablo Zan Nieto, "Modeling wind-price boundary conditions for wind farm project evaluation", MSc Thesis, Technical University of Denmark
> (DTU), 2026.

## License

MIT -- see `LICENSE`.
