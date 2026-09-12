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

Output: a CSV under `outputs/` with hourly `wind_speed_ms`, `wind_generation_MW`,
`solar_generation_MW`, `actual_load_MW`, `net_position_MW`, `gas_price_eur_mwh`,
and `day_ahead_price` columns, plus a printed validation table (marginal stats +
Spearman(wind speed, price), which should land near -0.45 in the post-crisis
regime -- see "Known limitations" below for when it won't).

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
validated range this is trustworthy over.

## Known limitations

- **Price MDN v11's response to solar reverses sign around `solar.scale≈3.0x`.**
  Confirmed both by an isolated single-feature probe and by the full correlated
  generator (stratified by season/day-night): mean price troughs at
  `solar.scale≈3.0x` in every season tested, then rises again at 4x/5x. Treat
  any run with `solar.scale` above roughly 2.0-2.5x as outside the validated
  region -- `config/example.yaml` warns at 2.5x and above. Wind's equivalent
  effect is far smaller (a sub-1-EUR/MWh wobble in one seasonal context) and
  not treated as a practical concern.
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
  gas level, worst in the crisis regime (ρ -0.45 -> -0.38 vs. a real -0.54).
  Prefer flat mode over trajectory mode when the wind-price correlation is
  what you actually care about getting right, at the cost of std -- flat gas
  undershoots the real crisis-regime price std (107 vs. a real 145) that the
  noisier modes get closer to.
- **Load and net position are drawn from real 2023-2025 history via a paired
  block bootstrap**, not a fitted parametric model -- this preserves their real
  joint distribution and autocorrelation structure but means simulated paths
  cannot express a future load/net-position regime that never occurred in that
  window.

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
  position, prices)
- [Copernicus Climate Data Store / ERA5 reanalysis](https://cds.climate.copernicus.eu/)
  (wind speed)
- [Energinet CapacityPerMunicipality](https://energinet.dk/) (installed
  capacity baseline, baked into `config/example.yaml`)

This repo's code is MIT-licensed (see `LICENSE`); that license covers the code,
not necessarily unrestricted redistribution of data derived from the sources
above under their own terms. **This has not been independently confirmed with
the data sources' terms of use or with the thesis supervisors before any public
release of this repo -- treat that as an open item, not a settled one.**

## Citation

If you use this in academic work, please cite the originating thesis:

> Pablo Zan Nieto, "Modeling wind-price boundary conditions for wind farm project evaluation", MSc Thesis, Technical University of Denmark
> (DTU), 2026.

## License

MIT -- see `LICENSE`.
