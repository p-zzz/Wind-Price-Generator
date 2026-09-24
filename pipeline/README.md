# Advanced tier: full data pipeline

The top-level `generator/` package is ready to run out of the box against the
fitted models and small historical data slices already checked into `models/`
and `data/`. Everything in this `pipeline/` folder is for the much smaller set of
people who want to **retrain the models from scratch** on fresh data, instead of
using the ones already shipped.

You do not need anything here to run the generator. Skip this folder unless you
specifically want to retrain.

## What you need that isn't provided

- An **ENTSO-E Transparency Platform** API token (free registration, then request
  REST API access by emailing transparency@entsoe.eu). Either export it as
  `ENTSOE_TOKEN` (takes precedence; nothing written to disk) or save it as
  `DATA/token.txt` at the repo root (gitignored). Run the download scripts from the
  repo root.
- A **Copernicus Climate Data Store (CDS)** API key (free registration) for ERA5
  wind reanalysis data, configured per `cdsapi`'s usual `~/.cdsapirc` convention.

Neither is shipped here, and no processed data derived from them is either
(beyond the small slices in `data/` at the repo root -- see the root README's
"Data provenance & terms" section for why those specific slices are included
but nothing more).

## Order of operations

```
pipeline/download/   ->  DATA/raw/...
pipeline/build/      ->  DATA/processed/...
pipeline/train/      ->  models/*/fitted/... (your own retrained fitted objects)
pipeline/validate/   ->  DATA/synthetic/... + analysis/outputs/... (sanity-check plots/tables)
```

1. `download/entsoe_download.py`, `download/era5_sites_wind_download.py`,
   `download/energinet_capacity_download.py` -- run manually, one at a time.
   These make real external API calls; nothing here runs them for you.
2. `build/build_price_dataset.py`, `build/era5_sites_builder.py`,
   `build/build_features.py` -- assemble the downloaded raw data into the
   processed parquets the training scripts expect.
3. `train/wind_transformer.py`, `train/wind_capacity_factor_arma.py`,
   `train/solar_model_capacity_factor.py`, `train/price_mdn_v11.py` -- the
   actual fitting/training scripts, unmodified from the thesis repo. Each
   writes its own fitted object(s) under `models/*/fitted/`.
4. `validate/simulate_wind.py` + `validate/validate_wind.py`,
   `validate/simulate_price.py` + `validate/validate_price.py` -- generate
   synthetic paths from what you just trained and diff them against the
   observed data, producing the same regime-broken-down (pre-crisis / crisis /
   post-crisis) plots and tables this project always reports.

None of this touches the ready-to-run `generator/` package or the fitted
objects it ships with in `models/` at the repo root -- if you retrain, you'll
need to point `config/example.yaml`'s `paths.fitted_models_dir` at wherever
your new fitted objects landed (or copy/rename them to match what
`generator/io.py` expects: `wind_transformer_r3.pkl`, `wind_cf_onshore_arma.pkl`,
`wind_cf_offshore_arma.pkl`, `solar_cf.pkl`, `price_mdn_v11.pkl`, plus the
matching `.pt` files for the two PyTorch models).

## Known limitations of this tier

- `train/wind_capacity_factor_arma.py`/`train/price_mdn_v11.py` will produce
  fitted objects with the same statsmodels-pickling bloat problem documented
  in the root README/`scripts/reexport_fitted_objects.py` (multi-hundred-MB to
  multi-GB pkls) -- re-run that script's approach against your own retrained
  objects if you want to keep them small.
- `validate/simulate_price.py` is a trimmed, v11-only version of the thesis
  repo's multi-model comparison harness -- it does not support retraining or
  comparing against the rejected price model variants (flows, v2-v9, v10);
  it exists purely to sanity-check a freshly retrained v11 against observed data.
