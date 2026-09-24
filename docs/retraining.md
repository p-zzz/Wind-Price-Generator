# Retraining (advanced)

You don't need this to use the generator: the fitted models ship in `models/`. The
`pipeline/` tier rebuilds them from raw data, for example to add newer years.

## What you need

- An **ENTSO-E Transparency Platform** API token. Register, then request REST API
  access by emailing transparency@entsoe.eu. Provide the token either as the
  `ENTSOE_TOKEN` environment variable (takes precedence; nothing is written to disk)
  or in `DATA/token.txt` at the repo root (gitignored).
- A **Copernicus CDS** API key (`~/.cdsapirc`) for ERA5 wind.
- The pipeline extras: `pip install -e ".[pipeline]"`.

Run everything from the repo root; scripts use paths relative to it (`DATA/raw`,
`DATA/processed`, `models/*/fitted`).

## Steps

1. **Download:** `pipeline/download/entsoe_download.py` (prices, load, generation,
   net position), `era5_sites_wind_download.py`, `energinet_capacity_download.py`.
   A rejected token stops the download immediately with a clear message. The ERA5
   script downloads a box around all sites in `config/sites.yaml`, re-downloading
   cached months that don't cover them, then extracts per-site wind with
   `pipeline/build/era5_sites_builder.py`. Rerun the builder alone after adding a site
   inside the box.
2. **Build:** `pipeline/build/build_price_dataset.py`, then `build_features.py`.
3. **Train** the component models in `pipeline/train/`.
4. **Train the price ensemble:** one run per seed.

   ```bash
   for s in 1 2 3 4 5 6 7 8 9 10; do
       PRICE_MDN_SEED=$s python pipeline/train/price_mdn_v11.py   # writes models/price/fitted/
       # copy each run's price_mdn_v11.pkl + price_mdn_v11_best.pt to its own folder
   done
   python pipeline/train/assemble_price_ensemble.py \
       --member runs/seed1 --member runs/seed2 ... --out models/
   ```

   The assembly script refuses members trained on different data or settings.
5. **Refresh the data slices:**
   `python scripts/extract_data_slices.py --source <root with DATA/processed>` and
   `python scripts/build_shear_table.py --source <root>`. The bootstrap pool must come
   from the same processed data the price model was trained on.
6. **Validate** with `pipeline/validate/`, then rerun the scale sweep in
   {doc}`validation/index`.

```{note}
Each price-model run takes about a minute on a laptop CPU; no GPU is needed. Keep
the training settings in `price_mdn_v11.py` unless you rerun a sweep: they were
chosen for low seed-to-seed spread on scenario inputs (see
{doc}`theory/price_model`).
```
