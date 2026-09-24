# Changelog

## 2026-09

**Correctness fixes**
- **Wind calendar:** the wind simulation now follows `start_date`. Before, it always
  ran on a January 2030 calendar, so wind seasonality didn't match solar, load and
  price when `start_date` wasn't January 1.
- **Calendar-matched bootstrap:** load and net-position blocks now come from the same
  hour of day and time of year. Blocks never cross data gaps or daylight-saving
  changes. Before, a January load block could land in July, and daily peaks could
  land at night.
- **Signed net position:** the ENTSO-E download parser dropped the flow direction,
  so the price model and bootstrap pool used unsigned magnitudes. Both are rebuilt
  with export-positive values, and the resulting model captures the merit-order
  effect more strongly.
- **ERA5 builder:** `pipeline/build/era5_sites_builder.py` was obsolete. It looked
  for per-site files from an older download layout and averaged the whole grid box.
  It now extracts the nearest grid point per site from the current files.
- **Price-model input order:** checked when the models load. A mismatch used to feed
  inputs in the wrong order without any error.

**Model**
- **Price model** retrained on signed net position with tuned settings (layers
  [128, 128, 64], weight decay $10^{-2}$), and shipped as a **10-seed ensemble**.
  This removes most of the model-choice noise that made single models' results
  depend on their random seed.

**New features**
- **Farm-site wind:** a `site:` block adds `site_wind_speed_hub_ms`, the wind at a
  chosen DK1 site, consistent with the system wind that drives prices. Sites live in
  `config/sites.yaml`; `scripts/build_site_model.py` fits and validates them.
  `thor`, `ringkobing` and `herning` are shipped; validated offshore, on the coast
  and inland, including the 1-3 h delay with which weather reaches inland sites. The ERA5 pipeline reads the same file, and
  now refuses month files that don't cover every site (before, it silently snapped
  to the box edge).
- **`wind_speed_hub_ms`:** wind at `wind.hub_height_m` (default 150 m), using wind
  shear measured from ERA5 at Horns Rev.
- **Validation printout:** Spearman(wind, price) is reported per gas regime.
- **Pipeline:** training is seeded, and `scripts/extract_data_slices.py` and
  `scripts/build_shear_table.py` take `--source`. The download fails fast on a
  rejected token, and the token can come from `ENTSOE_TOKEN` or `DATA/token.txt`.

**Documentation**
- NumPy docstrings for the public API, and this site.
- In-domain limits re-established on the ensemble. The previous model's solar-price
  reversal near 3x no longer occurs; instead the response saturates beyond ~3x.
