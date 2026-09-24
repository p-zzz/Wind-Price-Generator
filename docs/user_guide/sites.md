# Evaluating a wind-farm site

The generator's wind plays **two roles**:

1. **System wind** (`wind_speed_ms`, 10 m at Horns Rev) drives DK1 prices: it stands
   in for how windy DK1 is overall. It is the same whichever farm you evaluate, and
   the price model is trained on it.
2. **Farm wind** drives *your* farm's production. It depends on the farm's location
   and hub height.

Feeding a wind-farm model (such as WinPACT) the Horns Rev wind evaluates a farm **at
Horns Rev**. For a farm anywhere else, use `site_wind_speed_hub_ms`: the wind at your
site, generated together with the system wind, so the farm's output moves with the
prices as the real weather dictates.

```yaml
site:
  name: thor          # a site with a model in data/sites/<name>.json
  hub_height_m: 150
```

Shipped sites: `thor` (the Thor offshore area, off Thorsminde), `ringkobing`
(onshore, west coast) and `herning` (onshore, mid-Jutland, inland).

## Why not just use Horns Rev wind?

Validated against 36 years of ERA5 at a 150 m hub. The site model is driven by the
real Horns Rev history and compared with each site's real ERA5 wind:

| | Thor (offshore) | Ringkøbing (coast) | Herning (inland) |
|---|---|---|---|
| Gross capacity factor\*: true / site model / Horns Rev as the site | 0.616 / 0.615 / 0.599 | 0.563 / 0.563 / 0.599 | 0.449 / 0.450 / **0.599** |
| Farm power vs Horns Rev power, correlation: true / site model | 0.87 / 0.85 | 0.91 / 0.90 | 0.79 / 0.76 |

\* **Gross** capacity factor: the wind run through a generic turbine power curve
(cut-in 3 m/s, rated at 11 m/s, cut-out 25 m/s) with **no** wake, availability or
electrical losses. It is only a yardstick for comparing wind series, not a farm's
expected capacity factor.

Using Horns Rev wind as the site's wind gets the **energy** wrong: it overstates
Herning by a third. It also puts the farm in perfect lockstep with the system wind
(correlation 1.00). Real sites, especially inland, sometimes produce while Horns Rev
is calm, which is when prices are higher, so the shortcut exaggerates
cannibalisation too.

## Inland sites

Two things change inland, and the model reproduces both (checked for Herning, Viborg,
Aalborg and Vojens, 90-200 km from Horns Rev):

- **The link to Horns Rev weakens with distance.** Monthly correlation is 0.87-0.93 at
  Thor and 0.51-0.75 at Aalborg. It's fitted per site, so no tuning is needed.
- **Weather arrives later.** Fronts travel west to east, so inland wind peaks 1-3
  hours after Horns Rev. The site's wind is linked to Horns Rev over a window from
  6 hours before to 3 hours after, so this delay is reproduced; at the farthest
  sites the modelled delay is about an hour short.

The site's own daily cycle (onshore 100 m wind peaks at night) is also reproduced,
to within 0.02 m/s per hour of the day.

## Adding your own site

The site must lie inside the downloaded ERA5 box (roughly 54.75-57.5 N,
7.75-10.0 E: most of DK1 and the near-shore North Sea).

1. Add it to `config/sites.yaml`:

   ```yaml
   sites:
     my_farm:
       lat: 56.10
       lon: 8.60
       description: "Proposed onshore farm near Herning"
   ```

2. Extract its ERA5 wind (needs `pip install -e ".[pipeline]"` once). This takes
   seconds and needs no download if the site is inside the box:

   ```bash
   python pipeline/build/era5_sites_builder.py
   ```

   The raw ERA5 files (`DATA/raw/era5/era5_dk1_*.nc`, about 450 MB for 1990-2026)
   are not in the repo. If you already have them elsewhere, point the builder at them
   with `--raw-dir <folder>`. Otherwise run
   `pipeline/download/era5_sites_wind_download.py` once. It needs a Copernicus CDS key
   and makes about 440 monthly requests, typically a few hours in the CDS queue. It
   also re-downloads any month whose box doesn't cover all sites, so it handles
   sites outside the current box too.

3. Fit and validate the site model:

   ```bash
   python scripts/build_site_model.py --site my_farm
   ```

   It writes `data/sites/my_farm.json` and prints the validation table above for
   your site. Check that the "site model" row matches the "ERA5 truth" row.

4. Use it: `site: {name: my_farm, hub_height_m: <your hub height>}`.

Nothing is retrained: the price and wind models, and therefore all other outputs,
stay exactly as validated.

## Limits

- **ERA5 resolution:** a grid of ~0.25 deg (~28 km). The nearest grid point stands in
  for the site. That's fine offshore; onshore it smooths out local terrain,
  forests and roughness. For a site-specific yield assessment, calibrate against
  measured wind.
- **Co-movement with the system wind is slightly low** (by 0.02-0.03 in
  correlation), so cannibalisation for the farm is slightly understated.
- **Energy in generated scenarios:** within about 2% of the site's ERA5 climate
  (slightly low). The simulated system wind is calibrated on 2014-2025, the site
  models on 1990-2026.
- **Separate random stream:** the site wind uses its own random draws, so adding a
  site changes no other column (see {doc}`reproducibility`).
- **DK1 only:** prices are DK1's. Evaluating a farm in another bidding zone would
  need the whole model stack retrained for that zone.

How it works: {doc}`../theory/wind`.
