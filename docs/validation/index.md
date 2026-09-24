# Validation

All figures come from the shipped 10-seed ensemble, 2026-09.

## Against real DK1 data

Reference scenario: today's capacity (both scales 1.0), flat post-crisis gas
(45 EUR/MWh), 20 years, 3 paths.

| Quantity | Generator | Real DK1 |
|---|---|---|
| Mean day-ahead price | 72.0 EUR/MWh | about 71 (2024), about 81 (2025) |
| Offshore wind capture rate | 0.83 | 0.86 (2023), 0.85 (2024) |
| Onshore wind capture rate | 0.75 | 0.75 (2023), 0.76 (2024) |
| Spearman(wind speed, price) | -0.44 | negative (merit-order effect) |

The **capture rate** is the price wind actually earns (generation-weighted) divided by
the average price. Matching it matters most for this tool's purpose, the
cannibalisation of wind revenue, and it matches real DK1 within a few hundredths.

On its own historical inputs, the price model's expected price runs **1-7 EUR/MWh
below** the observed mean per half-year from 2023 H2 to 2025 H1, and about 18
below in 2025 H2 (see {doc}`../theory/price_model`).

## Scale-knob sweep

`solar.scale` 1-5x, `wind.scale` 1-5x and both together 1.25-3x, each with 5 years,
3 paths, flat post-crisis gas and seed 42 (so the weather and load are identical
across points).

| Scale | Mean price, solar knob | Mean price, wind knob | Solar capture | Wind capture |
|---|---|---|---|---|
| 1x | 71.9 | 71.9 | 0.59 | 0.79 |
| 1.25x | 68.4 | 68.5 | 0.49 | 0.76 |
| 1.5x | 65.8 | 65.5 | 0.42 | 0.74 |
| 2x | 62.3 | 60.7 | 0.33 | 0.70 |
| 3x | 58.6 | 54.4 | 0.28 | 0.68 |
| 4x | 57.4 | 50.6 | 0.30 | 0.68 |
| 5x | 57.0 | 47.6 | 0.35 | 0.69 |

(Each capture rate is for the knob being scaled.)

- **Prices fall steadily** with more capacity, and capture rates fall faster than
  prices: the cannibalisation the tool is built to capture.
- **Beyond ~3x the response saturates:** mean price barely moves, and capture rates
  stop falling, with solar's even rising. Physically, more capacity should keep
  lowering its own capture rate, so this marks where the model runs out of training
  data.
- **How far outside the training data:** at 1.25x about 1% of hours have a
  renewable/load ratio beyond anything seen in 2015-2025; at 2x, 12-13%; at 3x,
  22-33%. The ensemble members' disagreement grows alongside: +15% (solar) and +64%
  (wind) at 2x.

Hence the in-domain limit of **~1.25x**, with results beyond ~2x outside the
validated region. See {doc}`../user_guide/scenarios`.

## Model stability

- **Seed ensemble:** two separate 5-seed ensembles (seeds 1-5 and 6-10) agree to
  within 0.5 EUR/MWh on a scenario's mean price. The single models of the earlier
  configuration were 2.8 EUR/MWh apart. See {doc}`../theory/price_model`.
- **Reproducibility:** retraining any member with its recorded seed reproduces its
  weights bit for bit.

## Build pipeline

Rebuilding the processed training data from the raw downloads reproduces the thesis
feature file bit for bit (100,268 hours x 22 columns), so the retraining pipeline
(see {doc}`../retraining`) is faithful to the original.
