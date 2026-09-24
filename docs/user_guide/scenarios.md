# Designing scenarios

Each run represents **one fixed state of the world** applied uniformly over the
whole horizon, for example "1.25x today's wind capacity with post-crisis gas". It is
not a year-by-year build-out. To compare states, run one scenario per state with the
same seed (see {doc}`reproducibility`).

## Capacity scale knobs

`wind.scale` and `solar.scale` multiply the installed-capacity baselines (DK1 as of
2026-06-01 in the example config). They change:

- generation (`wind_generation_MW`, `solar_generation_MW`),
- the renewable share of load (`wind_load_ratio`, `solar_load_ratio`), which the
  price model reads, so **prices respond**: more capacity lowers prices, and lowers
  the capture rate of that technology even more (cannibalisation).

They do **not** change the simulated wind speed or the random draws.

### How far can you push them?

The price model learned from 2015-2025. At 1x (today's capacity) the renewable
ratios are already near the top of that range. A scale sweep on the shipped
ensemble (5 years, 3 paths, flat post-crisis gas) gives:

| Scale | Hours beyond the training range, solar / wind | Ensemble disagreement vs 1x, solar / wind |
|---|---|---|
| 1.25x | 1.2% / 0.4% | +2% / +14% |
| 1.5x | 4.4% / 2.2% | +6% / +30% |
| 2x | 13% / 12% | +15% / +64% |
| 3x | 22% / 33% | +37% / +123% |

Recommended use:

- **Up to ~1.25x:** in-domain.
- **~1.25x to 2x:** aggregate statistics (annual means, capture rates) usable with
  caution; individual hours are extrapolated.
- **Beyond ~2x:** outside the validated region. Beyond ~3x the price response
  saturates and capture rates stop falling (solar's rises again, from 0.28 at 3x to
  0.35 at 5x), which is an extrapolation artifact rather than physics.

Raising both knobs together behaves like the more extrapolated of the two.
Background: {doc}`../validation/index`.

## Gas price

- **Flat** at a historical regime level (`pre_crisis` 35, `crisis` 150,
  `post_crisis` 45 EUR/MWh) or a custom value. Best when the wind-price relationship
  is what matters.
- **Trajectory:** regimes in sequence with ramps and AR(1) noise, for price-level
  paths that change over time. The noise weakens the wind-price correlation
  somewhat, most of all in the crisis regime.

Gas is a schedule: it doesn't react to wind, solar or price. See {doc}`../theory/gas`.

## Start date and horizon

`start_date` sets the calendar (season, hour of day, weekday) every model conditions
on, so a scenario starting in July begins in summer conditions. Match it to your
project's timeline if you use the output alongside another tool.

## Number of paths

The price is a random draw per hour, so single paths vary. Use several paths (10 or
more for summary statistics) and report means with their spread across paths.
Beyond path-to-path variation, the price model itself carries some uncertainty:
about +/-0.4 EUR/MWh on a scenario's mean price (see {doc}`../theory/price_model`).
