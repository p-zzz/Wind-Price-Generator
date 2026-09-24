# Model overview

Each path is built in six steps; later steps condition on earlier ones:

```text
1. Wind Transformer + quantile mapper  -> wind_speed_ms (10 m, Horns Rev)
                                          wind_speed_hub_ms (hub height, output only)
2. Wind capacity-factor ARMA           -> wind_generation_MW   (x wind.scale)
3. Solar capacity-factor shape model   -> solar_generation_MW  (x solar.scale)
4. Paired block bootstrap (2023-2025)  -> actual_load_MW, net_position_MW
5. Gas schedule (flat or trajectory)   -> gas_price_eur_mwh
6. Price mixture density network       -> day_ahead_price
   inputs: calendar, wind speed, solar/load, wind/load, load, net position, gas
```

The key design choice is that **the scale knobs act on capacity, not on the
weather**. Wind speed is simulated once and never scaled; `wind.scale` and
`solar.scale` multiply installed capacity, so generation and its ratio to load change.
The price model reads those ratios, so prices respond to capacity while the weather
stays the same.

Joint behaviour comes from shared drivers rather than one big joint model:

- **Wind generation and price** both depend on the same simulated wind speed.
- **Load and net position** are resampled together, so their real co-movement is kept.
- **All models** see the same local-time calendar (season, hour of day).

```{toctree}
:maxdepth: 1

wind
solar
load_bootstrap
gas
price_model
```
