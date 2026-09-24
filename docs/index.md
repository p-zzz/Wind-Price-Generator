# DK1 Scenario Generator

A stochastic generator of **joint hourly scenarios** for the DK1 (Denmark West)
bidding zone: wind speed, wind and solar generation, load, net position, gas price
and day-ahead electricity price, over horizons of years. It is built for wind-farm
revenue and cannibalisation-risk analysis under different renewable-capacity and
gas-price assumptions.

It is a **scenario generator, not a forecaster**: it draws statistically realistic
paths consistent with what its models learned from 2015-2025 data, for one fixed
state of the world per run (for example "1.25x today's wind capacity, post-crisis
gas"). It does not predict what prices will be on a given date.

```{toctree}
:maxdepth: 2
:caption: Getting started

getting_started/installation
getting_started/quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide

user_guide/configuration
user_guide/outputs
user_guide/scenarios
user_guide/sites
user_guide/reproducibility
```

```{toctree}
:maxdepth: 2
:caption: Theory

theory/index
theory/wind
theory/solar
theory/load_bootstrap
theory/gas
theory/price_model
```

```{toctree}
:maxdepth: 2
:caption: Validation & limits

validation/index
validation/known_limitations
```

```{toctree}
:maxdepth: 1
:caption: Reference

retraining
api/index
changelog
```

## At a glance

| Output | Unit | Source |
|---|---|---|
| `wind_speed_ms` | m/s, 10 m | Transformer mixture model + seasonal Weibull quantile mapping |
| `wind_speed_hub_ms` | m/s, hub height | 10 m wind x ERA5-measured shear (default 150 m) |
| `site_wind_speed_hub_ms` | m/s, hub height | wind at a chosen farm site, consistent with the system wind (optional) |
| `wind_generation_MW` | MW | capacity-factor ARMA x installed capacity x `wind.scale` |
| `solar_generation_MW` | MW | deterministic seasonal capacity factor x capacity x `solar.scale` |
| `actual_load_MW`, `net_position_MW` | MW | calendar-matched block bootstrap of 2023-2025 history |
| `gas_price_eur_mwh` | EUR/MWh | flat level or multi-regime trajectory |
| `day_ahead_price` | EUR/MWh | 10-seed mixture-density-network ensemble |

All series are hourly on a Europe/Copenhagen local-time calendar.
