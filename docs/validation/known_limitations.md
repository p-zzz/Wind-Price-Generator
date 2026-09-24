# Known limitations

**Scope**
- **A scenario generator, not a forecaster.** Paths are realistic draws for one
  fixed state of the world. They don't predict specific dates.
- **Trained on 2015-2025.** Market changes after 2025 are not learned; retrain with
  newer data via {doc}`../retraining`.

**Scale knobs**
- **In-domain only up to ~1.25x** on `wind.scale` / `solar.scale`, alone or
  together. Beyond ~2x, results are outside the validated region.
- **Beyond ~3x the price response saturates** and capture rates stop falling. That's
  an extrapolation artifact. See {doc}`index`.

**Wind**
- **`wind_speed_ms` is 10 m wind at one offshore point** (Horns Rev). Use
  `wind_speed_hub_ms` for turbine power. Both describe DK1 North Sea offshore wind,
  not a specific onshore site.
- **Each path has the climatological wind distribution** over its horizon (quantile
  mapping), so there are no unusually windy or calm years overall.

**Solar**
- **Solar is deterministic:** no cloud-driven variability, identical in every path.

**Load and net position**
- **Bootstrapped from 2023-2025:** a load or net-position regime that never happened
  then cannot appear.
- **Net position is export-positive.** Versions before 2026-09 were unsigned.

**Gas**
- **A schedule, not a market model:** it doesn't react to wind, solar or price.
- **Trajectory-mode noise** weakens the wind-price correlation compared with flat
  gas.

**Price**
- **Runs slightly low:** by 1-7 EUR/MWh per half-year in 2023-25, and by about 18 in
  2025 H2.
- **Not clipped** to market price limits.
- **Ensemble uncertainty** of about +/-0.4 EUR/MWh on a scenario's mean price. Treat
  smaller scenario differences as not meaningful.
