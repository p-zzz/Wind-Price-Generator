# Solar

`solar_generation_MW` is the DK1 solar fleet's output:

$$\text{generation} = \text{CF}(\text{month}, \text{hour}) \times \text{capacity\_mw} \times \text{solar.scale}$$

The capacity factor comes from an OLS model of observed capacity factor on hour of
day and month (2015-2025). Night hours are exactly zero: a fixed set of 144
(month, hour) pairs counts as daytime.

```{warning}
Solar is **deterministic**: the seasonal-mean shape only, identical in every path and
every seed. Real cloud-driven day-to-day variability exists in the data but is not
simulated. Solar-driven price variability is therefore understated, and so is the
spread of solar capture rates across paths.
```

The price model sees solar through `solar_load_ratio` (solar MW divided by the
simulated load), so `solar.scale` moves prices; see {doc}`../user_guide/scenarios`
for how far.

API: `generator.solar.simulate_solar_cf`.
