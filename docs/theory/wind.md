# Wind

Wind has three parts: the wind speed itself, the fleet's generation, and a
hub-height version of the wind speed for turbine calculations.

## Wind speed: Transformer mixture model

`wind_speed_ms` is hourly **10 m** wind speed at the ERA5 grid point nearest Horns
Rev (55.55 N, 7.91 E), in the Danish North Sea. It is simulated autoregressively,
one hour at a time:

- **Inputs:** the last 56 hours of (normalised) wind speed, plus calendar features
  ($\sin$/$\cos$ of hour of day and month).
- **Network:** a small causal Transformer encoder (model width 32, 2 attention heads,
  1 layer) followed by an MLP.
- **Output:** a mixture of $K = 8$ Gaussians for the next hour's wind speed. The
  generator samples a component, then a value, and floors the result at 0 m/s.

It was trained on 2014-12 to 2025-12 ERA5 data, with hyperparameters tuned with
Optuna. Each path starts from the most recent observed hours
(`data/wind_speed_seed.parquet`) and runs a 200-hour warm-up just before the start
date, so it begins in the right season.

### Quantile mapping

Free-running autoregressive models drift towards a slightly wrong distribution. After
simulation, each path is **quantile-mapped** onto the fitted climatological Weibull
distribution, separately for each season x 6-hour block of the day. The ranks
(timing, persistence) come from the Transformer; the values come from climatology.

Because the mapping acts over the whole path, every path has the same long-run
distribution. A run can't produce an unusually windy or calm period overall; the
paths differ in *when* the wind blows.

## Wind generation: capacity-factor model

`wind_generation_MW` is the DK1 fleet's output, simulated separately for onshore and
offshore:

1. **Mean:** an OLS model of capacity factor on the simulated wind speed, month and
   hour of day. This ties generation to the same wind path the price model sees.
2. **Residual:** an ARMA process on the standardised residual (ARMA(2,1) onshore,
   ARMA(2,2) offshore), rescaled by month.
3. **Quantile mapping:** onto the empirical capacity-factor distribution per
   season x 6-hour block. Empirical, not parametric, because offshore capacity
   factor is bimodal: a calm mode, and a plateau at rated power.

$$\text{generation} = \text{CF} \times \text{installed capacity} \times \text{wind.scale}$$

## Hub-height wind

Every model runs on 10 m wind, but turbines sit at 100-170 m. `wind_speed_hub_ms`
extrapolates with a power law:

$$v_{\text{hub}} = v_{10} \left(\frac{h}{10\ \text{m}}\right)^{\alpha}$$

The shear exponent $\alpha$ is **measured**, not assumed. It comes from ERA5's own
10 m and 100 m wind at Horns Rev (1990-2026) and is stored per calendar month and
10 m wind-speed band in `data/horns_rev_shear.csv` (built by
`scripts/build_shear_table.py`). In each cell,
$\alpha = \log(\bar v_{100} / \bar v_{10}) / \log 10$, which is unbiased on mean
100 m wind speed.

| | Value |
|---|---|
| Average $\alpha$ | about 0.10 |
| Range by cell | 0.05 (light winds, autumn) to 0.12 (spring) |
| Error on hourly 100 m wind | 0.62 m/s (a single constant would give 0.72) |

Offshore shear is low because the sea surface is smooth. It is lowest in summer and
at light winds, and barely depends on time of day. A generic onshore exponent such as
0.2 would overstate hub-height wind by about a third (13.4 vs 10.0 m/s at 170 m).

```{warning}
Both wind columns describe **one offshore point**. They are representative of DK1
offshore wind in the North Sea, not of a specific onshore site. Wake losses,
availability and a farm's power curve are for downstream tools.
```

## Wind at a farm site

`site_wind_speed_hub_ms` gives the wind at a chosen farm site
(`config/sites.yaml`), generated **from the simulated system wind**, so the farm's
output stays correlated with the prices as in real weather. The model is a Gaussian
copula, fitted on every paired ERA5 hour since 1990, with distributions per calendar
month x local hour:

1. The system (Horns Rev) 10 m wind $x_t$ becomes a normal score
   $u_t = \Phi^{-1}(F_x(x_t))$, using its climatological distribution $F_x$ for that
   month and hour.
2. The site's score is a regression on the system scores over a window of hours,
   plus a persistent residual:

   $$w_t = \sum_{k=-3}^{6} \beta_{m,k}\, u_{t-k} + e_t, \qquad
   e_t = \phi\, e_{t-1} + \sigma_m\sqrt{1-\phi^2}\,\eta_t$$

   The lags let weather reach the site **later** than Horns Rev: inland sites lag by
   1-3 hours as fronts move east. Coefficients are fitted per month; $\phi \approx 0.9$
   is the persistence of the site's own local deviations.
3. $w_t$ is mapped back through the site's own 100 m wind distribution,
   $v_{100} = F_y^{-1}(\Phi(w_t))$, then taken to hub height with the site's
   ERA5-measured shear.

The site therefore keeps its own climate (distribution, seasonal and daily cycles)
and moves with the system wind as much as, and with the delay that, the real weather
does. Validation and the
workflow for new sites: {doc}`../user_guide/sites`.

API: `generator.wind.run_transformer_simulation`,
`generator.wind.simulate_wind_generation`, `generator.wind.hub_height_wind_speed`,
`generator.wind.site_wind_speed`.
