# Gas price

Gas (TTF, EUR/MWh) is an input to the price model, set by a **schedule**, not a
market model: it doesn't respond to anything else in the simulation.

## Regime levels

| Regime | Level (EUR/MWh) | Period it represents |
|---|---|---|
| `pre_crisis` | 35 | before the 2021-22 energy crisis |
| `crisis` | 150 | 2022 |
| `post_crisis` | 45 | 2023 onward |

## Flat mode

One constant level (a regime or a custom value) for the whole horizon. No random
draws.

## Trajectory mode

Regimes follow each other (`segments`, in years of 8760 h). Each change is a linear
ramp over `ramp_hours`. AR(1) noise is added on top:

$$g_t = \ell_t + \varepsilon_t, \qquad
\varepsilon_t = \phi\,\varepsilon_{t-1} + \sigma\sqrt{1-\phi^2}\,\eta_t,\quad \eta_t \sim N(0,1)$$

Here $\ell_t$ is the scheduled level, $\phi$ is `noise_ar`, $\sigma$ is `noise_std`
(the noise's stationary standard deviation) and $\varepsilon_0 = 0$. The result is
floored at `noise_floor`. The trajectory is drawn once and shared by all paths.

## Choosing a mode

Gas noise weakens the modelled wind-price correlation compared with a flat level,
most of all in the crisis regime. Use **flat** when the wind-price relationship is
what you need, and **trajectory** when you need gas-price levels that change over
time. When validating, compare the wind-price correlation within a regime; the
validation printout reports it per regime.

API: `generator.gas.build_gas_trajectory`, `generator.gas.flat_gas_price`.
