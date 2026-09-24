# Price model

`day_ahead_price` is drawn hour by hour from a **mixture density network** (MDN):
a neural network that outputs a full probability distribution of the price given
the hour's inputs, not a single forecast.

## Inputs and output

Eleven inputs per hour:

- calendar: $\sin$/$\cos$ of hour of day and month,
- 10 m wind speed (the merit-order signal),
- `solar_load_ratio` and `wind_load_ratio` (scaled generation / load; this is how
  the capacity knobs reach the price),
- load, net position, gas price,
- a wind x solar interaction term.

The continuous inputs are standardised with the training data's mean and standard
deviation. The output is a mixture of $K = 6$ Gaussians:

$$p(\text{price} \mid x) = \sum_{k=1}^{6} \pi_k(x)\, \mathcal N\!\left(\mu_k(x), \sigma_k(x)^2\right)$$

The generator samples one component, then one value, per hour. Hours are sampled
independently given their inputs; the price's persistence over time comes from its
persistent inputs (wind, load, gas).

## Training

- **Data:** DK1 day-ahead prices, 2015-2025 (ENTSO-E), with ERA5 wind, TTF gas and
  Energinet generation and capacity data.
- **Network:** layers of [128, 128, 64] with ReLU; weight decay $10^{-2}$; Adam with
  learning rate $10^{-3}$; early stopping on validation negative log-likelihood (NLL).
- **Post-crisis emphasis:** hours from 2023 on are oversampled 3.5x, so the model
  focuses on today's market.
- **Validation split:** the last 20% of each regime (pre-crisis, crisis,
  post-crisis), chronologically.

## Seed ensemble

Networks trained on identical data but with different random seeds agree closely on
historical inputs, yet **diverge on generated scenarios**. Scenarios combine inputs
the 2015-2025 data barely covers, such as flat gas with 2026 installed capacity. With
the original settings, single models spread by about 4.8 EUR/MWh in a scenario's mean
price, and picking the "best" seed by validation score did not pick a representative
one.

The shipped model is therefore an **equal-weight ensemble of 10 seeds**
(`generator.price.MDNEnsemble`). Its mixture has $10 \times 6 = 60$ components, each
weight divided by 10, which is exactly "pick a member at random, then sample from
it".

## Hyperparameter choice

The hidden layers and weight decay were chosen by a sweep of 33 configurations x 3-5
seeds. Each configuration was scored on two things: validation NLL, and **how much
its seeds disagree on scenario inputs**. The winner, compared with the original
[64, 64, 32] and weight decay $10^{-4}$:

| | Original | Shipped |
|---|---|---|
| Validation NLL | 4.80 | 4.73 |
| Seed disagreement on scenario hours | 9.4 EUR/MWh | 4.3 EUR/MWh |
| Seed spread of a scenario's mean price | 4.8 EUR/MWh | 1.3 EUR/MWh |

With 10 members, the ensemble's own uncertainty on a scenario's mean price is about
+/-0.4 EUR/MWh. Stronger weight decay collapses the unscaled price target towards
zero, and a standardised target, $K = 3$ or no oversampling all scored worse.

## Limits

- **Extrapolation:** inputs are never clipped. Beyond ~1.25x on the scale knobs, a
  growing share of hours falls outside the training data, and the price response
  saturates beyond ~3x; see {doc}`../validation/index`.
- **Unclipped prices:** sampled prices can fall outside the market's price limits.
- **Price level runs slightly low:** on the historical inputs, the ensemble's
  expected price is below the observed mean by 1-7 EUR/MWh per half-year from
  2023 H2 to 2025 H1. In 2025 H2, DK1 prices stayed around 80 EUR/MWh while gas fell
  to about 31, and the model (with that period in its validation set) is about
  18 EUR/MWh low there. Treat absolute price levels as a few EUR/MWh conservative.
  Differences between scenarios are more reliable than levels.

API: `generator.price.simulate_price_mdn`, `generator.price.MDNEnsemble`.
