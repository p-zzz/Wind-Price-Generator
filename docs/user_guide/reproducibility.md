# Reproducibility and the random seed

A run is fully determined by its configuration: the same config (including
`random_seed`) on the same package versions gives identical output on the same
hardware. Results can differ in the last digits between CPU and GPU.

## One random stream

`generate` creates **one** `numpy.random.Generator` from `random_seed` and consumes it
in a fixed order:

1. the gas trajectory (only in `gas.mode: trajectory`),
2. then, for each path in turn: the wind speed path, the onshore and offshore
   capacity-factor residuals, the load/net-position bootstrap blocks, and the price
   draws.

Solar uses no random numbers; it is the same in every path and every seed.

## Which settings keep the draws identical

This matters when you compare scenarios: if only the knob you changed differs, the
comparison is free of sampling noise ("common random numbers").

| Change | Same random draws? |
|---|---|
| `wind.scale`, `solar.scale` | **Yes.** Wind speed, load and net position are identical; only generation and price respond. |
| `wind.hub_height_m` | **Yes.** It is an output transform only. |
| adding / changing `site` | **Yes** for every other column: the site wind has its own stream, seeded from `(random_seed, 1)`. |
| capacity baselines | **Yes.** |
| `gas.flat.regime` / `custom_eur_mwh` | **Yes** (flat mode uses no draws). |
| `gas.mode` flat <-> trajectory | **No.** Trajectory mode draws first, so every later draw shifts. |
| trajectory settings (segments, ramps, noise) | Draw count stays the same, but the gas path changes. |
| `horizon_hours`, `block_size`, `bootstrap_window_days` | **No.** Every later draw shifts. |
| `start_date` | Draw count stays the same, but every model sees a different calendar, so outputs change. |
| `n_paths` | Path *i* is unaffected by how many paths come after it. |

## Other notes

- All paths share the same gas trajectory.
- All paths start from the same recent observed wind state
  (`data/wind_speed_seed.parquet`); they diverge through the random draws, after a
  200-hour warm-up.
- The price model is an ensemble of 10 networks with fixed training seeds, so
  retraining it with `pipeline/` reproduces it exactly (see {doc}`../retraining`).
