# Quickstart

## Command line

```bash
python -m generator.run --config config/example.yaml
```

This loads the models, simulates the scenario described in `config/example.yaml`
(5 years, 1 path, gas moving from post-crisis into a crisis and back), prints a
validation table and saves `outputs/scenarios_<timestamp>.csv`.

To make your own scenario, copy `config/example.yaml` and edit it. The most common
settings:

```yaml
generator:
  horizon_hours: 43800     # 5 years
  n_paths: 10              # independent paths
  random_seed: 42
  start_date: "2027-01-01" # first hour, local time
wind:
  scale: 1.25              # 1.25x today's installed wind capacity
  hub_height_m: 150        # your turbine's hub height
solar:
  scale: 1.0
gas:
  mode: "flat"
  flat:
    regime: "post_crisis"  # 45 EUR/MWh
```

All keys are described in {doc}`../user_guide/configuration`.

## Python

```python
from generator.config import Config
from generator.io import load_fitted_objects
from generator.run import gas_regimes_for, generate, get_device, print_validation, save_scenarios

cfg = Config.from_yaml("config/example.yaml")
device = get_device()
fitted = load_fitted_objects(cfg.paths.fitted_models_dir, device)

scenarios = generate(cfg, fitted, device)      # long-format DataFrame, see below
print_validation(scenarios, gas_regimes_for(cfg))
save_scenarios(scenarios, cfg.paths.output_dir)
```

`examples/run_example.py` builds the `Config` directly in Python instead of from
YAML.

## What you get

`generate` returns one row per hour **per path**: the timestamps repeat for each
path, and a `path` column (0, 1, ...) tells them apart. Select one path with:

```python
p0 = scenarios[scenarios["path"] == 0]
```

The columns, units and time conventions are listed in
{doc}`../user_guide/outputs`. Two things new users most often get wrong:

- `wind_speed_ms` is **10 m** wind at one offshore point (Horns Rev), the price
  model's system wind. For a farm's power, use `site_wind_speed_hub_ms` with a
  `site:` block (see {doc}`../user_guide/sites`).
- Scale knobs above **~1.25x** push the price model outside its training data; see
  {doc}`../user_guide/scenarios`.
