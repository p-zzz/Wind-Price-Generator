"""
Minimal end-to-end smoke test: builds a Config directly (bypassing YAML, so this
doubles as a runnable API example) with a short 1-year horizon, runs the full
generator stack, prints the validation table, and saves a CSV.

Run from the repo root: python examples/run_example.py
"""

from pathlib import Path

from generator.config import Config, GasConfig, GasFlatConfig, GasTrajectoryConfig, PathsConfig, SolarConfig, WindConfig
from generator.io import load_fitted_objects
from generator.run import generate, get_device, print_validation, save_scenarios

REPO_ROOT = Path(__file__).resolve().parent.parent

cfg = Config(
    horizon_hours=8760,  # 1 year, for a quick smoke test
    n_paths=1,
    random_seed=42,
    block_size=168,
    start_date="2024-01-01",
    wind=WindConfig(scale=1.0, onshore_capacity_mw=4162.6, offshore_capacity_mw=2661.7),
    solar=SolarConfig(scale=1.0, capacity_mw=3881.1),
    gas=GasConfig(
        mode="flat",
        flat=GasFlatConfig(regime="post_crisis"),
        trajectory=GasTrajectoryConfig(),
    ),
    paths=PathsConfig(
        fitted_models_dir=REPO_ROOT / "models",
        data_dir=REPO_ROOT / "data",
        output_dir=REPO_ROOT / "outputs",
    ),
)

device = get_device()
fitted = load_fitted_objects(cfg.paths.fitted_models_dir, device)
df_scenarios = generate(cfg, fitted, device)
print_validation(df_scenarios)
save_scenarios(df_scenarios, cfg.paths.output_dir)
