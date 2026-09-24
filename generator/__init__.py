"""Stochastic joint scenario generator for the DK1 (Denmark West) bidding zone.

Draws hourly paths of wind speed, wind and solar generation, load, net position,
gas price and day-ahead electricity price under configurable installed-capacity and
gas-price scenarios. It is a scenario generator, not a forecaster.

Typical use::

    from generator.config import Config
    from generator.io import load_fitted_objects
    from generator.run import generate, get_device

    cfg = Config.from_yaml("config/example.yaml")
    device = get_device()
    scenarios = generate(cfg, load_fitted_objects(cfg.paths.fitted_models_dir, device), device)

or from the command line: ``python -m generator.run --config config/example.yaml``.
"""
