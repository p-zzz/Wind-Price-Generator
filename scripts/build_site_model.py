"""
Fit the wind model for a farm site listed in config/sites.yaml and write
data/sites/<site>.json, which the generator uses for `site: {name: <site>}`.

What it learns (from every paired ERA5 hour, e.g. 1990-2026): per calendar month x
local hour, the system site's (Horns Rev) 10 m wind distribution and the farm site's
100 m wind distribution; per month, a regression of the site's normal score on the
system's scores from 6 hours before to 3 hours after (weather reaches inland sites
hours after Horns Rev); the persistence (AR(1)) of the site's own deviations; and the
site's wind shear per month x 100 m speed band. See generator.wind.site_wind_speed.

Then it validates: the model is driven by the *real* Horns Rev history and the result
compared with the site's *real* ERA5 wind (distribution, persistence, co-movement,
capacity factor), next to the naive alternative of using Horns Rev wind directly.

Run from the repo root after extracting the site's ERA5 wind
(pipeline/build/era5_sites_builder.py):
    python scripts/build_site_model.py --site thor [--era5 DATA/processed/era5_sites_wind_hourly.parquet]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import ndtri

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from generator.wind import _hour_block_cells, _lagged, site_wind_speed  # noqa: E402

N_PROBS = 200
PER_DAY = 24                                   # distribution cells: month x local hour
LAGS = list(range(-3, 7))                      # site hour t regressed on system hours t-6 .. t+3
SHEAR_EDGES = [0, 3, 5, 7, 9, 11, 14, 99]     # 100 m wind-speed bands, m/s
VALIDATION_HUB_M = 150.0


def normal_scores(x: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Rank-based normal scores within each cell."""
    z = np.empty(len(x))
    for c in np.unique(cells):
        m = cells == c
        r = pd.Series(x[m]).rank(method="average").values
        z[m] = ndtri((r - 0.5) / m.sum())
    return z


def fit(system10: pd.Series, site10: pd.Series, site100: pd.Series) -> dict:
    idx = system10.index
    cells = _hour_block_cells(idx, PER_DAY)
    months = np.asarray(idx.month) - 1
    probs = (np.arange(N_PROBS) + 0.5) / N_PROBS

    q_x = np.zeros((12, PER_DAY, N_PROBS)); q_y = np.zeros((12, PER_DAY, N_PROBS))
    for c in range(12 * PER_DAY):
        m, b = divmod(c, PER_DAY)
        sel = cells == c
        q_x[m, b] = np.quantile(system10.values[sel], probs)
        q_y[m, b] = np.quantile(site100.values[sel], probs)

    u = normal_scores(system10.values, cells)
    w = normal_scores(site100.values, cells)
    rho = np.array([np.corrcoef(u[months == m], w[months == m])[0, 1] for m in range(12)])
    U = _lagged(u, LAGS)
    beta = np.array([np.linalg.lstsq(U[months == m], w[months == m], rcond=None)[0] for m in range(12)])
    e = w - (U * beta[months]).sum(axis=1)
    e_std = np.array([e[months == m].std() for m in range(12)])
    # persistence of the site's own deviations, from consecutive hours only
    consecutive = np.diff(idx.asi8) == 3_600_000_000_000
    phi = float(np.corrcoef(e[:-1][consecutive], e[1:][consecutive])[0, 1])

    band = np.clip(np.searchsorted(SHEAR_EDGES, site100.values, side="right") - 1, 0, len(SHEAR_EDGES) - 2)
    alpha = np.full((12, len(SHEAR_EDGES) - 1), np.nan)
    for m in range(12):
        for k in range(len(SHEAR_EDGES) - 1):
            sel = (months == m) & (band == k)
            if sel.sum() >= 50:
                alpha[m, k] = np.log(site100.values[sel].mean() / site10.values[sel].mean()) / np.log(10)
        # sparse bands (e.g. calm or storm hours in some months): use the month's overall value
        overall = np.log(site100.values[months == m].mean() / site10.values[months == m].mean()) / np.log(10)
        alpha[m] = np.where(np.isnan(alpha[m]), overall, alpha[m])

    return {"probs": probs.tolist(), "q_system_10m": q_x.round(3).tolist(), "q_site_100m": q_y.round(3).tolist(),
            "rho": rho.round(4).tolist(), "lags": LAGS, "beta": beta.round(5).tolist(),
            "e_std": e_std.round(4).tolist(), "phi": round(phi, 4),
            "shear_v100_edges": SHEAR_EDGES, "shear_alpha": alpha.round(4).tolist()}


def gross_cf(v: np.ndarray) -> float:
    """Generic turbine: cubic 3-11 m/s, rated 11-25 m/s, cut-out 25 m/s."""
    p = np.clip((v**3 - 27) / (11**3 - 27), 0, 1)
    p[(v < 3) | (v >= 25)] = 0
    return float(p.mean())


def acf(x: np.ndarray, lag: int) -> float:
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def validate(model: dict, system10: pd.Series, site10: pd.Series, site100: pd.Series) -> pd.DataFrame:
    idx = system10.index
    months = np.asarray(idx.month) - 1
    edges = np.asarray(SHEAR_EDGES, float)
    band = np.clip(np.searchsorted(edges, site100.values, side="right") - 1, 0, len(edges) - 2)
    true_hub = site100.values * (VALIDATION_HUB_M / 100) ** np.asarray(model["shear_alpha"])[months, band]
    sim_hub = site_wind_speed(system10.values, idx, model, VALIDATION_HUB_M, np.random.default_rng(0))
    sys_hub = system10.values * (VALIDATION_HUB_M / 10) ** 0.096       # naive: Horns Rev wind as site wind
    rows = {}
    for name, v in [("site (ERA5 truth)", true_hub), ("site model", sim_hub), ("Horns Rev used as site", sys_hub)]:
        pw = np.clip((v**3 - 27) / (11**3 - 27), 0, 1) * ((v >= 3) & (v < 25))
        sys_pw = np.clip((system10.values * 15**0.096) ** 3 / 11**3, 0, 1)
        rows[name] = {"mean m/s": v.mean(), "std": v.std(), "p10": np.percentile(v, 10), "p90": np.percentile(v, 90),
                      "acf 1h": acf(v, 1), "acf 24h": acf(v, 24),
                      "corr w/ Horns Rev": np.corrcoef(v, system10.values)[0, 1],
                      "gross CF": gross_cf(v), "power corr w/ HR power": np.corrcoef(pw, sys_pw)[0, 1]}
    return pd.DataFrame(rows).T


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--site", required=True)
    ap.add_argument("--era5", type=Path, default=REPO / "DATA/processed/era5_sites_wind_hourly.parquet")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/sites")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO / "config/sites.yaml").read_text())
    system, site = cfg["system_site"], args.site
    if site not in cfg["sites"]:
        raise SystemExit(f"'{site}' is not in config/sites.yaml")
    era5 = pd.read_parquet(args.era5).tz_convert("Europe/Copenhagen")
    cols = [f"{system}_wind_speed_10m_ms", f"{site}_wind_speed_10m_ms", f"{site}_wind_speed_100m_ms"]
    missing = [c for c in cols if c not in era5.columns]
    if missing:
        raise SystemExit(f"{args.era5} lacks {missing} -- run pipeline/build/era5_sites_builder.py first")
    d = era5[cols].dropna()
    system10, site10, site100 = (d[c] for c in cols)

    model = fit(system10, site10, site100)
    model.update(site=site, system_site=system, lat=cfg["sites"][site]["lat"], lon=cfg["sites"][site]["lon"],
                 description=cfg["sites"][site].get("description", ""),
                 fit_period=f"{d.index.min().date()} .. {d.index.max().date()}", n_hours=len(d))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{site}.json"
    out.write_text(json.dumps(model, separators=(",", ":")))
    print(f"Fitted {site} on {len(d):,} hours ({model['fit_period']}); rho by month "
          f"{min(model['rho']):.2f}-{max(model['rho']):.2f}, phi {model['phi']:.3f} -> {out}")

    print(f"\nValidation at {VALIDATION_HUB_M:.0f} m hub, driven by the REAL Horns Rev history:")
    pd.set_option("display.width", 200)
    print(validate(model, system10, site10, site100).round(3).to_string())


if __name__ == "__main__":
    main()
