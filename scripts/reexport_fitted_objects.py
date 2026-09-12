"""
One-off re-export of fitted objects from the thesis repo (Code/) into this repo's
models/ directory, slimmed down for distribution. Run once, from within this repo,
with the thesis repo's venv active (needs torch + statsmodels).

Not run by end users of the generator -- this only exists so the provenance of
models/*.pkl in this repo is reproducible and auditable. Never modifies anything
in Code/.

Two things happen here:

1. wind_cf_{onshore,offshore}_arma.pkl are re-exported at ~1.3GB each in the thesis
   repo purely because statsmodels' ARIMAResults.fit() retains the full Kalman-filter
   state/covariance history, which .simulate() with the default anchor (unconditional
   simulation from the model's stationary initial state -- the call in
   joint_generator.py never passes anchor=) does not use at all. Reconstructing a
   fresh ARIMA model on a short dummy series and calling .filter(params) gives a
   results object with the same order/seasonal_order/params but none of that stored
   history. This is verified below, not assumed: both the original and reconstructed
   objects are used to simulate the same path (same random_state) and compared.

2. price_mdn_v11.pkl / wind_transformer_tuned_r3.pkl / solar_model_capacity_factor.pkl
   are already small (KB-MB scale) and are just copied through unchanged.

load_model.pkl / net_position_model.pkl are deliberately NOT re-exported -- see
../.claude/generator_extraction_plan.md in the thesis repo for why (dead loads,
never used by joint_generator.py's generate()).
"""

import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

CODE_ROOT = Path(__file__).resolve().parents[2] / "Code"
THIS_MODELS = Path(__file__).resolve().parents[1] / "models"
THIS_MODELS.mkdir(parents=True, exist_ok=True)

# WindCFQuantileMapper was pickled with __module__ == "__main__" (the fitting
# script runs directly, not imported) -- register it under __main__ here so
# unpickling resolves regardless of how this script is invoked.
sys.path.insert(0, str(CODE_ROOT / "models" / "wind"))
from wind_capacity_factor_arma import WindCFQuantileMapper  # noqa: E402

sys.modules["__main__"].WindCFQuantileMapper = WindCFQuantileMapper

VERIFY_N_SIM = 2000
VERIFY_SEED = 12345


# ------ Slim one ARMA results object ------


def slim_arma_pickle(src_path: Path, dst_path: Path, dummy_len: int = 300) -> None:
    print(f"\n{'=' * 60}\n{src_path.name}\n{'=' * 60}")
    print(f"  original size: {src_path.stat().st_size / 1e6:.1f} MB")

    with open(src_path, "rb") as f:
        fitted = pickle.load(f)

    arma_model = fitted["arma_model"]
    order = arma_model.model.order
    seasonal_order = arma_model.model.seasonal_order
    params = arma_model.params
    print(f"  order={order}  seasonal_order={seasonal_order}  n_params={len(params)}")

    rng = np.random.default_rng(0)
    dummy = rng.normal(size=dummy_len)
    slim_model = ARIMA(dummy, order=order, seasonal_order=seasonal_order)
    slim_results = slim_model.filter(params)

    # ------ Verify before trusting: identical simulate() output, fixed seed ------
    orig_sim = arma_model.simulate(
        nsimulations=VERIFY_N_SIM, random_state=np.random.default_rng(VERIFY_SEED)
    )
    slim_sim = slim_results.simulate(
        nsimulations=VERIFY_N_SIM, random_state=np.random.default_rng(VERIFY_SEED)
    )
    max_abs_diff = float(np.max(np.abs(np.asarray(orig_sim) - np.asarray(slim_sim))))
    print(f"  verify: max|orig_sim - slim_sim| over {VERIFY_N_SIM} steps = {max_abs_diff:.3e}")
    if max_abs_diff > 1e-8:
        raise RuntimeError(
            f"{src_path.name}: slimmed ARMA object diverges from original "
            f"(max abs diff {max_abs_diff:.3e}) -- do not use, fall back to "
            f"remove_data() approach instead."
        )

    fitted_slim = dict(fitted)
    fitted_slim["arma_model"] = slim_results
    with open(dst_path, "wb") as f:
        pickle.dump(fitted_slim, f)

    print(f"  slimmed size:  {dst_path.stat().st_size / 1e6:.2f} MB")
    print(f"  saved: {dst_path}")


# ------ Main ------


def main() -> None:
    slim_arma_pickle(
        CODE_ROOT / "models/wind/fitted/wind_cf_onshore_arma.pkl",
        THIS_MODELS / "wind_cf_onshore_arma.pkl",
    )
    slim_arma_pickle(
        CODE_ROOT / "models/wind/fitted/wind_cf_offshore_arma.pkl",
        THIS_MODELS / "wind_cf_offshore_arma.pkl",
    )

    print(f"\n{'=' * 60}\nCopying small fitted objects unchanged\n{'=' * 60}")
    small_files = [
        ("models/wind/fitted/wind_transformer_tuned_r3.pkl", "wind_transformer_r3.pkl"),
        # .pt filename must match the basename stored in the pkl's "best_model_path"
        # field exactly -- generator/io.py falls back to models_dir/<that basename>
        # when the pkl's original absolute path doesn't exist on this machine.
        ("models/wind/fitted/wind_transformer_tuned_r3_best.pt", "wind_transformer_tuned_r3_best.pt"),
        ("models/exog/fitted/solar_model_capacity_factor.pkl", "solar_cf.pkl"),
        ("models/price/fitted/price_mdn_v11.pkl", "price_mdn_v11.pkl"),
        ("models/price/fitted/price_mdn_v11_best.pt", "price_mdn_v11_best.pt"),
    ]
    for src_rel, dst_name in small_files:
        src = CODE_ROOT / src_rel
        dst = THIS_MODELS / dst_name
        shutil.copy2(src, dst)
        print(f"  {src_rel} -> models/{dst_name}  ({dst.stat().st_size / 1e3:.1f} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
