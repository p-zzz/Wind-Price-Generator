"""
Assemble several Price MDN v11 training runs (same data, different PRICE_MDN_SEED) into
the seed-ensemble format the generator loads (generator/io.py -> MDNEnsemble).

Why: single-model results vary a lot with the training seed -- in WinPACT, NPV equity
ranged 430-805 M EUR across five seeds on identical inputs, and picking the best
validation NLL did not pick a representative model. The ensemble pools all members'
mixture components with equal weight, which averages that spread out.

Usage (each --member is a training run's models/price/fitted directory):
    python pipeline/train/assemble_price_ensemble.py \\
        --member ws_seed1/models/price/fitted --member ws_seed2/models/price/fitted ... \\
        --out models/

Writes <out>/price_mdn_v11.pkl (with a "members" list) and one
<out>/price_mdn_v11_seed<k>.pt per member. Refuses members whose norm_stats,
feature_columns or hparams (other than the seed) differ.
"""

import argparse
import pickle
import shutil
from pathlib import Path

PKL = "price_mdn_v11.pkl"
PT = "price_mdn_v11_best.pt"


def _shared_hparams(hp: dict) -> dict:
    return {k: v for k, v in hp.items() if k != "seed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--member", type=Path, action="append", required=True,
                        help="a training run's fitted dir (contains price_mdn_v11.pkl + _best.pt)")
    parser.add_argument("--out", type=Path, required=True, help="usually the repo's models/")
    args = parser.parse_args()

    runs = []
    for d in args.member:
        with open(d / PKL, "rb") as f:
            p = pickle.load(f)
        if "seed" not in p["hparams"]:
            raise SystemExit(f"{d}: no hparams['seed'] -- retrain with the seeded "
                             f"pipeline/train/price_mdn_v11.py so the member is reproducible")
        runs.append((d, p))

    ref = runs[0][1]
    seeds = [p["hparams"]["seed"] for _, p in runs]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"duplicate seeds: {seeds}")
    for d, p in runs[1:]:
        for key, a, b in [("norm_stats", p["norm_stats"], ref["norm_stats"]),
                          ("feature_columns", p["feature_columns"], ref["feature_columns"]),
                          ("hparams", _shared_hparams(p["hparams"]), _shared_hparams(ref["hparams"])),
                          ("training_period", p["training_period"], ref["training_period"])]:
            if a != b:
                raise SystemExit(f"{d}: {key} differs from {runs[0][0]} -- not the same training setup")

    args.out.mkdir(parents=True, exist_ok=True)
    members = []
    for d, p in sorted(runs, key=lambda r: r[1]["hparams"]["seed"]):
        seed = p["hparams"]["seed"]
        name = f"price_mdn_v11_seed{seed}.pt"
        shutil.copy(d / PT, args.out / name)
        members.append({"seed": seed, "val_nll_best": p["val_nll_best"], "best_model_path": name})

    ensemble = {k: v for k, v in ref.items() if k not in ("best_model_path", "val_nll_best")}
    ensemble["hparams"] = {**_shared_hparams(ref["hparams"]), "ensemble_size": len(members)}
    ensemble["members"] = members
    with open(args.out / PKL, "wb") as f:
        pickle.dump(ensemble, f)

    print(f"Wrote {args.out / PKL}: {len(members)} members")
    for m in members:
        print(f"  seed {m['seed']:>3}  val NLL {m['val_nll_best']:.4f}  -> {m['best_model_path']}")


if __name__ == "__main__":
    main()
