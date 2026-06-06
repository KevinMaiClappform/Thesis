"""
extract_outputs.py — write per-model CSVs and plots from the backtest caches.

After `BACKTEST.py` and/or `COVAR_BACKTEST.py` have populated their cache
pickles, this script writes `output/<model>_forecasts.csv`,
`output/<model>_summary.csv`, and `output/<model>_grid.png` for every cached
model -- the same files that the individual `python QRF.py` etc. invocations
would have produced, but using the *already computed* cached results so there
is no duplicate compute.

Run from `repo/`:
    python extract_outputs.py [--no-univariate] [--no-multivariate]

Expected runtime: a few seconds. The cache must exist; run the backtest
scripts first.
"""

import sys
import pickle
from pathlib import Path

UNIV_CACHE  = Path("backtest_cache.pkl")
MULTV_CACHE = Path("covar_backtest_cache.pkl")


def extract_univariate(cache_path):
    if not cache_path.exists():
        print(f"[univariate] no cache at {cache_path} -- run BACKTEST.py first.")
        return

    from output import report
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    for model_name, results in cache.items():
        if model_name.startswith("_"):
            continue
        print(f"\n=== {model_name} ===")
        report(results, model_name=model_name)


def extract_multivariate(cache_path):
    if not cache_path.exists():
        print(f"[multivariate] no cache at {cache_path} -- run COVAR_BACKTEST.py first.")
        return

    from output_covar import report
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    extras = cache.get("_extras", {})
    best_specs = extras.get("_CoCAViaR_best_specs")

    for model_name in ["CoCAViaR", "DCCGARCH", "COVAR_QRF", "COVAR_QGB"]:
        if model_name not in cache:
            continue
        print(f"\n=== {model_name} ===")
        # All multivariate models plot ONLY the CoVaR line (supervisor
        # feedback). The multivariate target is the systemic CoVaR; the VaR is
        # merely the conditioning device in the two-step lexicographic
        # (VaR, CoVaR) estimator and is benchmarked separately by the
        # univariate GARCH-t. (This contrasts with the univariate (VaR, ES)
        # plots, which show both because that pair IS jointly forecast.)
        if model_name == "CoCAViaR":
            # CoCAViaR keeps the per-pair lex-best spec label in the title.
            kwargs = {"best_specs": best_specs, "show_var": False}
        else:
            kwargs = {"show_var": False}
        report(cache[model_name], model_name=model_name, **kwargs)


def main():
    do_uni   = "--no-univariate"   not in sys.argv
    do_multi = "--no-multivariate" not in sys.argv

    if do_uni:
        extract_univariate(UNIV_CACHE)
    if do_multi:
        extract_multivariate(MULTV_CACHE)

    print("\nDone. Per-model CSVs and PNGs are in output/.")


if __name__ == "__main__":
    main()
