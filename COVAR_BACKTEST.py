"""
COVAR_BACKTEST.py -- multivariate cross-model comparison for (VaR, CoVaR).

This is the bivariate analogue of BACKTEST.py. It evaluates and compares
the CoCAViaR specifications of Dimitriadis & Hoga (2026, six specs in
COCAVIAR.py) against the DCC-GARCH benchmark (DCCGARCH.py) on the same set
of (reference, system) pairs.

Following Dimitriadis & Hoga (2026, eqs. 5 and following), the
(VaR, CoVaR) pair is *not* jointly elicitable as a scalar score. We
therefore compare models on the two-component lexicographic score:

    S_VaR(v, x; beta)              = (1{x <= v} - beta) * (v - x)
    S_CoVaR((v, c), (x, y); alpha) = 1{x > v} * (1{y <= c} - alpha) * (c - y)

Per pair we report:

  * VaR distress rate (target = 1 - beta = 5%)
  * Conditional CoVaR violation rate (target = 1 - alpha = 5%)
  * Average S_VaR and S_CoVaR
  * Diebold-Mariano tests on the per-observation S_VaR and S_CoVaR
    differentials between every pair of models (Newey-West HAC).

Cache convention
----------------
Results are cached to ``covar_backtest_cache.pkl`` keyed by
``(model, ref_stock) -> DataFrame`` so a second invocation reuses the
forecasts without rerunning the (slow) CoCAViaR specs or the DCC-GARCH
Monte Carlo.

References
----------
- Dimitriadis & Hoga (2026), J. Bus. Econ. Stat. (Sec. 5.2 for the
  benchmark setup and pairwise comparison protocol).
- Diebold & Mariano (1995), J. Bus. Econ. Stat. (test).
- Newey & West (1987), Econometrica (HAC variance).
- Fissler & Hoga (2024), Annals of Statistics (multi-objective
  elicitability of (VaR, CoVaR)).
"""

import os
import pickle
from itertools import combinations

import numpy as np
import pandas as pd

from losses import s_var, s_covar
from BACKTEST import (newey_west_lrv, diebold_mariano,
                     kupiec_uc, christoffersen_cc)


CACHE_PATH = "covar_backtest_cache.pkl"

# Ordered fast-and-new first, slow-and-tested last:
#   1. COVAR_QGB  -- new ML code, fastest (LightGBM): fail-fast smoke test.
#   2. COVAR_QRF  -- new ML code, moderate runtime.
#   3. DCCGARCH   -- well-tested, moderate runtime.
#   4. COVAR      -- well-tested, slowest (~3 h on the bottleneck).
# Combined with the per-model checkpointing in
# load_or_compute_covar_results, this preserves the most work on a crash.
BASELINE_MULTV_MODELS = ["COVAR_QGB", "COVAR_QRF", "DCCGARCH", "CoCAViaR"]


# =========================================================================
# 1. Load (or compute) per-model CoVaR forecasts
# =========================================================================


def _ensure_scores(df, beta=0.95, alpha=0.95):
    """Add S_VaR and S_CoVaR columns if absent. COVAR's run_all_pairs
    output has them missing; DCCGARCH's output has them already."""
    out = df.copy()
    if "S_VaR" not in out.columns:
        out["S_VaR"] = s_var(out["VaR_X"].values, out["X_loss"].values, beta=beta)
    if "S_CoVaR" not in out.columns:
        out["S_CoVaR"] = s_covar(out["VaR_X"].values, out["CoVaR_Y"].values,
                                  out["X_loss"].values, out["Y_loss"].values,
                                  alpha=alpha)
    return out


def _run_one_multivariate_model(m):
    """Dispatch a model short-name to its run_all_pairs() output.

    Returns a (results, extras) tuple, where ``results`` is the
    {ref_name: forecast_df} mapping and ``extras`` is any auxiliary
    metadata to store under the consolidated cache's _extras key (e.g.
    CoCAViaR's lex-best-spec assignments).
    """
    if m == "CoCAViaR":
        from COCAVIAR import run_all_pairs as run_cocaviar
        all_specs, best_specs, results = run_cocaviar(verbose=False)
        return results, {"_CoCAViaR_best_specs": best_specs,
                         "_CoCAViaR_all_specs":  all_specs}
    if m == "DCCGARCH":
        from DCCGARCH import run_all_pairs as run_dcc
        return run_dcc(verbose=False), {}
    if m == "COVAR_QRF":
        from COVAR_ML import run_all_pairs as run_ml
        return run_ml(model_type="QRF", verbose=False), {}
    if m == "COVAR_QGB":
        from COVAR_ML import run_all_pairs as run_ml
        return run_ml(model_type="QGB", verbose=False), {}
    raise ValueError(m)


def load_or_compute_covar_results(cache_path=CACHE_PATH, models=None,
                                   force=False, beta=0.95, alpha=0.95):
    """Run the multivariate models on all (reference, system) pairs.

    `models` is a list of model short-names; defaults to
    BASELINE_MULTV_MODELS. Returns a nested dict
    {model: {ref_name: forecast_df}} plus a "_extras" key holding any
    auxiliary metadata (currently COVAR's lex-best spec per pair).

    Per-model checkpointing: each successfully computed model is written
    to ``<cache_path>.<MODEL>.partial`` immediately after it finishes,
    so a crash on a later model does not invalidate hours of work on
    earlier ones. A re-run picks up where the previous run left off.
    Mirrors the same protocol used by BACKTEST.load_or_compute_results.
    """
    if (not force) and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if models is None:
        models = BASELINE_MULTV_MODELS

    out    = {}
    extras = {}

    for m in models:
        partial = f"{cache_path}.{m}.partial"
        if (not force) and os.path.exists(partial):
            print(f"  loading checkpoint {partial}")
            with open(partial, "rb") as f:
                payload = pickle.load(f)
            out[m] = payload["results"]
            extras.update(payload.get("extras", {}))
            continue

        print(f"--- Running {m} ---")
        results, m_extras = _run_one_multivariate_model(m)

        # Standardise: ensure S_VaR / S_CoVaR columns are present before
        # checkpointing, so a re-load can skip recomputation.
        out[m] = {ref: _ensure_scores(df, beta=beta, alpha=alpha)
                  for ref, df in results.items()}
        extras.update(m_extras)

        with open(partial, "wb") as f:
            pickle.dump({"results": out[m], "extras": m_extras}, f)
        print(f"  checkpoint saved -> {partial}")

    out["_extras"] = extras

    # All models succeeded -> assemble final cache and clean up checkpoints.
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    for m in models:
        partial = f"{cache_path}.{m}.partial"
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass
    return out


# =========================================================================
# 2. Single-model coverage diagnostics for (VaR, CoVaR)
# =========================================================================


def covar_diagnostics_one(df, beta=0.95, alpha=0.95):
    """Per-pair (model, ref) calibration diagnostics plus formal coverage
    tests, the multivariate analogue of the univariate Kupiec / Christoffersen
    battery.

    Two violation sequences are tested:

      * VaR distress sequence  D_t = 1{X_t > VaR_t}, a daily 0/1 series whose
        nominal rate is (1 - beta). Tested with the Kupiec (1995) unconditional
        coverage LR and the Christoffersen (1998) conditional coverage LR
        (which also detects clustering of distress days).

      * CoVaR conditional-violation sequence H_t = 1{Y_t > CoVaR_t} restricted
        to the distress days {t : D_t = 1}, whose nominal rate is (1 - alpha).
        Tested with the Kupiec UC LR on that sub-sample. An independence test
        is not applied because the distress days are non-consecutive, so the
        first-order-Markov structure is not well defined.

    The Diebold-Mariano comparison on (S_VaR, S_CoVaR) remains the primary,
    Dimitriadis-Hoga (2026)-consistent model-ranking tool; these coverage
    tests are the calibration complement, mirroring the univariate analysis.
    """
    distress = (df["X_loss"] > df["VaR_X"]).values
    n_d = max(1, int(distress.sum()))

    # CoVaR conditional violations on distress days.
    covar_hit = (df["Y_loss"].values > df["CoVaR_Y"].values)
    cov_v = (covar_hit & distress).sum() / n_d
    covar_hit_distress = covar_hit[distress].astype(int)

    # Formal coverage tests.
    _, p_uc_var, _ = kupiec_uc(distress.astype(int), 1.0 - beta)
    cc_var = christoffersen_cc(distress.astype(int), 1.0 - beta)
    _, p_uc_covar, _ = kupiec_uc(covar_hit_distress, 1.0 - alpha)

    return {
        "n":                   int(len(df)),
        "VaR_distress_rate":   float(distress.mean()),     # target = 1 - beta
        "CoVaR_cond_viol":     float(cov_v),               # target = 1 - alpha
        "Avg_S_VaR":           float(df["S_VaR"].mean()),
        "Avg_S_CoVaR":         float(df["S_CoVaR"].mean()),
        "p_uc_VaR":            float(p_uc_var),
        "p_cc_VaR":            float(cc_var["p_cc"]),
        "p_uc_CoVaR":          float(p_uc_covar),
    }


def covar_diagnostics_table(results, beta=0.95, alpha=0.95):
    """Single-model diagnostics across (model, ref) -- long format."""
    rows = []
    for model, by_ref in results.items():
        if model.startswith("_"):
            continue
        for ref, df in by_ref.items():
            d = covar_diagnostics_one(df, beta=beta, alpha=alpha)
            d.update({"model": model, "reference": ref})
            rows.append(d)
    df = pd.DataFrame(rows)
    # Tidy column order
    return df[["model", "reference", "n",
               "VaR_distress_rate", "CoVaR_cond_viol",
               "Avg_S_VaR", "Avg_S_CoVaR",
               "p_uc_VaR", "p_cc_VaR", "p_uc_CoVaR"]]


def covar_lex_ranking(pooled, level=0.05):
    """Lexicographic tournament ranking of the multivariate models.

    Implements the Dimitriadis & Hoga (2026) lex rule directly from the pooled
    pairwise Diebold-Mariano table: a model scores a "win" on a component
    against an opponent when its expected score is significantly lower (a
    significant negative DM if it is model A, or a significant positive DM if
    it is model B). Models are ranked by significant $S_{VaR}$ wins first (the
    primary lexicographic component), with $S_{CoVaR}$ wins as the tie-break.
    This is the multivariate analogue of reading the lowest-FZ model off the
    univariate pooled-DM table.

    Returns a DataFrame with one row per model: lex_rank, S_VaR_wins,
    S_CoVaR_wins (each out of the n_models - 1 opponents).
    """
    models = sorted(set(pooled["model_a"]) | set(pooled["model_b"]))
    wins = {m: {"S_VaR": 0, "S_CoVaR": 0} for m in models}

    for _, r in pooled.iterrows():
        if r["p_value"] >= level:
            continue  # not a significant win for either side
        comp = r["component"]
        winner = r["model_a"] if r["DM"] < 0 else r["model_b"]
        wins[winner][comp] += 1

    rows = [{"model": m,
             "S_VaR_wins":   wins[m]["S_VaR"],
             "S_CoVaR_wins": wins[m]["S_CoVaR"]}
            for m in models]
    df = (pd.DataFrame(rows)
            .sort_values(["S_VaR_wins", "S_CoVaR_wins"], ascending=False)
            .reset_index(drop=True))
    df.insert(0, "lex_rank", range(1, len(df) + 1))
    return df


def covar_rejection_counts(single, level=0.05):
    """Per-model rejection counts (out of the nine reference pairs) for each
    coverage test, the multivariate counterpart of the univariate Table 8.

    Lower counts indicate better calibration.
    """
    rows = []
    for model, g in single.groupby("model"):
        rows.append({
            "model":                  model,
            "Kupiec UC (VaR)":        int((g["p_uc_VaR"]   < level).sum()),
            "Christoffersen CC (VaR)":int((g["p_cc_VaR"]   < level).sum()),
            "Kupiec UC (CoVaR)":      int((g["p_uc_CoVaR"] < level).sum()),
            "n_pairs":                int(len(g)),
        })
    return pd.DataFrame(rows)


# =========================================================================
# 3. Pairwise Diebold-Mariano on each score component
# =========================================================================


def dm_pairwise_covar(results, drop_outliers_q=0.005):
    """Pairwise DM tests between every model on S_VaR and on S_CoVaR.

    For each (model_a, model_b) pair and each reference asset, computes:
      * DM on S_VaR
      * DM on S_CoVaR
    Then pools across pairs.

    Returns (per_pair_df, pooled_df).
    """
    models = [m for m in results.keys() if not m.startswith("_")]
    refs   = list(results[models[0]].keys())

    per_pair_rows = []
    pooled_rows   = []

    for a, b in combinations(models, 2):
        var_diffs = []
        covar_diffs = []
        for ref in refs:
            if ref not in results[a] or ref not in results[b]:
                continue
            da = results[a][ref]
            db = results[b][ref]
            # Align on Date if available.
            if "Date" in da.columns and "Date" in db.columns:
                m = pd.merge(
                    da[["Date", "S_VaR", "S_CoVaR"]],
                    db[["Date", "S_VaR", "S_CoVaR"]],
                    on="Date", suffixes=(f"_{a}", f"_{b}"),
                )
                L1v, L2v = m[f"S_VaR_{a}"].values,   m[f"S_VaR_{b}"].values
                L1c, L2c = m[f"S_CoVaR_{a}"].values, m[f"S_CoVaR_{b}"].values
            else:
                T = min(len(da), len(db))
                L1v, L2v = da["S_VaR"].values[-T:],   db["S_VaR"].values[-T:]
                L1c, L2c = da["S_CoVaR"].values[-T:], db["S_CoVaR"].values[-T:]

            DMv, pv, mv = diebold_mariano(L1v, L2v, drop_outliers_q=drop_outliers_q)
            DMc, pc, mc = diebold_mariano(L1c, L2c, drop_outliers_q=drop_outliers_q)

            per_pair_rows.append({
                "model_a":  a, "model_b": b, "reference": ref,
                "DM_S_VaR":   DMv, "p_S_VaR":   pv, "diff_S_VaR":   mv,
                "DM_S_CoVaR": DMc, "p_S_CoVaR": pc, "diff_S_CoVaR": mc,
            })

            dv = L1v - L2v
            dc = L1c - L2c
            for arr, lst in [(dv, var_diffs), (dc, covar_diffs)]:
                arr = arr[np.isfinite(arr)]
                if drop_outliers_q is not None and len(arr) > 0:
                    cap = np.quantile(np.abs(arr), 1.0 - drop_outliers_q)
                    arr = np.clip(arr, -cap, cap)
                lst.append(arr)

        # Pool across all references for this (a, b) pair.
        for component, diffs in [("S_VaR", var_diffs), ("S_CoVaR", covar_diffs)]:
            pooled = np.concatenate(diffs) if diffs else np.array([])
            if len(pooled) < 10:
                continue
            from scipy.stats import norm as _norm
            T = len(pooled)
            mean_d = float(np.mean(pooled))
            LRV = newey_west_lrv(pooled)
            DM = mean_d / np.sqrt(LRV / T)
            p = 2.0 * (1.0 - _norm.cdf(abs(DM)))
            pooled_rows.append({
                "model_a": a, "model_b": b,
                "component": component,
                "DM": float(DM), "p_value": float(p),
                "mean_diff": mean_d, "n_obs": T,
            })

    return pd.DataFrame(per_pair_rows), pd.DataFrame(pooled_rows)


# =========================================================================
# 4. Top-level driver
# =========================================================================


def run_full_covar_backtest(models=None, cache_path=CACHE_PATH,
                             force=False, drop_outliers_q=0.005,
                             beta=0.95, alpha=0.95):
    """End-to-end: load/compute results, run all diagnostics, return
    (single_diagnostics, dm_per_pair, dm_pooled)."""
    results = load_or_compute_covar_results(
        cache_path=cache_path, models=models, force=force,
        beta=beta, alpha=alpha,
    )

    print()
    print("--- Single-model (VaR, CoVaR) diagnostics ---")
    single = covar_diagnostics_table(results, beta=beta, alpha=alpha)

    print("--- Pairwise Diebold-Mariano on S_VaR and S_CoVaR ---")
    per_pair, pooled = dm_pairwise_covar(results, drop_outliers_q=drop_outliers_q)

    return single, per_pair, pooled, results


if __name__ == "__main__":
    print(f"=== COVAR_BACKTEST ===")
    print(f"  cache : {CACHE_PATH}")
    print(f"  models: {BASELINE_MULTV_MODELS}")

    single, per_pair, pooled, results = run_full_covar_backtest(
        models=BASELINE_MULTV_MODELS, cache_path=CACHE_PATH,
    )

    print("\n========== SINGLE-MODEL (VaR, CoVaR) DIAGNOSTICS ==========")
    print(single.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n========== COVERAGE-TEST REJECTIONS (out of 9 pairs, 5% level) ==========")
    rej = covar_rejection_counts(single, level=0.05)
    print(rej.to_string(index=False))

    if len(pooled) > 0:
        print("\n========== POOLED DM ON S_VaR + S_CoVaR ==========")
        print(pooled.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print("\n========== LEXICOGRAPHIC RANKING (S_VaR primary, S_CoVaR tie-break) ==========")
        print(covar_lex_ranking(pooled, level=0.05).to_string(index=False))

    # If CoCAViaR was included, also report which spec won per reference.
    extras = results.get("_extras", {})
    best_specs = extras.get("_CoCAViaR_best_specs")
    if best_specs:
        print("\n========== CoCAViaR LEX-BEST SPEC PER REFERENCE ==========")
        for ref, spec in best_specs.items():
            print(f"  {ref:18s} -> {spec}")
