"""
FORECAST_COMBO.py

Forecast combination for (VaR, ES) following Taylor (2020):
``Forecast combinations for value at risk and expected shortfall'',
International Journal of Forecasting 36(2): 428-441.

Combines the four univariate dynamic forecasters (QRF, QGB, GAS, GARCH-t)
in `backtest_cache.pkl` with convex weights on the simplex
    w in Delta^4  iff  w_i >= 0 and sum(w_i) = 1,
optimised by minimising the Fissler-Ziegel L_FZ0 loss on a rolling
500-day training window, refit every 100 days. The optimised weights are
applied to the next 100 out-of-sample forecasts, mirroring the rolling
protocol of the individual models.

Output: combo_cache.pkl with one DataFrame per stock and the weight
history. The DataFrames have the standard schema (Date, Actual, VaR_1%,
VaR_5%, ES_1%, ES_5%, FZ_5%, FZ_1%) so the existing output.py /
extract_outputs.py pipeline can produce per-stock CSVs and plots.

Usage:
    cd repo
    python FORECAST_COMBO.py            # use baseline cache (QRF, QGB, GAS, GARCH)
    python FORECAST_COMBO.py --aug      # use augmented cache (*_AUG models)
"""

import os
import sys
import pickle
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from losses import fz_loss


# ========================================================================= #
# Weight optimisation                                                        #
# ========================================================================= #


def _project_to_simplex(w):
    """Euclidean projection of a vector onto the unit simplex.

    Implements the algorithm of Wang & Carreira-Perpinan (2013).
    """
    n = len(w)
    u = np.sort(w)[::-1]
    cs = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cs / (np.arange(n) + 1) > 0)[0][-1]
    lam = cs[rho] / (rho + 1)
    return np.maximum(w - lam, 0.0)


def optimise_weights(var_mat, es_mat, y, alpha):
    """Find convex weights w in the simplex that minimise the FZ_0 loss.

    Parameters
    ----------
    var_mat, es_mat : np.ndarray, shape (T, K)
        Per-day forecasts of VaR and ES from K models on T training days.
    y : np.ndarray, shape (T,)
        Realised returns on the same T days.
    alpha : float
        Confidence level (e.g. 0.05 or 0.01).

    Returns
    -------
    w : np.ndarray, shape (K,)
        Optimal convex weights, sum to 1.
    """
    K = var_mat.shape[1]

    def objective(w_raw):
        # Softmax reparameterisation guarantees w in the simplex without
        # constrained optimisation. Subtract max for numerical stability.
        w = np.exp(w_raw - np.max(w_raw))
        w = w / w.sum()
        v = var_mat @ w
        e = es_mat @ w
        loss = fz_loss(y, v, e, alpha=alpha)
        if not np.all(np.isfinite(loss)):
            return 1e10
        return float(np.mean(loss))

    # Initialise from equal weights.
    w0 = np.zeros(K)
    res = minimize(objective, w0, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-6, "disp": False})

    w = np.exp(res.x - np.max(res.x))
    w = w / w.sum()
    return w


# ========================================================================= #
# Rolling combination per stock                                              #
# ========================================================================= #


def combine_per_stock(model_dfs, stock, window_size=500, refit_every=100):
    """Run rolling forecast combination for one stock.

    Parameters
    ----------
    model_dfs : dict {model_name: forecast DataFrame}
        Each DataFrame has columns Date, Actual, VaR_5%, VaR_1%, ES_5%, ES_1%.
    stock : str
        Stock name (used for logging).
    window_size : int
        Rolling training window for weight optimisation.
    refit_every : int
        Re-optimise weights every this many days.

    Returns
    -------
    out : pd.DataFrame  Combined forecasts on the schema (Date, Actual,
                       VaR_5%, VaR_1%, ES_5%, ES_1%, FZ_5%, FZ_1%).
    weights_log : list of (Date, w_5%, w_1%) tuples for diagnostics.
    """
    model_names = list(model_dfs.keys())

    # Align all models on common dates.
    dates_sets = [set(df["Date"].astype(str)) for df in model_dfs.values()]
    common = sorted(set.intersection(*dates_sets))
    if len(common) < window_size + refit_every:
        return None, []

    aligned = {}
    for m, df in model_dfs.items():
        d = df.copy()
        d["Date"] = d["Date"].astype(str)
        d = d[d["Date"].isin(common)].sort_values("Date").reset_index(drop=True)
        aligned[m] = d

    y = aligned[model_names[0]]["Actual"].values.astype(float)
    dates = aligned[model_names[0]]["Date"].values
    n = len(common)

    var5_mat = np.column_stack([aligned[m]["VaR_5%"].values for m in model_names])
    es5_mat  = np.column_stack([aligned[m]["ES_5%"].values  for m in model_names])
    var1_mat = np.column_stack([aligned[m]["VaR_1%"].values for m in model_names])
    es1_mat  = np.column_stack([aligned[m]["ES_1%"].values  for m in model_names])

    v5_combo = np.full(n, np.nan)
    e5_combo = np.full(n, np.nan)
    v1_combo = np.full(n, np.nan)
    e1_combo = np.full(n, np.nan)

    w5 = np.full(len(model_names), 1.0 / len(model_names))
    w1 = w5.copy()
    weights_log = []

    for t in range(window_size, n):
        if (t - window_size) % refit_every == 0:
            # Refit weights on the last `window_size` days.
            w5 = optimise_weights(
                var5_mat[t - window_size:t], es5_mat[t - window_size:t],
                y[t - window_size:t], alpha=0.05,
            )
            w1 = optimise_weights(
                var1_mat[t - window_size:t], es1_mat[t - window_size:t],
                y[t - window_size:t], alpha=0.01,
            )
            weights_log.append((dates[t], w5.copy(), w1.copy()))

        v5_combo[t] = var5_mat[t] @ w5
        e5_combo[t] = es5_mat[t]  @ w5
        v1_combo[t] = var1_mat[t] @ w1
        e1_combo[t] = es1_mat[t]  @ w1

    out = pd.DataFrame({
        "Date":   dates,
        "Actual": y,
        "VaR_5%": v5_combo,
        "ES_5%":  e5_combo,
        "VaR_1%": v1_combo,
        "ES_1%":  e1_combo,
    }).dropna().reset_index(drop=True)

    out["FZ_5%"] = fz_loss(out["Actual"].values, out["VaR_5%"].values,
                            out["ES_5%"].values, alpha=0.05)
    out["FZ_1%"] = fz_loss(out["Actual"].values, out["VaR_1%"].values,
                            out["ES_1%"].values, alpha=0.01)
    return out, weights_log


# ========================================================================= #
# Driver                                                                     #
# ========================================================================= #


def plot_combining_weights(weights_history, models,
                           save_to="output/combo_weights_grid.png"):
    """Plot the cross-sectional average combining weights over time.

    One panel per confidence level (5% and 1%), reproducing the
    weight-evolution view of Taylor (2020, Figs 2--4). The weights are
    averaged across the assets that share the full refit history, so the
    figure summarises which constituent the minimum-score combination
    trusts at each point in time, and how that allocation shifts across
    volatility regimes.

    ``weights_history`` is the dict produced by ``run_combination``:
    {stock: [(date, w5, w1), ...]} with w5, w1 arrays ordered as ``models``.
    """
    stocks = list(weights_history.keys())
    if not stocks:
        return None
    n = max(len(weights_history[s]) for s in stocks)
    full = [s for s in stocks if len(weights_history[s]) == n]
    ref = weights_history[full[0]]
    dates = [pd.to_datetime(d) for (d, _, _) in ref]

    w5 = np.array([[e[1] for e in weights_history[s]] for s in full]).mean(0)
    w1 = np.array([[e[2] for e in weights_history[s]] for s in full]).mean(0)

    palette = {"QRF": "tab:blue", "QGB": "tab:red",
               "GAS": "black", "GARCH": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.5))
    for ax, wm, lvl in [(axes[0], w5, "5%"), (axes[1], w1, "1%")]:
        for j, m in enumerate(models):
            ax.step(dates, wm[:, j], where="post", label=m,
                    color=palette.get(m), linewidth=2.2)
        ax.set_title(f"Equally-weighted portfolio, {lvl} level", fontsize=15)
        ax.set_ylabel("Combining weight", fontsize=13)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
        ax.grid(alpha=0.2)
    fig.suptitle("Portfolio-average minimum-score combining weights "
                 "(equally-weighted across the asset cross-section)",
                 fontsize=16)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    plt.savefig(save_to, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved weight plot -> {save_to}")
    return fig


def run_combination(cache_path, models, save_to):
    """Load cache, run combination per stock, save results."""
    print(f"\n=== FORECAST COMBO from {cache_path} ===")
    print(f"  models: {models}")

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    available = [m for m in models if m in cache]
    if len(available) < 2:
        raise SystemExit(f"Need at least 2 models in cache. Got {available}.")
    print(f"  available: {available}")

    stocks = sorted(set.intersection(*[set(cache[m].keys()) for m in available]))
    print(f"  stocks (n={len(stocks)}): {stocks}")

    combo_results = {}
    weights_history = {}

    for stock in stocks:
        model_dfs = {m: cache[m][stock] for m in available}
        out, wlog = combine_per_stock(model_dfs, stock=stock)
        if out is None:
            print(f"  {stock}: insufficient overlap, skipped")
            continue
        combo_results[stock] = out
        weights_history[stock] = wlog

        v5 = (out["Actual"] < out["VaR_5%"]).mean()
        v1 = (out["Actual"] < out["VaR_1%"]).mean()
        fz5 = out["FZ_5%"].mean()
        fz1 = out["FZ_1%"].mean()
        print(f"  {stock:18s}  n={len(out):4d}  viol 5%={v5:.4f}  viol 1%={v1:.4f}  "
              f"FZ5={fz5:.4f}  FZ1={fz1:.4f}")

    with open(save_to, "wb") as f:
        pickle.dump({"results": combo_results,
                     "weights": weights_history,
                     "models": available}, f)
    print(f"\n  saved -> {save_to}")

    try:
        plot_combining_weights(weights_history, available)
    except Exception as e:
        print(f"  (weight plot skipped: {e})")

    return combo_results, weights_history


if __name__ == "__main__":
    use_aug = "--aug" in sys.argv

    if use_aug:
        cache_path = "backtest_cache_aug.pkl"
        models = ["QRF_AUG", "QGB_AUG", "GAS_AUG", "GARCH_AUG"]
        save_to = "combo_cache_aug.pkl"
    else:
        cache_path = "backtest_cache.pkl"
        models = ["QRF", "QGB", "GAS", "GARCH"]
        save_to = "combo_cache.pkl"

    run_combination(cache_path, models, save_to)
