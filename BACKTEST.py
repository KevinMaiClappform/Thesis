"""
BACKTEST.py — formal evaluation of (VaR, ES) forecasts.

Implements the standard backtests for univariate tail-risk models:

  Per (model, stock, alpha):
    * Kupiec unconditional coverage    (LR_uc, chi2(1))
    * Christoffersen independence       (LR_ind, chi2(1))
    * Christoffersen conditional cov.   (LR_cc, chi2(2))
    * Engle-Manganelli dynamic quantile (DQ,    chi2(K))
    * Acerbi-Szekely Z2 ES test         (bootstrap p-value)

  Pairwise across models (per stock, alpha):
    * Diebold-Mariano on FZ0 loss differentials with Newey-West HAC variance.

The test functions are loss-agnostic and dependency-light (numpy + scipy);
the driver `run_full_backtest()` wires them up to the QRF / QGB / GAS /
GARCH `run_all_stocks()` exports of this repo.

References
----------
- Kupiec (1995), J. Derivatives.
- Christoffersen (1998), Int. Econ. Review.
- Engle & Manganelli (2004), J. Bus. Econ. Stat.
- Diebold & Mariano (1995), J. Bus. Econ. Stat.
- Newey & West (1987), Econometrica.
- Acerbi & Szekely (2014), Risk magazine.
- Patton, Ziegel & Chen (2019), J. Econometrics (FZ0 loss for DM).
"""

import os
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from losses import fz_loss


# ========================================================================= #
# 1. Single-model coverage tests                                            #
# ========================================================================= #


def kupiec_uc(hits, alpha):
    """Kupiec (1995) unconditional coverage LR test.

    H0: P(Y_t < VaR_t) = alpha.

    Parameters
    ----------
    hits : array-like of {0, 1}
        Violation indicators 1{Y_t < VaR_t}.
    alpha : float
        Nominal tail probability.

    Returns
    -------
    (LR_uc, p_value, hit_rate)
    """
    hits = np.asarray(hits, dtype=int)
    T = len(hits)
    x = int(hits.sum())
    pi_hat = x / T if T > 0 else 0.0

    # Log-likelihoods with the 0*log(0) := 0 convention.
    def _xlogy(a, b):
        return 0.0 if a == 0 else a * np.log(b)

    log_L0 = _xlogy(T - x, 1 - alpha) + _xlogy(x, alpha)
    log_L1 = _xlogy(T - x, 1 - pi_hat) + _xlogy(x, pi_hat)

    LR = -2.0 * (log_L0 - log_L1)
    p = 1.0 - chi2.cdf(LR, df=1)
    return LR, p, pi_hat


def christoffersen_ind(hits):
    """Christoffersen (1998) independence LR test on the hit sequence.

    H0: hits are i.i.d. Bernoulli (no first-order dependence).

    Returns
    -------
    (LR_ind, p_value)
    """
    hits = np.asarray(hits, dtype=int)
    if len(hits) < 2:
        return 0.0, 1.0

    # Transition counts.
    prev = hits[:-1]
    curr = hits[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    n = n00 + n01 + n10 + n11
    if n == 0:
        return 0.0, 1.0

    pi = (n01 + n11) / n
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

    def _xlogy(a, b):
        return 0.0 if (a == 0 or b == 0) else a * np.log(b)

    log_L_indep = (_xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi))
    log_L_dep   = (_xlogy(n00, 1 - pi01) + _xlogy(n01, pi01)
                 + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11))

    LR = -2.0 * (log_L_indep - log_L_dep)
    LR = max(LR, 0.0)  # numerical floor
    p = 1.0 - chi2.cdf(LR, df=1)
    return LR, p


def christoffersen_cc(hits, alpha):
    """Christoffersen (1998) conditional coverage LR test.

    LR_cc = LR_uc + LR_ind, distributed chi2(2) under H0 of correct
    unconditional coverage AND independence.

    Returns
    -------
    dict with keys: LR_uc, p_uc, LR_ind, p_ind, LR_cc, p_cc, hit_rate.
    """
    LR_uc, p_uc, pi_hat = kupiec_uc(hits, alpha)
    LR_ind, p_ind = christoffersen_ind(hits)
    LR_cc = LR_uc + LR_ind
    p_cc = 1.0 - chi2.cdf(LR_cc, df=2)
    return dict(LR_uc=LR_uc, p_uc=p_uc,
                LR_ind=LR_ind, p_ind=p_ind,
                LR_cc=LR_cc, p_cc=p_cc,
                hit_rate=pi_hat)


def dq_test(hits, var, alpha, lags=4):
    """Engle & Manganelli (2004) Dynamic Quantile (DQ) test.

    Regressors: intercept + `lags` lagged hits + the contemporaneous VaR
    forecast itself. Test statistic is

        DQ = Hit' X (X'X)^{-1} X' Hit / (alpha * (1 - alpha))   ~ chi2(K)

    under H0 of a correctly specified VaR model, where K = number of
    columns of X and Hit_t = 1{Y_t < VaR_t} - alpha is the centred hit.

    Returns
    -------
    (DQ_stat, p_value, K)
    """
    hits = np.asarray(hits, dtype=float)
    var = np.asarray(var, dtype=float)
    T = len(hits)
    if T <= lags + 2:
        return np.nan, np.nan, np.nan

    Hit = hits[lags:] - alpha
    n = len(Hit)

    cols = [np.ones(n)]
    for k in range(1, lags + 1):
        cols.append(hits[lags - k: T - k])
    cols.append(var[lags:])
    X = np.column_stack(cols)
    K = X.shape[1]

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    DQ = float(Hit @ X @ XtX_inv @ X.T @ Hit / (alpha * (1.0 - alpha)))
    DQ = max(DQ, 0.0)
    p = 1.0 - chi2.cdf(DQ, df=K)
    return DQ, p, K


# ========================================================================= #
# 2. Acerbi-Szekely Z2 ES test                                              #
# ========================================================================= #


def acerbi_szekely_z2(y, var, es, alpha, n_boot=1000, seed=0):
    """Acerbi & Szekely (2014) Z2 unconditional ES test.

    Statistic
    ---------
        Z2 = mean_t [ Y_t * I_t / (alpha * ES_t) ] + 1
        I_t = 1{Y_t < VaR_t}

    Under H0 (correct ES forecast), E[Z2] = 0. P-values are obtained by a
    bootstrap of the empirical residual distribution under H0 by permuting
    the hit indicator (a simple non-parametric proxy for the standard
    Acerbi-Szekely Monte-Carlo simulation).

    Returns
    -------
    (Z2_stat, p_value)
    """
    y = np.asarray(y, dtype=float)
    var = np.asarray(var, dtype=float)
    es = np.asarray(es, dtype=float)
    es = np.where(es >= -1e-8, -1e-8, es)

    hits = (y < var).astype(float)
    Z2 = float(np.mean(y * hits / (alpha * es)) + 1.0)

    # Bootstrap p-value: resample the realised returns within the period
    # (preserving VaR / ES forecasts) to approximate the null distribution.
    rng = np.random.default_rng(seed)
    T = len(y)
    sims = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, T, size=T)
        y_b = y[idx]
        hits_b = (y_b < var).astype(float)
        sims[b] = np.mean(y_b * hits_b / (alpha * es)) + 1.0

    # Two-sided p-value: empirical fraction at least as extreme as observed.
    p = float(np.mean(np.abs(sims - np.mean(sims)) >= np.abs(Z2 - np.mean(sims))))
    return Z2, p


# ========================================================================= #
# 3. Diebold-Mariano with Newey-West HAC variance                           #
# ========================================================================= #


def newey_west_lrv(d, h=None):
    """Newey-West (1987) long-run variance estimator with Bartlett kernel.

    Parameters
    ----------
    d : array-like
        Loss differential series.
    h : int or None
        Truncation lag. Defaults to floor(T^{1/3}).

    Returns
    -------
    LRV : float
    """
    d = np.asarray(d, dtype=float)
    d = d - d.mean()
    T = len(d)
    if h is None:
        h = max(int(np.floor(T ** (1.0 / 3.0))), 1)

    gamma0 = float(np.dot(d, d) / T)
    LRV = gamma0
    for k in range(1, h + 1):
        gamma_k = float(np.dot(d[k:], d[:-k]) / T)
        w = 1.0 - k / (h + 1.0)
        LRV += 2.0 * w * gamma_k
    return max(LRV, 1e-12)


def diebold_mariano(loss1, loss2, h=None, drop_outliers_q=None):
    """Diebold-Mariano (1995) test of equal predictive accuracy.

    H0: E[L1_t - L2_t] = 0  (the two forecasts have equal expected loss).
    Negative DM statistic => model 1 is better.

    Parameters
    ----------
    loss1, loss2 : array-like
        Realised losses of the two competing forecasts on the same sample.
    h : int or None
        Newey-West truncation lag.
    drop_outliers_q : float or None
        If given, winsorise |d_t| above the (1 - q) quantile to limit the
        influence of single-observation FZ0 blow-ups (e.g. SPY in QGB).
        Recommended: 0.005 (drops the top 0.5% in absolute value).

    Returns
    -------
    (DM_stat, p_value, mean_diff)
    """
    L1 = np.asarray(loss1, dtype=float)
    L2 = np.asarray(loss2, dtype=float)
    d = L1 - L2
    mask = np.isfinite(d)
    d = d[mask]
    if len(d) < 10:
        return np.nan, np.nan, np.nan

    if drop_outliers_q is not None and 0 < drop_outliers_q < 0.5:
        cap = np.quantile(np.abs(d), 1 - drop_outliers_q)
        d = np.clip(d, -cap, cap)

    T = len(d)
    mean_d = float(np.mean(d))
    LRV = newey_west_lrv(d, h=h)
    DM = mean_d / np.sqrt(LRV / T)
    p = 2.0 * (1.0 - norm.cdf(abs(DM)))
    return float(DM), float(p), mean_d


# ========================================================================= #
# 4. Per-stock and cross-model drivers                                      #
# ========================================================================= #


def backtest_one(res, alpha, name=""):
    """Run all single-model tests on one (stock, model, alpha).

    `res` must be a DataFrame with columns Actual, VaR_{alpha}, ES_{alpha}.
    """
    pct = f"{int(alpha * 100)}%"
    var_col = f"VaR_{pct}"
    es_col  = f"ES_{pct}"

    y   = res["Actual"].values
    v   = res[var_col].values
    e   = res[es_col].values
    hits = (y < v).astype(int)

    cc = christoffersen_cc(hits, alpha)
    DQ, p_dq, K = dq_test(hits, v, alpha, lags=4)
    Z2, p_z2 = acerbi_szekely_z2(y, v, e, alpha, n_boot=500)

    out = dict(name=name, alpha=alpha, T=len(y), n_hits=int(hits.sum()))
    out.update(cc)
    out.update(dict(DQ=DQ, p_dq=p_dq, dq_K=K, Z2=Z2, p_z2=p_z2))
    return out


def backtest_all_models(model_results, alphas=(0.05, 0.01)):
    """`model_results`: dict {model_name: {stock_name: forecast_df}}.

    Returns a long-format DataFrame with one row per (model, stock, alpha)
    and a column per test statistic.
    """
    rows = []
    for model, per_stock in model_results.items():
        for stock, df in per_stock.items():
            for a in alphas:
                r = backtest_one(df, a, name=f"{model}|{stock}")
                r.update(dict(model=model, stock=stock))
                rows.append(r)
    return pd.DataFrame(rows)


def dm_pairwise(model_results, alpha=0.05, drop_outliers_q=0.005):
    """Pairwise DM tests across models, per stock and pooled.

    Returns
    -------
    per_stock : DataFrame
        Rows: stock; columns: MultiIndex of (model_a vs model_b, stat).
    pooled : DataFrame
        Rows: model_a vs model_b; columns: DM_stat, p_value, mean_diff,
        n_stocks. Pooling stacks loss differentials across all stocks.
    """
    pct = f"{int(alpha * 100)}%"
    fz_col = f"FZ_{pct}"
    models = list(model_results.keys())
    stocks = list(next(iter(model_results.values())).keys())

    per_stock_rows = []
    pooled_rows = []

    for a, b in combinations(models, 2):
        # Pooled differential across all stocks (after individual winsorising).
        pooled_d = []
        for s in stocks:
            df_a = model_results[a][s]
            df_b = model_results[b][s]
            # Align on Date if both have it; else assume same index.
            if "Date" in df_a.columns and "Date" in df_b.columns:
                m = pd.merge(df_a[["Date", fz_col]], df_b[["Date", fz_col]],
                             on="Date", suffixes=(f"_{a}", f"_{b}"))
                L1 = m[f"{fz_col}_{a}"].values
                L2 = m[f"{fz_col}_{b}"].values
            else:
                T = min(len(df_a), len(df_b))
                L1 = df_a[fz_col].values[-T:]
                L2 = df_b[fz_col].values[-T:]

            DM, p, md = diebold_mariano(L1, L2, drop_outliers_q=drop_outliers_q)
            per_stock_rows.append(dict(stock=s, model_a=a, model_b=b,
                                       DM=DM, p_value=p, mean_diff=md))
            d = L1 - L2
            d = d[np.isfinite(d)]
            if drop_outliers_q is not None:
                cap = np.quantile(np.abs(d), 1 - drop_outliers_q) if len(d) > 0 else np.inf
                d = np.clip(d, -cap, cap)
            pooled_d.append(d)

        pooled_d = np.concatenate(pooled_d) if pooled_d else np.array([])
        if len(pooled_d) >= 10:
            T = len(pooled_d)
            mean_d = float(np.mean(pooled_d))
            LRV = newey_west_lrv(pooled_d)
            DM = mean_d / np.sqrt(LRV / T)
            pval = 2.0 * (1.0 - norm.cdf(abs(DM)))
            pooled_rows.append(dict(model_a=a, model_b=b,
                                    DM=float(DM), p_value=float(pval),
                                    mean_diff=mean_d, n_obs=T))

    per_stock_df = pd.DataFrame(per_stock_rows)
    pooled_df = pd.DataFrame(pooled_rows)
    return per_stock_df, pooled_df


# ========================================================================= #
# 5. Top-level driver                                                       #
# ========================================================================= #


CACHE_PATH = "backtest_cache.pkl"


def load_or_compute_results(cache_path=CACHE_PATH, models=None, force=False):
    """Run the four univariate models and cache the results to disk.

    Re-running the four `run_all_stocks` takes minutes; subsequent calls
    load the cached pickle in milliseconds.
    """
    if (not force) and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if models is None:
        models = ["QRF", "QGB", "GAS", "GARCH"]

    out = {}
    for m in models:
        print(f"--- Running {m} ---")
        if m == "QRF":
            from QRF import run_all_stocks
        elif m == "QGB":
            from QGB import run_all_stocks
        elif m == "GAS":
            from GAS import run_all_stocks
        elif m == "GARCH":
            from GARCH import run_all_stocks
        else:
            raise ValueError(m)
        out[m] = run_all_stocks(verbose=False)

    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    return out


def run_full_backtest(alphas=(0.05, 0.01), models=None, cache_path=CACHE_PATH,
                      force=False, drop_outliers_q=0.005):
    """End-to-end: load (or compute) model results, run all tests, return
    three DataFrames: single-model results, per-stock DM, pooled DM."""
    results = load_or_compute_results(cache_path=cache_path,
                                      models=models, force=force)

    print("--- Single-model coverage tests ---")
    single = backtest_all_models(results, alphas=alphas)

    dm_per_stock = {}
    dm_pooled = {}
    for a in alphas:
        print(f"--- Pairwise DM (alpha = {a}) ---")
        per_stock, pooled = dm_pairwise(results, alpha=a,
                                        drop_outliers_q=drop_outliers_q)
        dm_per_stock[a] = per_stock
        dm_pooled[a] = pooled

    return single, dm_per_stock, dm_pooled


if __name__ == "__main__":
    single, dm_per_stock, dm_pooled = run_full_backtest()

    print("\n========== SINGLE-MODEL COVERAGE TESTS ==========")
    cols = ["model", "stock", "alpha", "hit_rate",
            "p_uc", "p_ind", "p_cc", "p_dq", "p_z2"]
    print(single[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    for a, df in dm_pooled.items():
        print(f"\n========== POOLED DM-FZ0  (alpha = {a}) ==========")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
