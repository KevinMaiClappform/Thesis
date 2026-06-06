"""
COVAR_ML.py -- Machine-learning (QRF / QGB) forecaster for (VaR, CoVaR).

Implements a machine-learning CoVaR forecaster in the spirit of the
neural-network quantile-regression CoVaR of Keilbar & Wang (2022), but with
the tree-based estimators (QRF, QGB) that suit the data regime of this
thesis. This is the multivariate analogue of QRF.py / QGB.py and the
machine-learning complement to COCAVIAR.py (CoCAViaR) and DCCGARCH.py.

Modeling philosophy (distress-conditioned two-step quantile regression)
-----------------------------------------------------------------------
CoVaR is itself a conditional quantile, so it can be estimated with a
flexible quantile estimator. We deviate in one deliberate respect from the
neural-network CoVaR of Keilbar & Wang (2022, eqs. 12-13): they fit the
quantile of Y on X on the full sample and then *evaluate* the fitted surface
at the point X = VaR(X), which targets the original Adrian & Brunnermeier
(2016) "equality" CoVaR Q_alpha(Y | X = VaR(X)). This thesis instead
evaluates every multivariate forecast under the Dimitriadis & Hoga (2026)
lexicographic score, whose CoVaR component S_CoVaR = 1{x>v}(1{y<=c}-alpha)(c-y)
is censored by the *inequality* distress event {X > VaR(X)} and is therefore
minimised by the "inequality" CoVaR Q_alpha(Y | X > VaR(X)). To target the
quantity the score actually rewards, we condition on the distress event
directly -- i.e. we fit the Step-2 quantile estimator on the distress
sub-sample {t : X_t > VaR(X)_t} -- which is the data-side analogue of the
distress indicator that Dimitriadis & Hoga (2026) place inside their CoVaR
score. The cost is the well-known data-hungriness of CoVaR estimation (only
~(1-beta) of the sample is retained), which we mitigate with a smaller leaf
size in Step 2.

Two-step procedure
------------------
For each (reference, system) pair, on the rolling 500-day window refit every
100 days, with beta = alpha = 0.95 in the losses convention of Dimitriadis &
Hoga (2026):

  Step 1 -- Reference VaR.
    A QRF / QGB quantile estimator forecasts VaR_beta(X) via the same dynamic
    two-pass recursion (warm-up fit, then refit with the lagged own VaR
    forecast appended) as the univariate QRF.py / QGB.py.

  Step 2 -- Conditional CoVaR quantile on the distress sub-sample.
    The distress event for selecting the Step-2 training subsample is defined
    by the empirical beta-quantile of the rolling-window losses, which yields
    ~(1-beta)*window distress days every refit regardless of the Step-1
    estimator's in-sample coverage. On that subsample a second QRF / QGB is
    fit to Q_alpha(Y | conditioning info), again with a two-pass recursion
    through the lagged own CoVaR forecast. The reported VaR_X forecast is the
    dynamic Step-1 quantile; only the Step-2 conditioning uses the empirical
    threshold.

Public API
----------
- rolling_ml_covar(model_type, X_loss, Y_loss, dates, ...): pairwise OOS
  (VaR, CoVaR) forecaster.
- run_all_pairs(model_type=..., reference_files=..., system_file=...):
  driver matching COVAR.run_all_pairs / DCCGARCH.run_all_pairs so the model
  plugs into COVAR_BACKTEST.

References
----------
- Keilbar & Wang (2022). Modelling systemic risk using neural-network
  quantile regression. Empirical Economics.
- Adrian & Brunnermeier (2016). CoVaR. Amer. Econ. Rev.
- Meinshausen (2006). Quantile regression forests. J. Mach. Learn. Res.
- Bauer (2024); Velthoen et al. (2023). Gradient-boosted quantile regression.
- Dimitriadis & Hoga (2026). Dynamic CoVaR modeling and estimation. JBES.
"""

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

from quantile_forest import RandomForestQuantileRegressor
import lightgbm as lgb

from features import load_returns, DEFAULT_FILES
from losses import s_var, s_covar


# =========================================================================
# 1. Feature construction
# =========================================================================


def _make_features_covar(X, Y, n_lags=5):
    """Base feature matrix used by both Step 1 (VaR) and Step 2 (CoVaR).

    Features (in order):
        lag_X_1, lag_Y_1, ..., lag_X_k, lag_Y_k,
        rv5_X, rv5_Y, rv22_X, rv22_Y

    The first max(n_lags, 22) rows contain NaN due to the lag and rolling
    operations; callers mask these out before fitting.
    """
    n = len(X)
    n_feats = 2 * n_lags + 4
    feats = np.full((n, n_feats), np.nan)

    sx = pd.Series(X)
    sy = pd.Series(Y)

    col = 0
    for lag in range(1, n_lags + 1):
        feats[:, col] = sx.shift(lag).values; col += 1
        feats[:, col] = sy.shift(lag).values; col += 1

    feats[:, col] = sx.rolling(5).std().values;  col += 1
    feats[:, col] = sy.rolling(5).std().values;  col += 1
    feats[:, col] = sx.rolling(22).std().values; col += 1
    feats[:, col] = sy.rolling(22).std().values; col += 1

    return feats


# =========================================================================
# 2. Quantile-model fit / predict dispatch
# =========================================================================


def _fit_quantile_model(model_type, X, y, q, min_leaf=None):
    """Fit a quantile estimator of the q-quantile of y given X.

    The Step-1 (VaR) hyperparameters mirror the univariate QRF.py / QGB.py
    settings. The Step-2 (CoVaR) fit passes a smaller ``min_leaf`` because the
    distress sub-sample is only ~(1-beta)*window observations (~25 days), for
    which the 500-day leaf sizes would force a degenerate single-leaf
    (constant-per-window) forecast.
    """
    if model_type == "QRF":
        leaf = min_leaf if min_leaf is not None else 10
        return RandomForestQuantileRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=leaf,
            random_state=42, n_jobs=-1,
        ).fit(X, y)
    if model_type == "QGB":
        leaf = min_leaf if min_leaf is not None else 20
        return lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=100, learning_rate=0.05,
            num_leaves=7, min_data_in_leaf=leaf,
            random_state=42, verbose=-1,
        ).fit(X, y)
    raise ValueError(f"Unknown model_type {model_type!r}; expected QRF or QGB.")


def _predict_quantile(model_type, model, X, q):
    """Predict the q-quantile from the fitted model. Always returns a 1D
    array of length len(X)."""
    if model_type == "QRF":
        # quantile_forest returns 1D for a single-quantile request and 2D for
        # multi-quantile; normalise to 1D either way.
        out = np.asarray(model.predict(X, quantiles=[q]))
        return out.ravel() if out.ndim == 1 else out[:, 0]
    if model_type == "QGB":
        return model.predict(X)
    raise ValueError(f"Unknown model_type {model_type!r}.")


# =========================================================================
# 3. Rolling-window two-step CoVaR forecaster
# =========================================================================


def rolling_ml_covar(model_type, X_loss, Y_loss, dates,
                     window_size=500, refit_every=100,
                     beta=0.95, alpha=0.95, n_lags=5,
                     min_distress=10, verbose=False):
    """Out-of-sample rolling-window machine-learning (VaR, CoVaR) forecast.

    Parameters
    ----------
    model_type : {"QRF", "QGB"}
        Underlying quantile estimator.
    X_loss, Y_loss : array-like
        Reference and system loss series (positive = bad), in the
        Dimitriadis & Hoga (2026) convention X = -return.
    dates : array-like
        Date index aligned with X_loss / Y_loss.
    window_size : int
        Rolling training-window length (default 500).
    refit_every : int
        Refit cadence in days (default 100).
    beta, alpha : float
        Confidence levels for VaR and CoVaR (default 0.95 each).
    n_lags : int
        Number of lagged losses used as features (default 5).
    min_distress : int
        Minimum number of distress days in a training window required to fit
        the Step-2 CoVaR model. With the empirical-quantile threshold this is
        ~(1-beta)*window (~25) and rarely binds.
    verbose : bool
        If True, print one diagnostic line per refit.

    Returns
    -------
    var_fc, covar_fc : tuple of np.ndarray of length len(X_loss)
        First ``window_size`` entries are NaN.
    """
    if model_type not in {"QRF", "QGB"}:
        raise ValueError(f"Unknown model_type {model_type!r}.")

    X = np.asarray(X_loss, dtype=float)
    Y = np.asarray(Y_loss, dtype=float)
    n = len(X)
    assert len(Y) == n

    feats = _make_features_covar(X, Y, n_lags=n_lags)

    var_fc   = np.full(n, np.nan)
    covar_fc = np.full(n, np.nan)

    # Memory of the model's own forecasts for the dynamic recursion.
    mem_var   = np.full(n, np.nan)
    mem_covar = np.full(n, np.nan)

    var_model = covar_model = None

    for t in range(window_size, n):

        if (t - window_size) % refit_every == 0:
            tr = slice(t - window_size, t)
            X_tr_b = feats[tr]
            X_tr   = X[tr]
            Y_tr   = Y[tr]
            tr_valid = ~np.isnan(X_tr_b).any(axis=1)

            if tr_valid.sum() < 50:
                var_model = covar_model = None
                continue

            # ----------------------------------------------------------------
            # Step 1: VaR(X) at level beta -- two-pass dynamic training
            # ----------------------------------------------------------------
            warmup_v = _fit_quantile_model(
                model_type, X_tr_b[tr_valid], X_tr[tr_valid], beta
            )
            insample_var0 = np.full(window_size, np.nan)
            insample_var0[tr_valid] = _predict_quantile(
                model_type, warmup_v, X_tr_b[tr_valid], beta
            )

            lag_v  = np.r_[np.nan, insample_var0[:-1]]
            X_tr_v = np.column_stack([X_tr_b, lag_v])
            valid_v = ~np.isnan(X_tr_v).any(axis=1)

            var_model = _fit_quantile_model(
                model_type, X_tr_v[valid_v], X_tr[valid_v], beta
            )
            insample_var = np.full(window_size, np.nan)
            insample_var[valid_v] = _predict_quantile(
                model_type, var_model, X_tr_v[valid_v], beta
            )

            # ----------------------------------------------------------------
            # Step 2: CoVaR(Y|X) at level alpha on the distress sub-sample
            # ----------------------------------------------------------------
            # Distress threshold for selecting the Step-2 training subsample:
            # the empirical beta-quantile of the rolling-window losses. This
            # guarantees ~(1-beta)*window distress days every refit, regardless
            # of the Step-1 estimator's in-sample coverage, so the subsample is
            # never starved. The reported VaR_X forecast is still the dynamic
            # Step-1 quantile; only the conditioning event uses this threshold.
            distress_thr = np.quantile(X_tr[tr_valid], beta)
            distress = (X_tr > distress_thr) & tr_valid
            n_dist = int(distress.sum())

            if n_dist < min_distress:
                if verbose:
                    print(f"    t={t}: only {n_dist} distress days, "
                          f"skipping CoVaR fit (VaR only).")
                covar_model = None
                mem_var[t - 1] = insample_var[-1]
                continue

            # The Step-2 fits run on the small (~25-day) distress subsample, so
            # they use a smaller leaf than the 500-day Step-1 VaR fit.
            step2_leaf = 5

            warmup_c = _fit_quantile_model(
                model_type, X_tr_b[distress], Y_tr[distress], alpha,
                min_leaf=step2_leaf,
            )
            insample_covar0 = np.full(window_size, np.nan)
            insample_covar0[tr_valid] = _predict_quantile(
                model_type, warmup_c, X_tr_b[tr_valid], alpha
            )

            lag_c  = np.r_[np.nan, insample_covar0[:-1]]
            X_tr_c = np.column_stack([X_tr_b, lag_c])
            valid_c = ~np.isnan(X_tr_c).any(axis=1)
            dist_dyn = distress & valid_c

            if dist_dyn.sum() < min_distress:
                covar_model = None
                mem_var[t - 1] = insample_var[-1]
                continue

            covar_model = _fit_quantile_model(
                model_type, X_tr_c[dist_dyn], Y_tr[dist_dyn], alpha,
                min_leaf=step2_leaf,
            )
            insample_covar = np.full(window_size, np.nan)
            insample_covar[valid_c] = _predict_quantile(
                model_type, covar_model, X_tr_c[valid_c], alpha
            )

            mem_var[t - 1]   = insample_var[-1]
            mem_covar[t - 1] = insample_covar[-1]

            if verbose:
                print(f"    refit at t={t}: {n_dist} distress days "
                      f"(target {(1 - beta) * window_size:.0f}).")

        # ----------------------------------------------------------------
        # Out-of-sample prediction at t
        # ----------------------------------------------------------------
        if var_model is None or np.isnan(feats[t]).any():
            continue

        x_v = np.concatenate([feats[t], [mem_var[t - 1]]]).reshape(1, -1)
        if np.isnan(x_v).any():
            continue
        v = float(_predict_quantile(model_type, var_model, x_v, beta)[0])
        var_fc[t] = v
        mem_var[t] = v

        if covar_model is not None and not np.isnan(mem_covar[t - 1]):
            x_c = np.concatenate([feats[t], [mem_covar[t - 1]]]).reshape(1, -1)
            c = float(_predict_quantile(model_type, covar_model, x_c, alpha)[0])
            covar_fc[t] = c
            mem_covar[t] = c

    return var_fc, covar_fc


# =========================================================================
# 4. Driver -- all (reference, system) pairs
# =========================================================================


def run_all_pairs(model_type="QRF", reference_files=None,
                  system_file="SPY.csv", data_dir="data", verbose=True,
                  window_size=500, refit_every=100,
                  beta=0.95, alpha=0.95, n_lags=5):
    """Run rolling_ml_covar on every (reference, system) pair.

    Returns
    -------
    dict {reference_name: DataFrame}
        Each DataFrame has columns Date, X_loss, Y_loss, VaR_X, CoVaR_Y,
        S_VaR, S_CoVaR -- the same schema as COCAVIAR.py / DCCGARCH.py output so
        the multivariate backtest framework can consume it directly.
    """
    if reference_files is None:
        reference_files = [f for f in DEFAULT_FILES if f != system_file]

    sys_path = os.path.join(data_dir, system_file) if data_dir else system_file
    sys_df = load_returns(sys_path).rename(columns={"DlyRet": "Ret_Y"})

    out = {}
    for f in reference_files:
        ref_name = f.replace(".csv", "")
        if verbose:
            print(f"Running ML CoVaR ({model_type}) for {ref_name} | "
                  f"{system_file.replace('.csv', '')}...")

        ref_path = os.path.join(data_dir, f) if data_dir else f
        ref_df = load_returns(ref_path).rename(columns={"DlyRet": "Ret_X"})
        pair = ref_df.merge(sys_df, on="DlyCalDt", how="inner").reset_index(drop=True)

        X_loss = -pair["Ret_X"].values
        Y_loss = -pair["Ret_Y"].values
        dates  = pair["DlyCalDt"].values

        var_fc, covar_fc = rolling_ml_covar(
            model_type, X_loss, Y_loss, dates,
            window_size=window_size, refit_every=refit_every,
            beta=beta, alpha=alpha, n_lags=n_lags,
        )

        res = pd.DataFrame({
            "Date":    dates,
            "X_loss":  X_loss,
            "Y_loss":  Y_loss,
            "VaR_X":   var_fc,
            "CoVaR_Y": covar_fc,
        }).dropna().reset_index(drop=True)

        res["S_VaR"]   = s_var(res["VaR_X"].values, res["X_loss"].values, beta=beta)
        res["S_CoVaR"] = s_covar(res["VaR_X"].values,  res["CoVaR_Y"].values,
                                  res["X_loss"].values, res["Y_loss"].values, alpha=alpha)

        out[ref_name] = res
        if verbose:
            distress = res["X_loss"] > res["VaR_X"]
            n_d = max(1, int(distress.sum()))
            cv = ((res["Y_loss"] > res["CoVaR_Y"]) & distress).sum() / n_d
            vd = distress.mean()
            print(f"  {ref_name}: VaR distress = {vd:.3f} (target {1-beta:.2f}) "
                  f"cond CoVaR violation = {cv:.3f} (target {1-alpha:.2f})  "
                  f"avg S_VaR = {res['S_VaR'].mean():.5f}  "
                  f"avg S_CoVaR = {res['S_CoVaR'].mean():.5f}")
    return out


if __name__ == "__main__":
    import sys
    from output_covar import report

    model_type = "QGB" if "--qgb" in sys.argv else "QRF"
    print(f"=== COVAR_ML: model_type = {model_type} ===")
    results = run_all_pairs(model_type=model_type)
    report(results, model_name=f"COVAR_{model_type}", show_var=False)
