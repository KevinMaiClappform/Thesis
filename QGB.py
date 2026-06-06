"""
Dynamic Quantile Gradient Boosting for VaR and ES.

Uses LightGBM with the pinball-loss objective (Bauer 2024; Velthoen et al.
2023). One independent quantile-LGBM is fit per quantile in
{0.01, 0.05, 0.50} ∪ ES_GRID_1 ∪ ES_GRID_5; ES is computed as the simple
average over the lower-tail grid. No post-hoc monotonicity correction or
ES <= VaR clamp is applied, so the forecasts faithfully reflect the
underlying model and may occasionally cross.

Same dynamic recursion structure as QRF.py: lagged own (VaR, ES) forecasts
are appended to the base feature set.

Public API
----------
- rolling_qgb(df, window_size, refit_every): main forecaster.
- run_all_stocks(files, data_dir): convenience driver.
"""

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import numpy as np
import pandas as pd
import lightgbm as lgb

from features import (make_lag_features, make_lag_features_realized,
                      load_returns, DEFAULT_FILES)
from losses import fz_loss


ES_GRID_5 = np.linspace(0.005, 0.05, 10)
ES_GRID_1 = np.linspace(0.001, 0.01, 10)

_VAR_QS = [0.01, 0.05, 0.50]
_ALL_QS = sorted(set(list(_VAR_QS) + list(ES_GRID_1) + list(ES_GRID_5)))


def _fit_lgb_quantile(X, y, alpha):
    """Conservative settings tuned for n_train=500: more trees / leaves
    over-fit hard on small windows and produce wild tail predictions."""
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=7,
        min_data_in_leaf=20,
        random_state=42,
        verbose=-1,
    ).fit(X, y)


def _fit_all(X_train, y_train):
    return {q: _fit_lgb_quantile(X_train, y_train, q) for q in _ALL_QS}


def _qgb_var_es(models, X):
    """Predict (VaR_1%, VaR_5%, Median, ES_1%, ES_5%) for every row of X.

    Each quantile is the raw output of its independently-fit LightGBM model.
    No post-hoc monotonicity correction or ES <= VaR clamp is applied, so
    the forecasts faithfully reflect the underlying model and may
    occasionally cross. Such crossings are part of the model's true output
    and inform discussion of its limitations rather than being hidden.
    """
    qs_sorted = sorted(_ALL_QS)
    raw = np.column_stack([models[q].predict(X) for q in qs_sorted])
    qmap = {q: raw[:, i] for i, q in enumerate(qs_sorted)}

    var_1 = qmap[0.01]
    var_5 = qmap[0.05]
    median = qmap[0.50]
    es_1 = np.mean([qmap[qq] for qq in ES_GRID_1], axis=0)
    es_5 = np.mean([qmap[qq] for qq in ES_GRID_5], axis=0)

    return {"VaR_1%": var_1, "VaR_5%": var_5, "Median": median,
            "ES_1%": es_1, "ES_5%": es_5}


def rolling_qgb(df, window_size=500, refit_every=100,
                use_realized=False, asset_name=None,
                intraday_dir="data_intraday"):
    """Dynamic (recursive) Quantile Gradient Boosting for VaR and ES.

    See `rolling_qrf` for the two-pass training rationale; this function is
    the gradient-boosting analogue with identical feature set and protocol.

    The ``use_realized`` flag turns this into the realized-augmented variant
    that adds lagged RV / BV / RR from ``<intraday_dir>/<asset>_realized.csv``
    to the feature set, restricting the effective sample to 2020-2025.

    Returns
    -------
    pandas.DataFrame
        Columns: Date, Actual, VaR_1%, VaR_5%, Median, ES_1%, ES_5%.
    """
    if use_realized:
        if asset_name is None:
            raise ValueError("asset_name is required when use_realized=True")
        df = make_lag_features_realized(df, asset_name=asset_name,
                                        n_lags=5, intraday_dir=intraday_dir)
        base_cols = [
            "lag_1", "lag_2", "lag_3", "lag_4", "lag_5",
            "rolling_vol_5", "rolling_vol_22",
            "lag_RV_5min", "lag_BV_5min", "lag_RR_5min",
        ]
    else:
        df = make_lag_features(df, n_lags=5)
        base_cols = [
            "lag_1", "lag_2", "lag_3", "lag_4", "lag_5",
            "rolling_vol_5", "rolling_vol_22",
        ]
    X_base = df[base_cols].values
    y = df["DlyRet"].values
    dates = df["DlyCalDt"].values
    n = len(df)

    mem = {k: np.full(n, np.nan) for k in ["VaR_1%", "VaR_5%", "ES_1%", "ES_5%"]}

    results = []
    models = None

    for t in range(window_size, n):

        if (t - window_size) % refit_every == 0:
            X_tr_b = X_base[t - window_size:t]
            y_tr = y[t - window_size:t]

            warmup_models = _fit_all(X_tr_b, y_tr)
            insample = _qgb_var_es(warmup_models, X_tr_b)

            lag_v1 = np.r_[np.nan, insample["VaR_1%"][:-1]]
            lag_v5 = np.r_[np.nan, insample["VaR_5%"][:-1]]
            lag_e1 = np.r_[np.nan, insample["ES_1%"][:-1]]
            lag_e5 = np.r_[np.nan, insample["ES_5%"][:-1]]

            X_tr_dyn = np.column_stack([X_tr_b, lag_v1, lag_v5, lag_e1, lag_e5])
            valid = ~np.isnan(X_tr_dyn).any(axis=1)

            models = _fit_all(X_tr_dyn[valid], y_tr[valid])

            mem["VaR_1%"][t - 1] = insample["VaR_1%"][-1]
            mem["VaR_5%"][t - 1] = insample["VaR_5%"][-1]
            mem["ES_1%"][t - 1] = insample["ES_1%"][-1]
            mem["ES_5%"][t - 1] = insample["ES_5%"][-1]

        x_test = np.concatenate([
            X_base[t],
            [mem["VaR_1%"][t - 1], mem["VaR_5%"][t - 1],
             mem["ES_1%"][t - 1], mem["ES_5%"][t - 1]],
        ]).reshape(1, -1)

        out = _qgb_var_es(models, x_test)
        var_1 = float(out["VaR_1%"][0])
        var_5 = float(out["VaR_5%"][0])
        median = float(out["Median"][0])
        es_1 = float(out["ES_1%"][0])
        es_5 = float(out["ES_5%"][0])

        mem["VaR_1%"][t] = var_1
        mem["VaR_5%"][t] = var_5
        mem["ES_1%"][t] = es_1
        mem["ES_5%"][t] = es_5

        results.append({
            "Date": dates[t],
            "Actual": y[t],
            "VaR_1%": var_1, "VaR_5%": var_5, "Median": median,
            "ES_1%": es_1, "ES_5%": es_5,
        })

    return pd.DataFrame(results)


def run_all_stocks(files=None, data_dir="data", verbose=True,
                   use_realized=False, intraday_dir="data_intraday"):
    if files is None:
        files = DEFAULT_FILES

    out = {}
    for f in files:
        stock = f.replace(".csv", "")
        if verbose:
            tag = " (+realized)" if use_realized else ""
            print(f"Running QGB{tag} for {stock}...")
        path = os.path.join(data_dir, f) if data_dir else f
        df = load_returns(path)
        res = rolling_qgb(df, use_realized=use_realized, asset_name=stock,
                          intraday_dir=intraday_dir)
        res["FZ_5%"] = fz_loss(res["Actual"].values, res["VaR_5%"].values, res["ES_5%"].values, alpha=0.05)
        res["FZ_1%"] = fz_loss(res["Actual"].values, res["VaR_1%"].values, res["ES_1%"].values, alpha=0.01)
        out[stock] = res
        if verbose:
            v5 = (res["Actual"] < res["VaR_5%"]).mean()
            v1 = (res["Actual"] < res["VaR_1%"]).mean()
            print(f"  {stock}: viol5={v5:.3f}  viol1={v1:.3f}  "
                  f"FZ5={res['FZ_5%'].mean():.3f}  FZ1={res['FZ_1%'].mean():.3f}")
    return out


if __name__ == "__main__":
    import sys
    from output import report
    use_realized = "--realized" in sys.argv
    model_name = "QGB_AUG" if use_realized else "QGB"
    results = run_all_stocks(use_realized=use_realized)
    report(results, model_name=model_name)
