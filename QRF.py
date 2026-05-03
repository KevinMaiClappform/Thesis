"""
Dynamic Quantile Random Forest for VaR and ES.

Implements Meinshausen (2006) QRF via leaf-membership weights (using the
quantile-forest package) extended to a recursive (VaR, ES) process: the
forecast at t depends on the model's own forecast at t-1 through lagged
features, in the spirit of Engle & Manganelli (2004) CAViaR and
Patton-Ziegel-Chen (2019).

Public API
----------
- rolling_qrf(df, window_size, refit_every): main forecaster.
- run_all_stocks(files, data_dir): convenience driver over a list of CSVs.
"""

import os
import numpy as np
import pandas as pd

from quantile_forest import RandomForestQuantileRegressor

from features import make_lag_features, load_returns, DEFAULT_FILES
from losses import fz_loss


# ES grids: average of conditional quantiles in (0, alpha], per draft eq. (10).
ES_GRID_5 = np.linspace(0.005, 0.05, 10)
ES_GRID_1 = np.linspace(0.001, 0.01, 10)

_VAR_QS = [0.01, 0.05, 0.50]
_ALL_QS = sorted(set(list(_VAR_QS) + list(ES_GRID_1) + list(ES_GRID_5)))


def _fit_qrf(X, y):
    """Meinshausen (2006) QRF via leaf-membership weights, not leaf means."""
    return RandomForestQuantileRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    ).fit(X, y)


def _qrf_var_es(model, X):
    """Predict (VaR_1%, VaR_5%, Median, ES_1%, ES_5%) for every row of X.
    ES is computed as the mean of conditional quantiles in (0, alpha]
    and is clamped to be at most as large as VaR (i.e. at least as extreme)."""
    q = model.predict(X, quantiles=list(_ALL_QS))
    qmap = {qq: q[:, i] for i, qq in enumerate(_ALL_QS)}

    var_1 = qmap[0.01]
    var_5 = qmap[0.05]
    median = qmap[0.50]
    es_1 = np.mean([qmap[qq] for qq in ES_GRID_1], axis=0)
    es_5 = np.mean([qmap[qq] for qq in ES_GRID_5], axis=0)

    es_1 = np.minimum(es_1, var_1)
    es_5 = np.minimum(es_5, var_5)

    return {"VaR_1%": var_1, "VaR_5%": var_5, "Median": median,
            "ES_1%": es_1, "ES_5%": es_5}


def rolling_qrf(df, window_size=500, refit_every=100):
    """
    Dynamic (recursive) Quantile Random Forest for VaR and ES.

    Two-pass training inside each refit window:
      Pass 1 (warmup):  fit QRF on base features only and produce in-sample
                        VaR/ES forecasts.
      Pass 2 (dynamic): augment features with the lagged Pass-1 forecasts
                        for VaR_1%, VaR_5%, ES_1%, ES_5% and refit QRF.

    At prediction time, the t-1 feature is the model's own forecast at
    t-1, propagated through a rolling memory across timesteps.
    Refit every `refit_every` days, matching Dimitriadis & Hoga (2026).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain `DlyCalDt`, `DlyRet`. Pre-sort recommended.
    window_size : int
        Rolling training window length, default 500.
    refit_every : int
        Refit period in days, default 100.

    Returns
    -------
    pandas.DataFrame
        Columns: Date, Actual, VaR_1%, VaR_5%, Median, ES_1%, ES_5%.
    """
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
    model = None

    for t in range(window_size, n):

        if (t - window_size) % refit_every == 0:
            X_tr_b = X_base[t - window_size:t]
            y_tr = y[t - window_size:t]

            warmup = _fit_qrf(X_tr_b, y_tr)
            insample = _qrf_var_es(warmup, X_tr_b)

            lag_v1 = np.r_[np.nan, insample["VaR_1%"][:-1]]
            lag_v5 = np.r_[np.nan, insample["VaR_5%"][:-1]]
            lag_e1 = np.r_[np.nan, insample["ES_1%"][:-1]]
            lag_e5 = np.r_[np.nan, insample["ES_5%"][:-1]]

            X_tr_dyn = np.column_stack([X_tr_b, lag_v1, lag_v5, lag_e1, lag_e5])
            valid = ~np.isnan(X_tr_dyn).any(axis=1)

            model = _fit_qrf(X_tr_dyn[valid], y_tr[valid])

            mem["VaR_1%"][t - 1] = insample["VaR_1%"][-1]
            mem["VaR_5%"][t - 1] = insample["VaR_5%"][-1]
            mem["ES_1%"][t - 1] = insample["ES_1%"][-1]
            mem["ES_5%"][t - 1] = insample["ES_5%"][-1]

        x_test = np.concatenate([
            X_base[t],
            [mem["VaR_1%"][t - 1], mem["VaR_5%"][t - 1],
             mem["ES_1%"][t - 1], mem["ES_5%"][t - 1]],
        ]).reshape(1, -1)

        out = _qrf_var_es(model, x_test)
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


def run_all_stocks(files=None, data_dir="data", verbose=True):
    """Run rolling_qrf over a list of CSVs and attach FZ losses.

    Returns dict {stock_name: forecast_dataframe_with_FZ_columns}.
    """
    if files is None:
        files = DEFAULT_FILES

    out = {}
    for f in files:
        if verbose:
            print(f"Running QRF for {f}...")
        path = os.path.join(data_dir, f) if data_dir else f
        df = load_returns(path)
        res = rolling_qrf(df)
        res["FZ_5%"] = fz_loss(res["Actual"].values, res["VaR_5%"].values, res["ES_5%"].values, alpha=0.05)
        res["FZ_1%"] = fz_loss(res["Actual"].values, res["VaR_1%"].values, res["ES_1%"].values, alpha=0.01)
        stock = f.replace(".csv", "")
        out[stock] = res
        if verbose:
            v5 = (res["Actual"] < res["VaR_5%"]).mean()
            v1 = (res["Actual"] < res["VaR_1%"]).mean()
            print(f"  {stock}: viol5={v5:.3f}  viol1={v1:.3f}  "
                  f"FZ5={res['FZ_5%'].mean():.3f}  FZ1={res['FZ_1%'].mean():.3f}")
    return out


if __name__ == "__main__":
    run_all_stocks()
