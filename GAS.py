"""
One-factor Generalized Autoregressive Score (GAS) model for VaR and ES.

Implements Patton, Ziegel & Chen (2019) Section 2.3 verbatim:

    v_t = a * exp(kappa_t),   e_t = b * exp(kappa_t),   b < a < 0
    kappa_t = omega + beta * kappa_{t-1} + gamma * s_{t-1}
    s_t = (1/e_t) * ((1/alpha) * 1{Y_t <= v_t} * Y_t - e_t)        (PZC eq. 18)

omega is fixed at 0 for identification (PZC p. 394).
Parameters (beta, gamma, a, b) estimated by minimising in-sample average
FZ0 loss on a rolling window.

Public API
----------
- gas_filter(params, y, alpha): forward recursion under given params.
- rolling_gas(y, dates, alpha, window_size, refit_every): one-step OOS forecasts.
- run_all_stocks(files, data_dir): convenience driver.
"""

import os
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import norm

from features import load_returns, DEFAULT_FILES
from losses import fz_loss


def gas_filter(params, y, alpha):
    """
    One-factor GAS recursion of Patton, Ziegel & Chen (2019), Section 2.3.

    Reparameterisation (so that constraints are satisfied unconstrained):
        beta  = sigmoid(beta_raw)   in (0, 1)
        gamma = exp(gamma_raw)      > 0
        a     = -exp(a_raw)         < 0
        b     = a - exp(d_raw)      < a < 0   (ensures ES <= VaR by construction)

    Returns
    -------
    var, es, kappa : tuple of np.ndarray
    """
    beta_raw, gamma_raw, a_raw, d_raw = params

    omega = 0.0
    beta = 1.0 / (1.0 + np.exp(-beta_raw))
    gamma = np.exp(gamma_raw)
    a = -np.exp(a_raw)
    b = a - np.exp(d_raw)

    T = len(y)
    kappa = np.empty(T)
    var = np.empty(T)
    es = np.empty(T)

    kappa[0] = np.log(np.std(y) + 1e-8)
    var[0] = a * np.exp(kappa[0])
    es[0] = b * np.exp(kappa[0])

    for t in range(1, T):
        prev_v = a * np.exp(kappa[t - 1])
        prev_e = b * np.exp(kappa[t - 1])
        ind = 1.0 if y[t - 1] <= prev_v else 0.0

        s = (1.0 / prev_e) * ((1.0 / alpha) * ind * y[t - 1] - prev_e)
        s = np.clip(s, -10, 10)

        kappa[t] = omega + beta * kappa[t - 1] + gamma * s
        kappa[t] = np.clip(kappa[t], -10, 5)

        var[t] = a * np.exp(kappa[t])
        es[t] = b * np.exp(kappa[t])

    return var, es, kappa


def _start_params(alpha):
    """Sensible starting values from the standard normal approximation
    (PZC 2019, footnote 4)."""
    q = norm.ppf(alpha)
    e = -norm.pdf(q) / alpha
    return np.array([
        3.0,                            # beta_raw  -> beta ~ 0.95
        -4.0,                           # gamma_raw -> gamma ~ 0.018
        np.log(abs(q)),                 # a_raw
        np.log(abs(e - q)),             # d_raw     -> b - a < 0
    ])


def _fit_gas(y, alpha, warm_start=None):
    """Fit one-factor GAS by minimising in-sample average FZ0 loss.

    Optimisation uses BFGS (quasi-Newton with finite-difference gradient).
    The FZ0 loss is smooth almost everywhere in the GAS parameter space --
    the only non-smoothness is the indicator 1{Y_t <= v_t}, which under a
    continuously distributed Y_t contributes only a measure-zero set of
    kinks. In practice BFGS converges faster and to lower in-sample FZ
    values than Nelder-Mead on this objective. As a safeguard we fall
    back to Nelder-Mead from the same starting value if BFGS does not
    converge or returns a worse objective.
    """
    y = np.asarray(y, dtype=float)

    def objective(params):
        var, es, _ = gas_filter(params, y, alpha)
        loss = fz_loss(y, var, es, alpha=alpha)
        if np.any(~np.isfinite(loss)):
            return 1e10
        return float(np.mean(loss))

    starts = [warm_start] if warm_start is not None else []
    starts.append(_start_params(alpha))

    best = None
    for x0 in starts:
        # Primary: BFGS (quasi-Newton, gradient via finite differences).
        res = minimize(objective, x0, method="BFGS",
                       options={"maxiter": 500, "gtol": 1e-6, "disp": False})

        # Fallback: Nelder-Mead from the same start if BFGS fails outright.
        if not np.isfinite(res.fun) or res.fun >= 1e9:
            res_nm = minimize(objective, x0, method="Nelder-Mead",
                              options={"maxiter": 3000, "xatol": 1e-6, "disp": False})
            if np.isfinite(res_nm.fun) and res_nm.fun < res.fun:
                res = res_nm

        if best is None or res.fun < best.fun:
            best = res
    return best


def _unpack(params):
    beta_raw, gamma_raw, a_raw, d_raw = params
    beta = 1.0 / (1.0 + np.exp(-beta_raw))
    gamma = np.exp(gamma_raw)
    a = -np.exp(a_raw)
    b = a - np.exp(d_raw)
    return beta, gamma, a, b


def rolling_gas(y, dates, alpha, window_size=500, refit_every=100):
    """
    Out-of-sample one-factor GAS forecast for VaR_alpha and ES_alpha.

    Estimation protocol matches the QRF / QGB / GARCH notebooks for fair
    comparison: rolling 500-day window, refit every 100 days, warm-started
    from previous solution. Between refits the recursion is propagated
    forward with the most recent parameter estimate.

    Returns
    -------
    var_fc, es_fc : tuple of np.ndarray of length len(y)
        First `window_size` entries are NaN.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    var_fc = np.full(n, np.nan)
    es_fc = np.full(n, np.nan)

    params = None
    beta_p = gamma_p = a_p = b_p = None
    omega = 0.0
    kappa_state = np.nan

    for t in range(window_size, n):

        if (t - window_size) % refit_every == 0:
            fit = _fit_gas(y[t - window_size:t], alpha=alpha, warm_start=params)
            params = fit.x
            beta_p, gamma_p, a_p, b_p = _unpack(params)

            _, _, kappa_train = gas_filter(params, y[t - window_size:t], alpha)
            kappa_state = kappa_train[-1]

        prev_v = a_p * np.exp(kappa_state)
        prev_e = b_p * np.exp(kappa_state)
        ind = 1.0 if y[t - 1] <= prev_v else 0.0
        s = (1.0 / prev_e) * ((1.0 / alpha) * ind * y[t - 1] - prev_e)
        s = np.clip(s, -10, 10)

        kappa_state = omega + beta_p * kappa_state + gamma_p * s
        kappa_state = np.clip(kappa_state, -10, 5)

        var_fc[t] = a_p * np.exp(kappa_state)
        es_fc[t] = b_p * np.exp(kappa_state)

    return var_fc, es_fc


def run_all_stocks(files=None, data_dir="data", verbose=True):
    if files is None:
        files = DEFAULT_FILES

    out = {}
    for f in files:
        if verbose:
            print(f"Running GAS for {f}...")
        path = os.path.join(data_dir, f) if data_dir else f
        df = load_returns(path)
        y = df["DlyRet"].values
        dates = df["DlyCalDt"].values

        var_5, es_5 = rolling_gas(y, dates, alpha=0.05)
        var_1, es_1 = rolling_gas(y, dates, alpha=0.01)

        res = pd.DataFrame({
            "Date": dates, "Actual": y,
            "VaR_1%": var_1, "VaR_5%": var_5,
            "ES_1%": es_1, "ES_5%": es_5,
        }).dropna().reset_index(drop=True)

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
    from output import report
    results = run_all_stocks()
    report(results, model_name="GAS")
