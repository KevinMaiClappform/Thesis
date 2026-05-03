"""
GARCH(1,1) with standardised Student-t innovations for VaR and ES.

Bollerslev (1986) volatility recursion + Bollerslev (1987) t-innovations.
Conditional VaR and ES are derived analytically from the fitted innovation
distribution using McNeil, Frey & Embrechts (2015, eq. 2.31).

Public API
----------
- t_var_es_factor(alpha, nu): closed-form (q, e) for std-t at level alpha.
- rolling_garch(y, dates, window_size, refit_every): one-step OOS forecasts.
- run_all_stocks(files, data_dir): convenience driver.
"""

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

from arch import arch_model
from scipy.stats import t as student_t, norm

from features import load_returns, DEFAULT_FILES
from losses import fz_loss


def t_var_es_factor(alpha, nu):
    """VaR and ES factors of a *standardized* Student-t innovation
    (variance 1, d.o.f. nu) at level alpha.

    Uses the closed-form tail expectation of the Student-t from
    McNeil, Frey & Embrechts (2005, eq. 2.31):

        E[X | X <= q] = -f_T(q) * (nu + q^2) / [(nu - 1) * alpha]

    where q = t-quantile at alpha and f_T is the t pdf. Both q and the
    tail expectation are then rescaled by sqrt((nu - 2) / nu) so that the
    innovation has unit variance.

    Returns
    -------
    (q_std, es_std) : tuple of float
    """
    q_t = student_t.ppf(alpha, df=nu)
    pdf_q = student_t.pdf(q_t, df=nu)
    es_unstd = -pdf_q * (nu + q_t ** 2) / ((nu - 1) * alpha)

    scale = np.sqrt((nu - 2) / nu)
    return q_t * scale, es_unstd * scale


def normal_var_es_factor(alpha):
    """VaR and ES factors of a standard normal innovation at level alpha."""
    q = norm.ppf(alpha)
    es = -norm.pdf(q) / alpha
    return q, es


def _fit_garch_t(y_train):
    """Fit GARCH(1,1)-t on returns multiplied by 100 (arch convention)."""
    am = arch_model(
        y_train * 100,
        vol="GARCH", p=1, q=1,
        mean="Constant",
        dist="t",
        rescale=False,
    )
    return am.fit(disp="off", show_warning=False, options={"maxiter": 200})


def rolling_garch(y, dates, window_size=500, refit_every=100):
    """
    Out-of-sample GARCH(1,1)-t forecasts for VaR_alpha and ES_alpha at
    alpha in {0.01, 0.05}.

    Returns
    -------
    var_1, var_5, es_1, es_5 : tuple of np.ndarray of length len(y)
        First `window_size` entries are NaN.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    var_5 = np.full(n, np.nan); es_5 = np.full(n, np.nan)
    var_1 = np.full(n, np.nan); es_1 = np.full(n, np.nan)

    mu_s = omega = a_p = b_p = nu = sigma2_state = None
    q5 = e5 = q1 = e1 = None

    for t in range(window_size, n):

        if (t - window_size) % refit_every == 0:
            res = _fit_garch_t(y[t - window_size:t])
            mu_s = float(res.params["mu"])
            omega = float(res.params["omega"])
            a_p = float(res.params["alpha[1]"])
            b_p = float(res.params["beta[1]"])
            nu = float(res.params["nu"])

            cv = np.asarray(res.conditional_volatility)
            sigma2_state = float(cv[-1]) ** 2

            q5, e5 = t_var_es_factor(0.05, nu)
            q1, e1 = t_var_es_factor(0.01, nu)

        eps_prev_s = y[t - 1] * 100 - mu_s
        sigma2_state = omega + a_p * eps_prev_s ** 2 + b_p * sigma2_state
        sigma_s = np.sqrt(sigma2_state)

        var_5[t] = (mu_s + sigma_s * q5) / 100
        es_5[t] = (mu_s + sigma_s * e5) / 100
        var_1[t] = (mu_s + sigma_s * q1) / 100
        es_1[t] = (mu_s + sigma_s * e1) / 100

    return var_1, var_5, es_1, es_5


def run_all_stocks(files=None, data_dir="data", verbose=True):
    if files is None:
        files = DEFAULT_FILES

    out = {}
    for f in files:
        if verbose:
            print(f"Running GARCH(1,1)-t for {f}...")
        path = os.path.join(data_dir, f) if data_dir else f
        df = load_returns(path)
        y = df["DlyRet"].values
        dates = df["DlyCalDt"].values

        var_1, var_5, es_1, es_5 = rolling_garch(y, dates)

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
    run_all_stocks()
