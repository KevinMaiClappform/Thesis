"""
DCC-GARCH(1,1) model for VaR and CoVaR forecasting.

Implements the Dynamic Conditional Correlation GARCH of Engle (2002) for
the bivariate (reference, system) setting used in the CoVaR analysis of
Dimitriadis & Hoga (2026, sec. 5.2). DCC-GARCH is the standard parametric
multivariate benchmark against which the CoCAViaR models are compared.

Two-step estimation
-------------------
Step 1: a univariate GARCH(1,1) is fit per asset (via the ``arch`` package).
        Lets D_t = diag(sigma_{X,t}, sigma_{Y,t}) and z_t = D_t^{-1} eps_t.

Step 2: the standardised residuals z_t are used to estimate the DCC(1,1)
        correlation dynamics:

            Q_t  = (1 - a - b) * Q_bar + a * z_{t-1} z_{t-1}' + b * Q_{t-1}
            R_t  = corr_from_Q(Q_t)

        with *variance targeting*: Q_bar is fixed at the sample correlation
        of z (this is the "intercept = unconditional cov" reduction the
        supervisor was referring to). The two remaining parameters (a, b)
        are estimated by maximum likelihood under bivariate normality of
        z_t given R_t.

The conditional joint distribution at t is then bivariate normal with
covariance D_t R_t D_t. VaR and CoVaR are obtained by Monte-Carlo from
this distribution following the bivariate-normal CoVaR formulation that
appears in Dimitriadis & Hoga (2026, sec. 5.2).

Conventions
-----------
Daily returns Y in this codebase are gains-positive, losses-negative.
Internally we convert to losses X = -Y so that VaR / CoVaR are in the
upper tail (matching the COCAVIAR.py convention).

Public API
----------
- rolling_dcc_covar(X_ret, Y_ret, dates, window_size, refit_every,
                    beta, alpha, n_sim): pairwise OOS CoVaR forecaster.
- run_all_pairs(reference_files, system_file, data_dir, verbose):
  driver matching COVAR.run_all_pairs (currently implicit) so the model
  plugs into COVAR_BACKTEST.

References
----------
- Engle, R. F. (2002). Dynamic conditional correlation. JBES, 20(3), 339-350.
- Engle, R. F. (2009). Anticipating Correlations. Princeton.
- Bauwens, Laurent, Rombouts (2006). Multivariate GARCH models: a survey.
  J. Appl. Econ., 21(1), 79-109.
- Dimitriadis, T. & Hoga, Y. (2026). Dynamic CoVaR Modeling and Estimation.
  J. Bus. Econ. Stat. (sec. 5.2 for DCC-GARCH CoVaR benchmark).
"""

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize

from features import load_returns, DEFAULT_FILES
from losses import s_var, s_covar


# =======================================================================
# 1. Step-1: univariate GARCH(1,1)
# =======================================================================


def _fit_univariate_garch(returns):
    """Fit GARCH(1,1) with constant mean and Gaussian innovations.

    Returns
    -------
    sigma : np.ndarray  (in *return* scale)
    z     : np.ndarray  (standardised residuals)
    params: dict  (mu, omega, a, b -- last three in the 100x-scaled space
                   used internally by the arch package)
    """
    returns = np.asarray(returns, dtype=float)
    am = arch_model(returns * 100, vol="GARCH", p=1, q=1,
                    mean="Constant", dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False, options={"maxiter": 200})

    params = dict(
        mu    = float(res.params["mu"]),       # in 100x scale
        omega = float(res.params["omega"]),
        a     = float(res.params["alpha[1]"]),
        b     = float(res.params["beta[1]"]),
    )
    sigma_scaled = np.asarray(res.conditional_volatility, dtype=float)
    sigma = sigma_scaled / 100.0
    # Standardised residuals (unitless).
    z = (returns - params["mu"] / 100.0) / sigma
    return sigma, z, params


def _garch_one_step(params, last_sigma2_scaled, last_eps_scaled):
    """One-step-ahead forecast of sigma_{t+1} given current GARCH state.

    All quantities in the 100x scale used internally. Returns the next
    sigma^2 in the 100x scale and sigma in the *return* scale.
    """
    sigma2_next_scaled = (params["omega"]
                          + params["a"] * last_eps_scaled ** 2
                          + params["b"] * last_sigma2_scaled)
    return sigma2_next_scaled, np.sqrt(sigma2_next_scaled) / 100.0


# =======================================================================
# 2. Step-2: DCC(1,1) correlation with variance targeting
# =======================================================================


def _fit_dcc_bivariate(z_x, z_y):
    """Maximum-likelihood estimation of the DCC(1,1) parameters (a, b) on
    standardised residuals (z_x, z_y), with variance targeting.

    Q_bar is fixed at the sample correlation matrix of z, eliminating the
    intercept terms and reducing the multivariate-GARCH parameter count
    (Aielli, 2013). Returns (a_hat, b_hat, rho_bar, Q_T) where Q_T is the
    pseudo-correlation matrix at the last observation, used to seed the
    out-of-sample recursion.
    """
    Z = np.column_stack([z_x, z_y]).astype(float)
    n = len(Z)
    rho_bar = float(np.corrcoef(Z[:, 0], Z[:, 1])[0, 1])
    Q_bar = np.array([[1.0, rho_bar], [rho_bar, 1.0]])

    def neg_ll(theta):
        a, b = theta
        if a < 1e-5 or b < 1e-5 or a + b > 0.999:
            return 1e10
        Q = Q_bar.copy()
        ll = 0.0
        for t in range(1, n):
            z_prev = Z[t - 1].reshape(2, 1)
            Q = (1.0 - a - b) * Q_bar + a * (z_prev @ z_prev.T) + b * Q
            d = np.sqrt(np.diag(Q))
            if d[0] <= 0 or d[1] <= 0:
                return 1e10
            rho = float(Q[0, 1] / (d[0] * d[1]))
            if abs(rho) >= 0.9999:
                return 1e10
            det_R = 1.0 - rho ** 2
            zt = Z[t]
            quad = (zt[0] ** 2 + zt[1] ** 2 - 2.0 * rho * zt[0] * zt[1]) / det_R
            # Bivariate normal log-likelihood relative to independence
            ll += -0.5 * np.log(det_R) - 0.5 * (quad - zt[0] ** 2 - zt[1] ** 2)
        return -ll

    res = minimize(neg_ll, x0=[0.05, 0.90], method="L-BFGS-B",
                   bounds=[(1e-4, 0.30), (1e-4, 0.99)])
    a_hat, b_hat = float(res.x[0]), float(res.x[1])

    # Replay the filter once more to get Q_T (state for the next step).
    Q = Q_bar.copy()
    for t in range(1, n):
        z_prev = Z[t - 1].reshape(2, 1)
        Q = (1.0 - a_hat - b_hat) * Q_bar + a_hat * (z_prev @ z_prev.T) + b_hat * Q
    return a_hat, b_hat, rho_bar, Q


def _dcc_one_step(Q_state, z_prev_pair, a, b, rho_bar):
    """One-step-ahead update of the DCC pseudo-correlation and the
    resulting correlation rho_{t+1}."""
    Q_bar = np.array([[1.0, rho_bar], [rho_bar, 1.0]])
    z_prev = np.asarray(z_prev_pair, dtype=float).reshape(2, 1)
    Q_next = (1.0 - a - b) * Q_bar + a * (z_prev @ z_prev.T) + b * Q_state
    d = np.sqrt(np.diag(Q_next))
    rho_next = float(Q_next[0, 1] / (d[0] * d[1]))
    rho_next = max(min(rho_next, 0.9999), -0.9999)
    return Q_next, rho_next


# =======================================================================
# 3. Monte-Carlo (VaR, CoVaR) from bivariate normal one-step-ahead
# =======================================================================


def _mc_var_covar(sigma_X, sigma_Y, rho, beta, alpha, n_sim=10000, seed=0):
    """Monte-Carlo estimator of (VaR_beta(X-loss), CoVaR_alpha|beta(Y-loss)).

    Draws (X_ret, Y_ret) ~ N(0, Sigma) one-step-ahead with
        Sigma = diag(sigma) * [[1, rho], [rho, 1]] * diag(sigma).
    Converts returns to losses (loss = -return). VaR = upper beta-quantile
    of L_X. CoVaR = upper alpha-quantile of L_Y restricted to L_X > VaR.
    Returns (VaR_X, CoVaR_Y), both as positive *loss* numbers.
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[sigma_X ** 2, rho * sigma_X * sigma_Y],
                    [rho * sigma_X * sigma_Y, sigma_Y ** 2]])
    samples = rng.multivariate_normal([0.0, 0.0], cov, size=n_sim)
    L_X = -samples[:, 0]
    L_Y = -samples[:, 1]

    var_X = float(np.quantile(L_X, beta))
    distress = L_X > var_X
    if distress.sum() < 5:
        return var_X, var_X
    return var_X, float(np.quantile(L_Y[distress], alpha))


# =======================================================================
# 4. Rolling-window DCC-GARCH CoVaR forecaster
# =======================================================================


def rolling_dcc_covar(X_returns, Y_returns, dates,
                     window_size=500, refit_every=100,
                     beta=0.95, alpha=0.95, n_sim=10000):
    """Out-of-sample rolling-window DCC-GARCH CoVaR forecast.

    Estimation protocol mirrors COCAVIAR.py for a fair head-to-head:
      - rolling training window of ``window_size`` days,
      - parameters re-estimated every ``refit_every`` days,
      - between refits, the univariate GARCH variance recursion and the
        DCC correlation recursion are propagated forward.

    Returns (var_fc, covar_fc) in *loss* coordinates (positive = bad,
    matching the COCAVIAR.py output). First ``window_size`` entries are NaN.
    """
    X_ret = np.asarray(X_returns, dtype=float)
    Y_ret = np.asarray(Y_returns, dtype=float)
    n = len(X_ret)
    assert len(Y_ret) == n

    var_fc   = np.full(n, np.nan)
    covar_fc = np.full(n, np.nan)

    # State carried between refits.
    g_X = g_Y = None                       # univariate GARCH params
    sigma2_X_state = sigma2_Y_state = None  # 100x scale
    z_x_state = z_y_state = 0.0
    a_dcc = b_dcc = rho_bar = None
    Q_state = None

    for t in range(window_size, n):

        if (t - window_size) % refit_every == 0:
            X_tr = X_ret[t - window_size:t]
            Y_tr = Y_ret[t - window_size:t]

            # Step 1: univariate GARCH per asset.
            sigma_X_path, z_X_path, g_X = _fit_univariate_garch(X_tr)
            sigma_Y_path, z_Y_path, g_Y = _fit_univariate_garch(Y_tr)

            # Step 2: DCC correlation.
            a_dcc, b_dcc, rho_bar, Q_state = _fit_dcc_bivariate(z_X_path, z_Y_path)

            # Initialise state at the end of the training window.
            sigma2_X_state = (sigma_X_path[-1] * 100.0) ** 2
            sigma2_Y_state = (sigma_Y_path[-1] * 100.0) ** 2
            z_x_state = float(z_X_path[-1])
            z_y_state = float(z_Y_path[-1])

        # ----- One-step-ahead GARCH variance forecast for time t. -----
        eps_X_scaled = X_ret[t - 1] * 100.0 - g_X["mu"]
        eps_Y_scaled = Y_ret[t - 1] * 100.0 - g_Y["mu"]

        sigma2_X_next, sigma_X_next = _garch_one_step(g_X, sigma2_X_state, eps_X_scaled)
        sigma2_Y_next, sigma_Y_next = _garch_one_step(g_Y, sigma2_Y_state, eps_Y_scaled)

        # Standardised residual at t-1 for the DCC update.
        z_x_tm1 = eps_X_scaled / np.sqrt(sigma2_X_state) if sigma2_X_state > 0 else 0.0
        z_y_tm1 = eps_Y_scaled / np.sqrt(sigma2_Y_state) if sigma2_Y_state > 0 else 0.0

        # ----- One-step-ahead DCC correlation forecast for time t. -----
        Q_state, rho_next = _dcc_one_step(Q_state, [z_x_tm1, z_y_tm1],
                                          a_dcc, b_dcc, rho_bar)

        # ----- Monte-Carlo (VaR, CoVaR) from bivariate normal. -----
        var_fc[t], covar_fc[t] = _mc_var_covar(
            sigma_X_next, sigma_Y_next, rho_next,
            beta=beta, alpha=alpha, n_sim=n_sim,
            seed=12345 + t,           # deterministic but varies over t
        )

        # ----- Roll state forward. -----
        sigma2_X_state = sigma2_X_next
        sigma2_Y_state = sigma2_Y_next
        z_x_state = z_x_tm1
        z_y_state = z_y_tm1

    return var_fc, covar_fc


# =======================================================================
# 4b. Full-sample parameter estimates (for the results table)
# =======================================================================


def dcc_param_table(reference_files=None, system_file="SPY.csv",
                    data_dir="data", verbose=True):
    """Full-sample DCC-GARCH parameter estimates per (reference, SPY) pair.

    Reports the distinctive correlation parameters: the unconditional
    correlation ``rho_bar`` (the variance-targeting intercept, fixed at the
    sample correlation of the standardised residuals, Aielli 2013) and the
    DCC(1,1) dynamics ``a`` and ``b`` with their persistence ``a + b``. Each
    pair additionally uses univariate GARCH(1,1)-Gaussian fits whose volatility
    dynamics mirror the GARCH-t benchmark; those are not re-tabulated here.

    Returns a pandas DataFrame with one row per reference asset.
    """
    if reference_files is None:
        reference_files = [f for f in DEFAULT_FILES if f != system_file]

    sys_path = os.path.join(data_dir, system_file) if data_dir else system_file
    sys_df = load_returns(sys_path).rename(columns={"DlyRet": "Ret_Y"})

    rows = []
    for f in reference_files:
        ref_name = f.replace(".csv", "")
        if verbose:
            print(f"Fitting DCC-GARCH (full sample) for {ref_name} | "
                  f"{system_file.replace('.csv', '')} ...")
        ref_path = os.path.join(data_dir, f) if data_dir else f
        ref_df = load_returns(ref_path).rename(columns={"DlyRet": "Ret_X"})
        pair = ref_df.merge(sys_df, on="DlyCalDt", how="inner").reset_index(drop=True)

        _, z_x, _ = _fit_univariate_garch(pair["Ret_X"].values)
        _, z_y, _ = _fit_univariate_garch(pair["Ret_Y"].values)
        a_dcc, b_dcc, rho_bar, _ = _fit_dcc_bivariate(z_x, z_y)

        rows.append({
            "Reference": ref_name,
            "rho_bar": float(rho_bar),
            "a": float(a_dcc),
            "b": float(b_dcc),
            "ab": float(a_dcc + b_dcc),
        })
    return pd.DataFrame(rows)


# =======================================================================
# 5. Driver -- pair-wise, matching COCAVIAR.py output shape
# =======================================================================


def run_all_pairs(reference_files=None, system_file="SPY.csv",
                  data_dir="data", verbose=True,
                  window_size=500, refit_every=100,
                  beta=0.95, alpha=0.95, n_sim=10000):
    """Run rolling_dcc_covar on every (reference, system) pair.

    Returns
    -------
    dict {reference_name: DataFrame with columns Date, X_loss, Y_loss,
                                          VaR_X, CoVaR_Y, S_VaR, S_CoVaR}
    The S_VaR and S_CoVaR columns are the lex-loss components from
    Dimitriadis & Hoga (2026, eq. 5) so the output is directly
    comparable to COVAR.rolling_cocaviar's downstream usage.
    """
    if reference_files is None:
        reference_files = [f for f in DEFAULT_FILES if f != system_file]

    sys_path = os.path.join(data_dir, system_file) if data_dir else system_file
    sys_df = load_returns(sys_path).rename(columns={"DlyRet": "Ret_Y"})

    out = {}
    for f in reference_files:
        ref_name = f.replace(".csv", "")
        if verbose:
            print(f"Running DCC-GARCH for {ref_name} | {system_file.replace('.csv','')}...")

        ref_path = os.path.join(data_dir, f) if data_dir else f
        ref_df = load_returns(ref_path).rename(columns={"DlyRet": "Ret_X"})
        pair = ref_df.merge(sys_df, on="DlyCalDt", how="inner").reset_index(drop=True)

        X_returns = pair["Ret_X"].values
        Y_returns = pair["Ret_Y"].values
        dates     = pair["DlyCalDt"].values

        var_fc, covar_fc = rolling_dcc_covar(
            X_returns, Y_returns, dates,
            window_size=window_size, refit_every=refit_every,
            beta=beta, alpha=alpha, n_sim=n_sim,
        )

        # In loss coordinates, matching the COCAVIAR.py output convention.
        X_loss = -X_returns
        Y_loss = -Y_returns

        res = pd.DataFrame({
            "Date":   dates,
            "X_loss": X_loss,
            "Y_loss": Y_loss,
            "VaR_X":   var_fc,
            "CoVaR_Y": covar_fc,
        }).dropna().reset_index(drop=True)

        res["S_VaR"]   = s_var(res["VaR_X"].values,    res["X_loss"].values, beta=beta)
        res["S_CoVaR"] = s_covar(res["VaR_X"].values,  res["CoVaR_Y"].values,
                                 res["X_loss"].values, res["Y_loss"].values, alpha=alpha)

        out[ref_name] = res
        if verbose:
            distress = res["X_loss"] > res["VaR_X"]
            cv = ((res["Y_loss"] > res["CoVaR_Y"]) & distress).sum() / max(1, distress.sum())
            vd = distress.mean()
            print(f"  {ref_name}: VaR distress = {vd:.3f} (target {1-beta:.2f}) "
                  f"cond CoVaR violation = {cv:.3f} (target {1-alpha:.2f})  "
                  f"avg S_VaR = {res['S_VaR'].mean():.5f}  "
                  f"avg S_CoVaR = {res['S_CoVaR'].mean():.5f}")
    return out


if __name__ == "__main__":
    from output_covar import report
    results = run_all_pairs()
    # show_var=False: only the CoVaR_Y line is plotted (supervisor
    # feedback -- the DCC-GARCH VaR is not the quantity of interest).
    report(results, model_name="DCCGARCH", show_var=False)
