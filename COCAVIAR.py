"""
CoCAViaR models for the joint (VaR, CoVaR) of Dimitriadis & Hoga (2026).

Implements all six specifications of D&H Table 1: SAV-diag, SAV-fullA,
SAV-full, AS-pos, AS-signs, AS-mixed. Estimation is by the two-step
M-estimator of D&H eq. (6)-(7) using the lexicographic R^2-valued scoring
function of D&H eq. (5).

Public API
----------
- SPECS: dict mapping spec name to filter and parameter-count metadata.
- rolling_cocaviar(spec_name, X_loss, Y_loss, dates, ...): one-step OOS forecasts.
- evaluate_pair_all_specs(X_loss, Y_loss, dates, ...): run all six specs.
- lex_select(spec_results): pick the lex-best spec from such a result dict.
- run_all_pairs(reference_files, system_file, data_dir): convenience driver.
"""

import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

from scipy.optimize import minimize

from features import load_returns
from losses import s_var, s_covar


# ------------------------------------------------------------------------- #
# Filters: one pair (filter_v_*, filter_c_*) per spec.                       #
# Reparameterisation throughout: omega = exp(omega_raw) > 0,                 #
# A = exp(A_raw) >= 0, B = sigmoid(B_raw) in (0, 1).                         #
# Initialisation: v_1 = omega_v, c_1 = omega_c, per D&H sec. 5.              #
# ------------------------------------------------------------------------- #


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def filter_v_sav_diag(raw, X, Y):
    omega, A, B = np.exp(raw[0]), np.exp(raw[1]), _sig(raw[2])
    n = len(X); v = np.empty(n); v[0] = omega
    for t in range(1, n):
        v[t] = omega + A * abs(X[t - 1]) + B * v[t - 1]
    return v


def filter_c_sav_diag(raw, X, Y, v_path=None):
    omega, A, B = np.exp(raw[0]), np.exp(raw[1]), _sig(raw[2])
    n = len(Y); c = np.empty(n); c[0] = omega
    for t in range(1, n):
        c[t] = omega + A * abs(Y[t - 1]) + B * c[t - 1]
    return c


def filter_v_sav_fullA(raw, X, Y):
    omega = np.exp(raw[0]); A1, A2 = np.exp(raw[1]), np.exp(raw[2]); B = _sig(raw[3])
    n = len(X); v = np.empty(n); v[0] = omega
    for t in range(1, n):
        v[t] = omega + A1 * abs(X[t - 1]) + A2 * abs(Y[t - 1]) + B * v[t - 1]
    return v


def filter_c_sav_fullA(raw, X, Y, v_path=None):
    omega = np.exp(raw[0]); A1, A2 = np.exp(raw[1]), np.exp(raw[2]); B = _sig(raw[3])
    n = len(Y); c = np.empty(n); c[0] = omega
    for t in range(1, n):
        c[t] = omega + A1 * abs(X[t - 1]) + A2 * abs(Y[t - 1]) + B * c[t - 1]
    return c


def filter_v_sav_full(raw, X, Y):
    return filter_v_sav_fullA(raw, X, Y)


def filter_c_sav_full(raw, X, Y, v_path):
    """c_t depends on v_{t-1}; requires v_path (the fitted Step-1 path)."""
    omega = np.exp(raw[0]); A1, A2 = np.exp(raw[1]), np.exp(raw[2])
    Bv, Bc = _sig(raw[3]), _sig(raw[4])
    n = len(Y); c = np.empty(n); c[0] = omega
    for t in range(1, n):
        c[t] = omega + A1 * abs(X[t - 1]) + A2 * abs(Y[t - 1]) + Bv * v_path[t - 1] + Bc * c[t - 1]
    return c


def filter_v_as_pos(raw, X, Y):
    omega = np.exp(raw[0]); A1, A2 = np.exp(raw[1]), np.exp(raw[2]); B = _sig(raw[3])
    n = len(X); v = np.empty(n); v[0] = omega
    for t in range(1, n):
        v[t] = omega + A1 * max(X[t - 1], 0.0) + A2 * max(Y[t - 1], 0.0) + B * v[t - 1]
    return v


def filter_c_as_pos(raw, X, Y, v_path=None):
    omega = np.exp(raw[0]); A1, A2 = np.exp(raw[1]), np.exp(raw[2]); B = _sig(raw[3])
    n = len(Y); c = np.empty(n); c[0] = omega
    for t in range(1, n):
        c[t] = omega + A1 * max(X[t - 1], 0.0) + A2 * max(Y[t - 1], 0.0) + B * c[t - 1]
    return c


def filter_v_as_signs(raw, X, Y):
    omega = np.exp(raw[0])
    A1, A2 = np.exp(raw[1]), np.exp(raw[2])
    A3, A4 = np.exp(raw[3]), np.exp(raw[4])
    B = _sig(raw[5])
    n = len(X); v = np.empty(n); v[0] = omega
    for t in range(1, n):
        xp = max(X[t - 1], 0.0); xn = -min(X[t - 1], 0.0)
        yp = max(Y[t - 1], 0.0); yn = -min(Y[t - 1], 0.0)
        v[t] = omega + A1 * xp + A2 * xn + A3 * yp + A4 * yn + B * v[t - 1]
    return v


def filter_c_as_signs(raw, X, Y, v_path=None):
    omega = np.exp(raw[0])
    A1, A2 = np.exp(raw[1]), np.exp(raw[2])
    A3, A4 = np.exp(raw[3]), np.exp(raw[4])
    B = _sig(raw[5])
    n = len(Y); c = np.empty(n); c[0] = omega
    for t in range(1, n):
        xp = max(X[t - 1], 0.0); xn = -min(X[t - 1], 0.0)
        yp = max(Y[t - 1], 0.0); yn = -min(Y[t - 1], 0.0)
        c[t] = omega + A1 * xp + A2 * xn + A3 * yp + A4 * yn + B * c[t - 1]
    return c


def filter_v_as_mixed(raw, X, Y):
    return filter_v_sav_fullA(raw, X, Y)


def filter_c_as_mixed(raw, X, Y, v_path=None):
    """c uses positive and negative components of X only (no Y)."""
    omega = np.exp(raw[0]); A1, A2 = np.exp(raw[1]), np.exp(raw[2]); B = _sig(raw[3])
    n = len(Y); c = np.empty(n); c[0] = omega
    for t in range(1, n):
        xp = max(X[t - 1], 0.0); xn = -min(X[t - 1], 0.0)
        c[t] = omega + A1 * xp + A2 * xn + B * c[t - 1]
    return c


SPECS = {
    "SAV-diag":  {"f_v": filter_v_sav_diag,  "f_c": filter_c_sav_diag,  "n_v": 3, "n_c": 3, "c_uses_v": False},
    "SAV-fullA": {"f_v": filter_v_sav_fullA, "f_c": filter_c_sav_fullA, "n_v": 4, "n_c": 4, "c_uses_v": False},
    "SAV-full":  {"f_v": filter_v_sav_full,  "f_c": filter_c_sav_full,  "n_v": 4, "n_c": 5, "c_uses_v": True},
    "AS-pos":    {"f_v": filter_v_as_pos,    "f_c": filter_c_as_pos,    "n_v": 4, "n_c": 4, "c_uses_v": False},
    "AS-signs":  {"f_v": filter_v_as_signs,  "f_c": filter_c_as_signs,  "n_v": 6, "n_c": 6, "c_uses_v": False},
    "AS-mixed":  {"f_v": filter_v_as_mixed,  "f_c": filter_c_as_mixed,  "n_v": 4, "n_c": 4, "c_uses_v": False},
}


# ------------------------------------------------------------------------- #
# Two-step M-estimator (Dimitriadis & Hoga 2026, eq. 6-7).                   #
# ------------------------------------------------------------------------- #


def _start_v(spec_name, X_loss):
    omega_raw = np.log(np.std(X_loss) + 1e-8)
    n = SPECS[spec_name]["n_v"]
    A_raw = np.log(0.1)
    B_raw = 3.0
    if spec_name == "SAV-diag":
        return np.array([omega_raw, A_raw, B_raw])
    if n == 4:
        return np.array([omega_raw, A_raw, A_raw, B_raw])
    if spec_name == "AS-signs":
        return np.array([omega_raw, A_raw, A_raw, A_raw, A_raw, B_raw])
    raise ValueError(spec_name)


def _start_c(spec_name, Y_loss):
    omega_raw = np.log(np.std(Y_loss) + 1e-8)
    A_raw = np.log(0.1)
    B_raw = 3.0
    if spec_name == "SAV-diag":
        return np.array([omega_raw, A_raw, B_raw])
    if spec_name in ("SAV-fullA", "AS-pos", "AS-mixed"):
        return np.array([omega_raw, A_raw, A_raw, B_raw])
    if spec_name == "SAV-full":
        return np.array([omega_raw, A_raw, A_raw, 1.0, B_raw])
    if spec_name == "AS-signs":
        return np.array([omega_raw, A_raw, A_raw, A_raw, A_raw, B_raw])
    raise ValueError(spec_name)


def _fit_v(spec_name, X_loss, Y_loss, beta, warm_start=None):
    f_v = SPECS[spec_name]["f_v"]

    def obj(theta):
        v = f_v(theta, X_loss, Y_loss)
        if not np.all(np.isfinite(v)):
            return 1e10
        return float(np.mean(s_var(v, X_loss, beta)))

    starts = [warm_start] if warm_start is not None else []
    starts.append(_start_v(spec_name, X_loss))
    best = None
    for x0 in starts:
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-6, "disp": False})
        if best is None or res.fun < best.fun:
            best = res
    return best


def _fit_c(spec_name, v_hat, X_loss, Y_loss, alpha, warm_start=None):
    f_c = SPECS[spec_name]["f_c"]
    c_uses_v = SPECS[spec_name]["c_uses_v"]

    def obj(theta):
        c = f_c(theta, X_loss, Y_loss, v_hat) if c_uses_v else f_c(theta, X_loss, Y_loss)
        if not np.all(np.isfinite(c)):
            return 1e10
        return float(np.mean(s_covar(v_hat, c, X_loss, Y_loss, alpha)))

    starts = [warm_start] if warm_start is not None else []
    starts.append(_start_c(spec_name, Y_loss))
    best = None
    for x0 in starts:
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": 3000, "xatol": 1e-6, "disp": False})
        if best is None or res.fun < best.fun:
            best = res
    return best


def rolling_cocaviar(spec_name, X_loss, Y_loss, dates,
                     window_size=500, refit_every=100,
                     beta=0.95, alpha=0.95):
    """Out-of-sample CoCAViaR forecasts of (VaR, CoVaR) for the given spec."""
    f_v = SPECS[spec_name]["f_v"]
    f_c = SPECS[spec_name]["f_c"]
    c_uses_v = SPECS[spec_name]["c_uses_v"]

    X = np.asarray(X_loss, dtype=float)
    Y = np.asarray(Y_loss, dtype=float)
    n = len(X)
    assert len(Y) == n

    var_fc = np.full(n, np.nan)
    covar_fc = np.full(n, np.nan)

    theta_v = theta_c = None

    for refit_t in range(window_size, n, refit_every):
        X_tr = X[refit_t - window_size:refit_t]
        Y_tr = Y[refit_t - window_size:refit_t]

        fit_v = _fit_v(spec_name, X_tr, Y_tr, beta=beta, warm_start=theta_v)
        theta_v = fit_v.x
        v_tr = f_v(theta_v, X_tr, Y_tr)

        fit_c = _fit_c(spec_name, v_tr, X_tr, Y_tr, alpha=alpha, warm_start=theta_c)
        theta_c = fit_c.x

        next_refit = min(refit_t + refit_every, n)
        v_full = f_v(theta_v, X[:next_refit], Y[:next_refit])
        if c_uses_v:
            c_full = f_c(theta_c, X[:next_refit], Y[:next_refit], v_full)
        else:
            c_full = f_c(theta_c, X[:next_refit], Y[:next_refit])

        for t in range(refit_t, next_refit):
            var_fc[t] = v_full[t]
            covar_fc[t] = c_full[t]

    return var_fc, covar_fc, theta_v, theta_c


def evaluate_pair_all_specs(X_loss, Y_loss, dates, beta=0.95, alpha=0.95):
    """Run all six SAV / AS specs on the given (X, Y) pair."""
    out = {}
    for spec_name in SPECS:
        var_fc, covar_fc, theta_v, theta_c = rolling_cocaviar(
            spec_name, X_loss, Y_loss, dates,
            window_size=500, refit_every=100,
            beta=beta, alpha=alpha,
        )
        df = pd.DataFrame({
            "Date": dates,
            "X_loss": X_loss,
            "Y_loss": Y_loss,
            "VaR_X": var_fc,
            "CoVaR_Y": covar_fc,
        }).dropna().reset_index(drop=True)

        avg_var = float(np.mean(s_var(df["VaR_X"].values, df["X_loss"].values, beta)))
        avg_covar = float(np.mean(s_covar(df["VaR_X"].values, df["CoVaR_Y"].values,
                                          df["X_loss"].values, df["Y_loss"].values, alpha)))
        out[spec_name] = {"df": df, "avg_var": avg_var, "avg_covar": avg_covar,
                          "theta_v": theta_v, "theta_c": theta_c}
    return out


def lex_select(spec_results):
    """Lexicographic-best spec: lowest VaR-score first, then lowest CoVaR-score."""
    var_scores = {s: r["avg_var"] for s, r in spec_results.items()}
    min_var = min(var_scores.values())
    eps = 1e-6 * abs(min_var)
    candidates = [s for s, v in var_scores.items() if v <= min_var + eps]
    return min(candidates, key=lambda s: spec_results[s]["avg_covar"])


def run_all_pairs(reference_files=None, system_file="SPY.csv",
                  data_dir="data", verbose=True):
    """Run all six specs for each (reference, system) pair. Returns dicts
    keyed by reference name with all-spec results, lex-best name, and the
    lex-best DataFrame."""
    if reference_files is None:
        reference_files = [
            "MICROSOFT.csv", "ASML.csv", "CITIGROUP.csv", "GENERALDYNAMICS.csv",
            "JPM.csv", "NVIDIA.csv", "PEPSICO.csv", "QQQ.csv", "DIAGEO.csv",
        ]

    sys_path = os.path.join(data_dir, system_file) if data_dir else system_file
    sys_df = load_returns(sys_path).rename(columns={"DlyRet": "Ret_Y"})

    cocaviar_all = {}
    cocaviar_best = {}
    cocaviar_results = {}

    for ref_file in reference_files:
        ref_name = ref_file.replace(".csv", "")
        if verbose:
            print(f"Running CoCAViaR (all 6 specs) for {ref_name} | {system_file.replace('.csv', '')} ...")

        ref_path = os.path.join(data_dir, ref_file) if data_dir else ref_file
        ref_df = load_returns(ref_path).rename(columns={"DlyRet": "Ret_X"})
        pair = ref_df.merge(sys_df, on="DlyCalDt", how="inner").reset_index(drop=True)

        X_loss = -pair["Ret_X"].values
        Y_loss = -pair["Ret_Y"].values
        dates = pair["DlyCalDt"].values

        spec_results = evaluate_pair_all_specs(X_loss, Y_loss, dates)
        best_spec = lex_select(spec_results)

        cocaviar_all[ref_name] = spec_results
        cocaviar_best[ref_name] = best_spec
        cocaviar_results[ref_name] = spec_results[best_spec]["df"]

        if verbose:
            best_var = spec_results[best_spec]["avg_var"]
            best_covar = spec_results[best_spec]["avg_covar"]
            print(f"  {ref_name}: best = {best_spec:10s}  "
                  f"VaR-score = {best_var:.5f}  CoVaR-score = {best_covar:.5f}")

    return cocaviar_all, cocaviar_best, cocaviar_results


if __name__ == "__main__":
    from output_covar import report
    cocaviar_all, cocaviar_best, cocaviar_results = run_all_pairs()
    # show_var=False: plot only the CoVaR line. The VaR is the conditioning
    # device in the two-step (VaR, CoVaR) estimator, not the focal object.
    report(cocaviar_results, model_name="CoCAViaR", best_specs=cocaviar_best,
           show_var=False)
