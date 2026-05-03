"""
Loss / scoring functions for tail-risk forecasting.

References:
- Patton, A.J., Ziegel, J.F., Chen, R. (2019). Dynamic semiparametric models
  for expected shortfall (and value-at-risk). Journal of Econometrics, 211(2),
  388--413. Eq. (6) for the FZ0 loss.
- Dimitriadis, T., Hoga, Y. (2026). Dynamic CoVaR Modeling and Estimation.
  Journal of Business & Economic Statistics. Eq. (5) for the (VaR, CoVaR)
  lexicographic scoring functions.
"""

import numpy as np


def fz_loss(y, var, es, alpha=0.05):
    """Fissler-Ziegel FZ0 loss for the joint (VaR, ES) functional.

    L_FZ0(y, v, e; alpha) = -1/(alpha*e) * 1{y<=v} * (v - y)
                            + v/e + log(-e) - 1

    Parameters
    ----------
    y : array-like
        Realised returns.
    var : array-like
        VaR forecasts at level alpha (must be negative for losses).
    es : array-like
        ES forecasts at level alpha (must satisfy es <= var, both negative).
    alpha : float
        Tail probability, e.g. 0.05 for 5% VaR/ES.

    Returns
    -------
    np.ndarray
        Per-observation FZ0 loss values.
    """
    y = np.asarray(y)
    var = np.asarray(var)
    es = np.asarray(es)

    es = np.where(es >= -1e-8, -1e-8, es)
    indicator = (y <= var).astype(float)

    return -indicator * (var - y) / (alpha * es) + var / es + np.log(-es) - 1


def tick_loss(y, q, alpha):
    """Standard tick (pinball) loss for a single quantile.

    L_alpha(y, q) = (1{y<=q} - alpha) * (q - y)

    Strictly consistent for the alpha-quantile and used as the VaR component
    of the Dimitriadis & Hoga (2026) lexicographic (VaR, CoVaR) scoring rule.
    """
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    return ((y <= q).astype(float) - alpha) * (q - y)


def s_var(v, x, beta):
    """VaR component of the Dimitriadis & Hoga (2026, eq. 5) score for upper-tail
    losses. Identical to tick_loss but kept under the paper's notation.

        S_VaR(v, x; beta) = (1{x <= v} - beta) * (v - x)
    """
    v = np.asarray(v, dtype=float)
    x = np.asarray(x, dtype=float)
    return ((x <= v).astype(float) - beta) * (v - x)


def s_covar(v, c, x, y, alpha):
    """CoVaR component of the Dimitriadis & Hoga (2026, eq. 5) score:

        S_CoVaR((v, c), (x, y); alpha) = 1{x > v} * (1{y <= c} - alpha) * (c - y)

    Tick loss in y, restricted to days on which the reference asset x exceeds
    its VaR (the distress event).
    """
    v = np.asarray(v, dtype=float)
    c = np.asarray(c, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (x > v).astype(float) * ((y <= c).astype(float) - alpha) * (c - y)
