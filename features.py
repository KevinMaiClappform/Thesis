"""
Feature engineering for the QRF / QGB notebooks.

The same feature set is used by both ML models so the comparison is on the
function class, not the inputs.
"""

import pandas as pd


def make_lag_features(df, n_lags=5):
    """Add lagged returns and rolling-volatility proxies to a returns frame.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a column `DlyRet`.
    n_lags : int
        Number of lagged returns to construct (default 5).

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with columns `lag_1`, ..., `lag_{n_lags}`,
        `rolling_vol_5`, `rolling_vol_22`. Rows with any NaN dropped.
    """
    df = df.copy()

    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["DlyRet"].shift(lag)

    df["rolling_vol_5"] = df["DlyRet"].rolling(5).std()
    df["rolling_vol_22"] = df["DlyRet"].rolling(22).std()

    df = df.dropna().reset_index(drop=True)

    return df


def load_returns(path, sep=";"):
    """Load a single CSV in the WRDS / CRSP daily-return schema and return
    a clean, date-sorted DataFrame with parsed `DlyCalDt` and numeric `DlyRet`.

    Used by every model notebook in the repo.
    """
    df = pd.read_csv(path, sep=sep)
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"], dayfirst=True)
    df["DlyRet"] = pd.to_numeric(df["DlyRet"], errors="coerce")
    df = df.dropna(subset=["DlyCalDt", "DlyRet"])
    return df.sort_values("DlyCalDt").reset_index(drop=True)


# Standard ten-asset universe used throughout the thesis.
DEFAULT_FILES = [
    "ALPHABET.csv",
    "ASML.csv",
    "CITIGROUP.csv",
    "GENERALDYNAMICS.csv",
    "JPM.csv",
    "NVIDIA.csv",
    "PEPSICO.csv",
    "QQQ.csv",
    "SPY.csv",
    "UNILEVER.csv",
]
