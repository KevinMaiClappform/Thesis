"""
Feature engineering for the QRF / QGB notebooks and shared data loaders.

Two feature sets:

- ``make_lag_features``: the baseline daily feature set (5 lagged returns +
  2 rolling-window volatility proxies). Used by the daily-only model
  variants on the full 2000--2025 sample.

- ``make_lag_features_realized``: the baseline daily feature set PLUS
  lagged realized-variance, bipower-variation, realized-range and
  (optionally) realized covariance with SPY, merged from
  ``data_intraday/<ASSET>_realized.csv``. Used by the realized-augmented
  model variants, restricted to the 2020--2025 sub-sample where TAQ
  intraday data is available.
"""

import os
from pathlib import Path

import numpy as np
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

    # Drop rows with NaN only in the feature columns and the target, NOT in
    # unrelated CSV columns (e.g. `openprc`, which is missing for the early
    # part of some series such as DIAGEO and would otherwise silently delete
    # ~1000 valid observations).
    feature_cols = [f"lag_{lag}" for lag in range(1, n_lags + 1)] + \
                   ["rolling_vol_5", "rolling_vol_22", "DlyRet"]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df


def make_lag_features_realized(df, asset_name, n_lags=5,
                               intraday_dir="data_intraday",
                               include_rcov_spy=False):
    """Daily features PLUS lagged realized measures merged from intraday.

    Reads `<intraday_dir>/<asset_name>_realized.csv` and merges the realized
    variance (RV), bipower variation (BV), realized range (RR), and
    optionally the realized covariance with SPY (RCov_SPY), all in lagged
    form, into the feature set.

    Parameters
    ----------
    df : pandas.DataFrame
        Daily-return frame with `DlyCalDt` and `DlyRet`.
    asset_name : str
        Thesis short-name of the asset; used to locate the realized CSV.
    n_lags : int
        Number of lagged returns (default 5).
    intraday_dir : str or Path
        Folder containing `<asset_name>_realized.csv` (default ``data_intraday``).
    include_rcov_spy : bool
        Whether to include `lag_RCov_SPY` as an additional feature
        (default False; set True for multivariate / CoVaR work).

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with the base daily features AND
        `lag_RV_5min`, `lag_BV_5min`, `lag_RR_5min` (and optionally
        `lag_RCov_SPY_5min`). Rows with any NaN are dropped, so the
        effective sample is the intersection of the daily and intraday
        coverage (typically 2020--2025).
    """
    df = make_lag_features(df, n_lags=n_lags)

    path = Path(intraday_dir) / f"{asset_name}_realized.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No intraday file at {path}. Run INTRADAY_TAQ.ipynb first."
        )

    intra = pd.read_csv(path, sep=";")
    intra["DlyCalDt"] = pd.to_datetime(intra["DlyCalDt"])
    keep_cols = ["DlyCalDt", "RV_5min", "BV_5min", "RR_5min"]
    if include_rcov_spy and "RCov_SPY_5min" in intra.columns:
        keep_cols.append("RCov_SPY_5min")
    intra = intra[keep_cols]

    # Align DlyCalDt timezone / format. df["DlyCalDt"] is already datetime
    # by load_returns.
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    merged = df.merge(intra, on="DlyCalDt", how="left")

    # One-day lag of each realized column.
    for col in ["RV_5min", "BV_5min", "RR_5min"]:
        merged[f"lag_{col}"] = merged[col].shift(1)
    if include_rcov_spy:
        merged["lag_RCov_SPY_5min"] = merged["RCov_SPY_5min"].shift(1)

    # Drop the contemporaneous realized columns (we only use their lags as
    # features; keeping the contemporaneous values would leak future info).
    drop_cols = ["RV_5min", "BV_5min", "RR_5min"]
    if include_rcov_spy:
        drop_cols.append("RCov_SPY_5min")
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    # Drop rows with any NaN (typically pre-2020 days without intraday).
    merged = merged.dropna().reset_index(drop=True)
    return merged


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


def load_realized(asset_name, intraday_dir="data_intraday"):
    """Load realized-measures CSV produced by INTRADAY_TAQ.ipynb.

    Returns a DataFrame with columns DlyCalDt, RV_5min, BV_5min, RR_5min,
    RCov_SPY_5min, plus the (open, close, ret_oc_log, n_ticks, n_bins)
    diagnostic columns. Raises FileNotFoundError if the file does not exist.
    """
    path = Path(intraday_dir) / f"{asset_name}_realized.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No intraday file at {path}. Run INTRADAY_TAQ.ipynb first."
        )
    df = pd.read_csv(path, sep=";")
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
    return df.sort_values("DlyCalDt").reset_index(drop=True)


# Standard ten-asset universe used throughout the thesis.
DEFAULT_FILES = [
    "MICROSOFT.csv",
    "ASML.csv",
    "CITIGROUP.csv",
    "GENERALDYNAMICS.csv",
    "JPM.csv",
    "NVIDIA.csv",
    "PEPSICO.csv",
    "QQQ.csv",
    "SPY.csv",
    "DIAGEO.csv",
]
