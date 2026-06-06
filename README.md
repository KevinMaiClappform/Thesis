# Dynamic Modeling of Expected Shortfall using Machine Learning, Score-Driven Models, and Forecast Combination

Master's thesis code by **Kevin Mai** (2026).

This repository implements and compares dynamic models for forecasting
Value-at-Risk (VaR), Expected Shortfall (ES), and Conditional VaR (CoVaR)
on daily equity returns over the 2000-2025 sample.

```python
# univariate (VaR, ES) forecasters
from QRF             import rolling_qrf
from QGB             import rolling_qgb
from GAS             import rolling_gas
from GARCH           import rolling_garch
from FORECAST_COMBO  import combine_per_stock

# multivariate (VaR, CoVaR) forecasters
from COCAVIAR import rolling_cocaviar
from DCCGARCH import rolling_dcc_covar
from COVAR_ML import rolling_ml_covar
```

## Methods

### Univariate VaR / ES forecasters

| Module       | Model                                          | Key reference                          |
| ------------ | ---------------------------------------------- | -------------------------------------- |
| `QRF.py`     | Dynamic Quantile Random Forest                 | Meinshausen (2006)                     |
| `QGB.py`     | Dynamic Quantile Gradient Boosting (LightGBM)  | Bauer (2024); Velthoen et al. (2023)   |
| `GAS.py`     | One-factor GAS for (VaR, ES)                   | Patton, Ziegel & Chen (2019, sec 2.3)  |
| `GARCH.py`   | GARCH(1,1) with Student-t innovations          | Bollerslev (1986, 1987)                |

All baseline models share the rolling-window protocol (window = 500 days,
refit every 100 days) and base feature set (5 lagged returns + 2 rolling-
volatility proxies). The two ML models (QRF, QGB) use a two-pass dynamic
recursion in which their own lagged forecasts of (VaR, ES) enter the
feature set, mirroring the dynamic semiparametric models of
Patton-Ziegel-Chen (2019).

### Multivariate (VaR, CoVaR) forecasters

| Module           | Model                                          | Key reference                              |
| ---------------- | ---------------------------------------------- | ------------------------------------------ |
| `COCAVIAR.py`    | CoCAViaR (six specs)                           | Dimitriadis & Hoga (2026, Table 1)         |
| `DCCGARCH.py`    | DCC-GARCH(1,1) with variance targeting + MC CoVaR | Engle (2002); Dimitriadis & Hoga (2026, sec 5.2) |
| `COVAR_ML.py`    | Two-step ML CoVaR (QRF or QGB)                 | Meinshausen (2006); Bauer (2024); Adrian & Brunnermeier (2016) |

### Forecast combination

| Module              | Purpose                                                                |
| ------------------- | ---------------------------------------------------------------------- |
| `FORECAST_COMBO.py` | FZ-loss-weighted convex combination of the four univariate forecasters (QRF, QGB, GAS, GARCH). Weights are optimised on the simplex by minimising the in-sample FZ0 loss on a rolling 500-day window, refit every 100 days, with a softmax reparameterisation. Also produces `output/combo_weights_grid.png` -- the cross-sectional average combining weights over time, per confidence level (Taylor 2020, Figs 2-4). Reference: Taylor (2020). |

### Diagnostics

| Module              | Purpose                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| `diagnostics.ipynb` | Per-asset price grid (log-scale), returns grid, return distribution histograms with normal overlay, plus a summary table with mean / std / min / max / skew / excess kurtosis / Jarque-Bera / Ljung-Box(20) on r and r² (the canonical ARCH-effects diagnostic). Generates the Data-section figures and table in the thesis. |

### Cross-model evaluation

| Module                | Purpose                                                                |
| --------------------- | ---------------------------------------------------------------------- |
| `BACKTEST.py`         | Cross-model comparison of the univariate forecasters: Kupiec UC, Christoffersen CC, Engle-Manganelli DQ, Acerbi-Szekely Z2, and pairwise Diebold-Mariano on the FZ0 loss with Newey-West HAC. |
| `COVAR_BACKTEST.py`   | Cross-model comparison of the multivariate (VaR, CoVaR) forecasters: single-model coverage diagnostics + pairwise Diebold-Mariano on the lex-loss components S_VaR and S_CoVaR. |
| `extract_outputs.py`  | Generates the per-model `<model>_forecasts.csv`, `<model>_summary.csv` and `<model>_grid.png` files in `output/` from the cached backtest results. Avoids re-running the (slow) models when only the per-model output files are missing. Flags `--no-univariate` and `--no-multivariate` skip either side. |

Both backtests cache their model forecasts to a pickle so subsequent
invocations finish in seconds rather than re-running the full grid.

### Shared infrastructure

| Module           | Purpose                                                                    |
| ---------------- | -------------------------------------------------------------------------- |
| `features.py`    | `make_lag_features` (baseline daily features), `load_returns`, `DEFAULT_FILES` (the ten-asset universe). |
| `losses.py`      | `fz_loss` (Patton-Ziegel-Chen 2019 eq 6); `s_var`, `s_covar` (Dimitriadis-Hoga 2026 eq 5). |
| `output.py`      | `combine_results`, `summary_table`, `plot_forecast_grid`, `report` -- the helpers used by `extract_outputs.py` (and each univariate model's `__main__` block) to produce per-model CSVs and the 5x2 plot grid. |
| `output_covar.py`| Same helpers but for the bivariate (X_loss, Y_loss, VaR_X, CoVaR_Y) schema used by `COVAR.py` and `DCCGARCH.py`. |

## Repository structure

```
repo/
├── README.md                         (this file)
├── requirements.txt                  pip install -r requirements.txt
│
├── losses.py                         fz_loss, tick_loss, s_var, s_covar
├── features.py                       make_lag_features, load_returns
│
├── QRF.py                            dynamic Quantile Random Forest
├── QGB.py                            dynamic Quantile Gradient Boosting
├── GAS.py                            one-factor GAS for (VaR, ES)
├── GARCH.py                          GARCH(1,1)-t benchmark
├── COCAVIAR.py                       CoCAViaR (six specs)
├── DCCGARCH.py                       DCC-GARCH (variance targeting + MC CoVaR)
├── COVAR_ML.py                       two-step ML CoVaR (QRF or QGB)
│
├── FORECAST_COMBO.py                 FZ-loss-weighted convex combination
│
├── BACKTEST.py                       univariate cross-model backtest
├── COVAR_BACKTEST.py                 multivariate (VaR, CoVaR) backtest
├── extract_outputs.py                regenerate per-model CSVs/plots from cache
│
├── output.py                         shared CSV/plot helpers (univariate)
├── output_covar.py                   shared CSV/plot helpers (CoVaR)
│
├── diagnostics.ipynb                 Data-section figures + summary table generator
│
├── data/                             daily-return CSVs (CRSP DlyRet + prc_adj + tri)
│   ├── MICROSOFT.csv     ├── JPM.csv     ├── QQQ.csv
│   ├── ASML.csv          ├── NVIDIA.csv  ├── SPY.csv
│   ├── CITIGROUP.csv     ├── PEPSICO.csv └── DIAGEO.csv
│   └── GENERALDYNAMICS.csv
│
└── output/                           produced by extract_outputs.py or individual <model>.py
    ├── <model>_forecasts.csv,   <model>_summary.csv,   <model>_grid.png
    └── for model in {qrf, qgb, gas, garch, cocaviar, dccgarch, covar_qrf, covar_qgb}
```

## Installation

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Plus WRDS credentials in `~/.pgpass` (set up on first `wrds.Connection()` call)
if you need to (re-)download the daily CRSP data via the `DAILY.ipynb`
notebook in the project root.

## Workflow — full pipeline from raw data to thesis tables

```bash
# ====================================================================
# STEP 0 (optional): re-download daily data from WRDS
# ====================================================================
# Project-root notebook (not in repo/):
#   DAILY.ipynb           ~1 min     -> data_daily/<TICKER>.csv
# Then run the sync cell at the end of DAILY.ipynb to copy data_daily/
# into data/ (and repo/data/).

# ====================================================================
# STEP 1: diagnostics (Data-section figures)
# ====================================================================
cd repo
jupyter nbconvert --execute diagnostics.ipynb     # ~30 s
# Produces output/diagnostics_prices_grid.png,
#          output/diagnostics_returns_grid.png,
#          output/diagnostics_distributions_grid.png,
#          output/diagnostics_summary.csv  (with JB + LB tests)

# ====================================================================
# STEP 2: univariate baseline models (QRF, QGB, GAS, GARCH)
# ====================================================================
python BACKTEST.py                          # ~1.5 h
python extract_outputs.py --no-multivariate # ~5 s, writes per-model files
                                            # for QRF / QGB / GAS / GARCH

# ====================================================================
# STEP 3: forecast combination
# ====================================================================
python FORECAST_COMBO.py                    # ~5 min, requires Step 2 cache
                                            # writes combo_cache.pkl +
                                            # output/combo_weights_grid.png

# ====================================================================
# STEP 4: multivariate (CoVaR) models
# ====================================================================
python COVAR_BACKTEST.py                     # ~4 h (CoCAViaR + ML are the bottleneck)
python extract_outputs.py --no-univariate    # ~5 s
                                             # writes per-model files
                                             # for CoCAViaR, DCCGARCH,
                                             # COVAR_QRF and COVAR_QGB
```

Steps 2 and 4 are independent and can run in parallel terminals. Step 3
depends on the cache produced by Step 2. Total wall-time when Steps 2
and 4 run in parallel: ~3 hours (bound by `COVAR_BACKTEST.py`).

## Quick start — run one model interactively

```python
from features import load_returns
from QRF      import rolling_qrf
from losses   import fz_loss

df = load_returns("data/NVIDIA.csv")
res = rolling_qrf(df, window_size=500, refit_every=100)
res["FZ_5%"] = fz_loss(res["Actual"], res["VaR_5%"], res["ES_5%"], alpha=0.05)
print(res.head())
```

Or, to reproduce the per-model output files interactively:

```python
from QRF    import run_all_stocks
from output import report

results = run_all_stocks()
report(results, model_name="QRF", save_dir="output")
```

## Data

Daily returns for the ten-asset universe are in `data/`:

MICROSOFT (MSFT), ASML, CITIGROUP (C), GENERALDYNAMICS (GD),
JPM, NVIDIA (NVDA), PEPSICO (PEP), QQQ, SPY, DIAGEO (DEO).

The CSVs follow the WRDS / CRSP daily schema with columns `DlyCalDt`
(day-first dates), `DlyRet` (daily total return, already adjusted for
splits and dividends), and the price columns `prc_raw`, `prc_adj`, `tri`
written by `DAILY.ipynb`. The QQQ series additionally has its late-2004
to early-2011 segment merged in under the historical ticker `QQQQ` via
the PERMNO-based lookup in `DAILY.ipynb`; see the thesis text for details.

## Methodology

The full methodology, including the explicit FZ0 formula, the
CoCAViaR scoring functions, the BFGS-based GAS estimation, and the
FZ-loss-weighted forecast combination, is documented in the thesis text
(`../kladblok/scriptie.txt`).

Highlights:
- The six CoCAViaR specifications follow Dimitriadis & Hoga (2026, Table 1) verbatim and are estimated by the two-step lexicographic M-estimator of their equations (6)-(7).
- The DCC-GARCH benchmark uses *variance targeting* (Aielli 2013) so that the unconditional correlation matrix serves as the intercept, reducing the parameter count to the two DCC dynamics parameters (a, b).
- The one-factor GAS model is the parsimonious specification of Patton, Ziegel & Chen (2019, sec 2.3), with the reparameterisation `b = a - exp(d)` enforcing `ES <= VaR` by construction. Parameters are estimated by minimising the in-sample FZ0 loss via BFGS with a Nelder-Mead fallback.
- The forecast combination uses a softmax reparameterisation `w_i = exp(theta_i) / sum(exp(theta_j))` to enforce the simplex constraint during unconstrained optimisation, following Taylor (2020).
- The two-step ML CoVaR forecaster in `COVAR_ML.py` mirrors the linear quantile-regression CoVaR of Adrian & Brunnermeier (2016) but replaces both quantile regressions with non-parametric QRF / QGB estimators. Step 1 fits VaR(X) at level beta on the full training window; Step 2 fits CoVaR(Y|X) at level alpha on the distress sub-sample {X > VaR(X)}. Both steps use a two-pass dynamic recursion through their own lagged forecasts, identical in spirit to the univariate QRF / QGB.

## License

For thesis purposes only.
