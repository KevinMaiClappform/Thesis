# Dynamic Modeling of Expected Shortfall using Machine Learning, Score-Driven Models, and Tail Risk Methods

Master's thesis code by **Kevin Mai** (March 2026).

This repository implements and compares dynamic models for forecasting
Value-at-Risk (VaR), Expected Shortfall (ES), and Conditional VaR (CoVaR)
on daily equity returns. Each model family lives in a single, importable
Python module so they can be used directly from a backtesting script:

```python
from QRF   import rolling_qrf
from QGB   import rolling_qgb
from GAS   import rolling_gas
from GARCH import rolling_garch
from COVAR import rolling_cocaviar
```

## Methods

| Module     | Model                                     | Key reference                         |
| ---------- | ----------------------------------------- | ------------------------------------- |
| `QRF.py`   | Dynamic Quantile Random Forest            | Meinshausen (2006)                    |
| `QGB.py`   | Dynamic Quantile Gradient Boosting (LGBM) | Bauer (2024); Velthoen et al. (2023)  |
| `GAS.py`   | One-factor GAS for (VaR, ES)              | Patton, Ziegel & Chen (2019, sec 2.3) |
| `GARCH.py` | GARCH(1,1) with Student-t innovations     | Bollerslev (1986, 1987)               |
| `COVAR.py` | CoCAViaR (six specs) for (VaR, CoVaR)     | Dimitriadis & Hoga (2026)             |

All four univariate models use the same rolling-window protocol (training
window of 500 days, refit every 100 days), the same 7-feature base set
(5 lagged returns + 2 rolling-volatility proxies), and the same
Fissler-Ziegel FZ0 loss for evaluation, ensuring an apples-to-apples
comparison. The QRF and QGB additionally include a two-pass dynamic
recursion in which the lagged own forecasts of (VaR, ES) enter the
feature set, mirroring the dynamic semiparametric models of
Patton-Ziegel-Chen (2019).

## Repository structure

```
repo/
├── README.md           # this file
├── requirements.txt    # pip install -r requirements.txt
├── .gitignore
├── losses.py           # fz_loss, tick_loss, s_var, s_covar (shared)
├── features.py         # make_lag_features, load_returns, DEFAULT_FILES
├── QRF.py              # Quantile Random Forest
├── QGB.py              # Quantile Gradient Boosting
├── GAS.py              # one-factor Generalized Autoregressive Score
├── GARCH.py            # GARCH(1,1)-t benchmark
├── COVAR.py            # CoCAViaR for (VaR, CoVaR)
└── data/               # daily-return CSVs (semicolon-separated)
    ├── ALPHABET.csv
    ├── ASML.csv
    ├── ...
    └── UNILEVER.csv
```

## Installation

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## Usage

### Run one model on all ten assets

```bash
python QRF.py        # or QGB.py / GAS.py / GARCH.py / COVAR.py
```

Each module's `__main__` block calls `run_all_stocks()` (or
`run_all_pairs()` for COVAR) with the default ten-asset universe and
prints per-stock violation rates and FZ0 losses to stdout.

### Use a single model in your own code

```python
from features import load_returns
from QRF      import rolling_qrf
from losses   import fz_loss

df  = load_returns("data/NVIDIA.csv")
res = rolling_qrf(df, window_size=500, refit_every=100)
res["FZ_5%"] = fz_loss(res["Actual"], res["VaR_5%"], res["ES_5%"], alpha=0.05)
print(res.head())
```

### Compare models with a custom backtest

```python
# planned: BACKTEST.py
from QRF   import run_all_stocks as run_qrf
from QGB   import run_all_stocks as run_qgb
from GAS   import run_all_stocks as run_gas
from GARCH import run_all_stocks as run_garch

qrf, qgb, gas, garch = run_qrf(), run_qgb(), run_gas(), run_garch()
# ...Diebold-Mariano on FZ0 loss differentials,
# Kupiec / Christoffersen / DQ tests on violation sequences, etc.
```

## Data

Daily returns for the ten-asset universe are in `data/`:

ALPHABET (GOOGL), ASML, CITIGROUP (C), GENERALDYNAMICS (GD),
JPM, NVIDIA (NVDA), PEPSICO (PEP), QQQ, SPY, UNILEVER (UL).

CSVs follow the WRDS / CRSP daily schema with columns including
`DlyCalDt` (day-first dates) and `DlyRet` (decimal returns).

## Methodology

The full methodology, including the explicit FZ0 formula, the
CoCAViaR scoring functions, and the choice of features, is documented
in the thesis text (`scriptie.tex`). The six CoCAViaR specifications
(SAV-diag, SAV-fullA, SAV-full, AS-pos, AS-signs, AS-mixed) follow
Dimitriadis & Hoga (2026, Table 1) verbatim and are estimated by the
two-step lexicographic M-estimator of their equations (6)-(7).

## License

For thesis purposes only.
