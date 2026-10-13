# Crypto Forecasting and Backtesting Engine

An offline daily-close research pipeline: load prices, build causal features,
select a forecasting model on chronological training folds, and evaluate a
long-only or opt-in short strategy on an untouched final 30% of observations.

## Run the complete demo

```bash
git clone https://github.com/jadenfix/python_crypto_trading_engine.git
cd python_crypto_trading_engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --demo
```

The demo uses **seeded synthetic daily prices**, not historical ADA/BTC data.
It needs no API key, network connection after installation, or broker account.
`python run_ada_backtest.py` runs the same synthetic example for compatibility.

Open `results/backtest_plot.png` to compare the strategy against buy and hold
with the same maximum exposure and fee rate. Inspect `performance.json` and
`backtest_results.csv` for predictions, settlement dates, turnover, dollar fees,
daily net returns, and equity. The model is not guaranteed to beat the baseline.

## Use your own daily prices

Supply at least 100 consecutive daily closes in strictly increasing order:

```csv
date,close
2024-01-01,42000.00
2024-01-02,42500.00
```

```bash
python -m src.main --csv daily_prices.csv --model rf --output results-btc \
  --fee-bps 10 --max-position 0.5 --signal-threshold 0.001
```

Models: `ridge` (scaled Ridge regression) and `rf` (Random Forest). Add
`--enable-shorting` to permit negative exposure; defaults are long-only.
`--initial-balance` defaults to 10000. Costs and exposure are experiment inputs,
not estimates of a particular exchange's costs.

## Evaluation and accounting

- Features at close t use only prices through t: lagged return, 5/20-day
  mean return, 20-day volatility, and distance from the trailing moving average.
- Targets are the following daily return. The final 30% of samples are held
  out. One sample is purged at the boundary; three training-only expanding
  cross-validation folds also use a one-sample gap. Scaling is fitted inside
  each training fold.
- Signals choose a fixed target fraction of equity. Fees apply to entry,
  changes from the previous position's drifted weight, and final liquidation.
  Strategy and baseline share the same costs and exposure limit.
- Daily annualization uses 365 periods. Maximum drawdown includes starting
  equity. Sharpe uses zero risk-free return and daily sample volatility.

This is a daily-close simulation with execution at the decision close. It does
not model spread, slippage, funding, borrow fees, intraday liquidity, or fills.
The outputs describe this simulation; they are not executable trade orders.

## Tests

```bash
python -m unittest discover -s tests -v
python -m src.main --demo --model rf --output results-rf
```

Tests exercise chronological feature isolation, next-day target alignment,
invalid input, entry/exit costs, drift rebalance, drawdown, short opt-in, and the
complete deterministic CSV → model → backtest → JSON/CSV/PNG workflow.

The input CSV is preserved. If it is also a report destination
(`backtest_results.csv`, `performance.json`, or `backtest_plot.png`), including
an alias to the same file, the pipeline rejects the run before training or
writing reports. Choose another `--output` directory. A noncolliding input
such as `prices.csv` can remain in the report directory.

## Source

| File | Purpose |
| --- | --- |
| `src/datasets.py` | Daily CSV validation, synthetic fixture, causal features |
| `src/models.py` | Purged chronological model selection |
| `src/backtest.py` | Position, fee, equity and performance accounting |
| `src/main.py` | CLI and complete pipeline |
| `tests/test_pipeline.py` | Accounting and end-to-end regression checks |
| `tests/test_input_preservation.py` | Source-file collision and CLI preservation checks |

The repository previously contained only a README describing unavailable
scripts. This implementation supports the documented commands above; optional
XGBoost models, multi-asset portfolios, and live execution are not implemented.
