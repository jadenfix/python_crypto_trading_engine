# Crypto Forecasting and Backtesting Engine

A robust backtesting system for cryptocurrency trading strategies based on machine learning models with time series cross-validation.

## Overview

This project provides a complete pipeline for:

1. **Data Processing**: Standardize and prepare cryptocurrency data for model training and simulation
2. **Model Training**: Train machine learning models with hyperparameter optimization and time series cross-validation
3. **Backtesting**: Simulate trading strategies with realistic assumptions and track performance

## Features

- **Data Standardization**: Ensures consistent datetime formats between training and simulation data
- **Feature Engineering**: Automatically generates technical indicators and other useful features
- **Time Series Cross-Validation**: Prevents data leakage when evaluating model performance
- **Hyperparameter Optimization**: Finds optimal model parameters with grid search
- **Realistic Backtesting**: Includes transaction costs, position sizing, and risk management
- **Performance Metrics**: Calculates comprehensive trading metrics including Sharpe ratio, win rate, drawdown, etc.
- **Visualization**: Generates plots of equity curves, drawdowns, and trade entry/exit points

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd crypto_forecasting

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

To run a complete example with ADA cryptocurrency:

```bash
python run_ada_backtest.py
```

This will:
1. Process the ADA data
2. Train Random Forest, XGBoost, and Ridge regression models
3. Backtest the models on the simulation data
4. Save results in the `results` directory

## Usage

### Data Processing

```bash
python -m src.data_processor
```

This will process all available cryptocurrencies and prepare the data for model training and backtesting.

### Training and Backtesting

```bash
python -m src.main --coins ada eth btc solana --model-types rf xgb ridge --n-rows 1000
```

Command-line arguments:
- `--coins`: List of coins to process and backtest
- `--model-types`: List of model types to train (rf, xgb, ridge, lasso, bayesian, gbm)
- `--n-rows`: Number of rows for reduced dataset
- `--skip-processing`: Skip data processing step
- `--skip-training`: Skip model training step
- `--skip-backtesting`: Skip backtesting step
- `--initial-balance`: Initial balance for backtesting (default: 10000.0)
- `--enable-shorting`: Enable short positions in backtesting
- `--signal-threshold`: Signal threshold for entering positions (default: 0.0001)

### Data Structure

- `/data/train/`: Contains training data files
- `/data/train/simulation/`: Contains simulation data files
- `/data/processed/`: Contains processed data ready for model training
- `/models/`: Saved trained models and performance metrics
- `/results/`: Backtest results, plots, and performance metrics

## Project Structure

- `src/data_processor.py`: Data preprocessing and standardization
- `src/datasets.py`: Dataset loading and feature engineering
- `src/models.py`: Model training with time series cross-validation
- `src/backtest.py`: Backtesting engine for trading strategies
- `src/main.py`: Main script to run the full pipeline
- `run_ada_backtest.py`: Example script for ADA cryptocurrency

## Customization

You can customize the backtesting parameters in `src/config.py`:

- `TRANSACTION_COSTS`: Transaction costs for each cryptocurrency
- `INITIAL_BALANCE`: Initial account balance for backtesting
- `SIGNAL_THRESHOLD`: Minimum signal threshold to enter a position
- `ENABLE_SHORTING`: Whether to allow short positions
- `RISK_PER_TRADE`: Risk per trade as a fraction of account balance
- `MAX_POSITION_SIZE`: Maximum position size as a fraction of account balance

## Example Results

After running the backtester, you'll find results in the `results/` directory:

- `<coin>/<model_type>_backtest_results.csv`: Detailed backtest results
- `<coin>/<model_type>_trades.csv`: Individual trade details
- `<coin>/<model_type>_performance.json`: Performance metrics
- `<coin>/<model_type>_backtest_plot.png`: Visualization of the backtest

## Transaction Costs
Transaction costs are modeled specifically for each coin:
- BTC: 5-10 bps
- ETH: 6-12 bps
- SOL: 12-20 bps
- ADA: 14-25 bps

## Running Backtests
```python
from src.runner import run_backtest

# Run backtest with default models (Random Forest, XGBoost, Logistic Regression)
results = run_backtest()

# Or specify custom model types
results = run_backtest(model_types=['rf', 'xgb'])
```

## Output
The backtesting engine generates two key output files in the `results/` directory:
1. `backtest_results.csv`: Performance summary across strategies and coins
2. `detailed_backtest_results.csv`: Comprehensive trade-level results

### Performance Metrics
- Total Return
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
[Specify your license here]
# python_crypto_trading_engine
