"""Daily target-weight accounting, including entry, rebalance and exit fees."""
import numpy as np


def backtest(predictions, returns, initial_balance=10000.0, fee_bps=10.0,
             threshold=0.001, max_position=0.5, shorting=False):
    predictions, returns = np.asarray(predictions), np.asarray(returns)
    if predictions.ndim != 1 or predictions.shape != returns.shape or not len(returns):
        raise ValueError("Predictions and returns must be nonempty matching vectors")
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(returns)):
        raise ValueError("Inputs must be finite")
    if (not np.isfinite([initial_balance, fee_bps, threshold, max_position]).all()
            or initial_balance <= 0 or not 0 <= fee_bps < 10000
            or threshold < 0 or not 0 < max_position <= 1 or np.any(returns <= -1)):
        raise ValueError("Invalid balance, fees, threshold, exposure, or returns")
    positions = np.where(predictions > threshold, max_position, 0.0)
    if shorting:
        positions = np.where(predictions < -threshold, -max_position, positions)
    equity, fees, turnovers, net_returns = [], [], [], []
    cash = float(initial_balance)
    previous_weight = 0.0
    rate = fee_bps / 10000
    for i, (weight, daily_return) in enumerate(zip(positions, returns)):
        turnover = abs(weight - previous_weight)
        entry_fee = cash * rate * turnover
        gross_factor = 1 + weight * daily_return
        after_entry = cash - entry_fee
        marked = after_entry * gross_factor
        if marked <= 0:
            raise ValueError("Position became insolvent; reduce short exposure")
        drifted_weight = weight * (1 + daily_return) / gross_factor
        exit_fee = marked * rate * abs(drifted_weight) if i == len(returns) - 1 else 0.0
        new_cash = marked - exit_fee
        if new_cash <= 0:
            raise ValueError("Fees exhausted equity; reduce fee rate or exposure")
        fees.append(entry_fee + exit_fee)
        turnovers.append(turnover + (abs(drifted_weight) if exit_fee else 0.0))
        net_returns.append(new_cash / cash - 1)
        equity.append(new_cash)
        cash = new_cash
        previous_weight = drifted_weight
    equity, net_returns = np.array(equity), np.array(net_returns)
    full = np.r_[initial_balance, equity]
    drawdown = full / np.maximum.accumulate(full) - 1
    volatility = net_returns.std(ddof=1) if len(net_returns) > 1 else 0.0
    metrics = {
        "initial_balance": initial_balance, "final_balance": float(cash),
        "total_return": float(cash / initial_balance - 1),
        "annualized_return": float((cash / initial_balance) ** (365 / len(returns)) - 1),
        "annualized_volatility": float(volatility * np.sqrt(365)),
        "sharpe_ratio": float(net_returns.mean() / volatility * np.sqrt(365)) if volatility else 0.0,
        "maximum_drawdown": float(drawdown.min()), "total_fees": float(sum(fees)),
        "active_days": int(np.count_nonzero(positions)), "evaluation_days": len(returns),
    }
    return {"equity": equity, "positions": positions, "fees": fees,
            "turnover": turnovers, "net_returns": net_returns, "metrics": metrics}
