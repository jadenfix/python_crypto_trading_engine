"""CSV ingestion and strictly backward-looking daily-close features."""
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np


def load_prices(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 100:
        raise ValueError("At least 100 daily date,close rows are required")
    try:
        dates = [date.fromisoformat(row["date"]) for row in rows]
        prices = np.array([float(row["close"]) for row in rows])
    except (KeyError, ValueError) as exc:
        raise ValueError("CSV requires ISO date and numeric close columns") from exc
    if any(b - a != timedelta(days=1) for a, b in zip(dates, dates[1:])):
        raise ValueError("Dates must be unique, increasing, and consecutive daily observations")
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("Close prices must be finite and strictly positive")
    return np.array([d.isoformat() for d in dates]), prices


def demo_prices(path, days=730, seed=42):
    """Write explicitly synthetic prices; never fetch or trade real assets."""
    if days < 100:
        raise ValueError("Demo requires at least 100 days")
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0, 0.018, days - 1)
    returns = np.zeros(days - 1)
    for t in range(len(returns)):
        returns[t] = 0.0003 + innovations[t] + (0.15 * returns[t - 1] if t else 0)
    prices = 100 * np.exp(np.r_[0.0, np.cumsum(returns)])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "close"])
        for t, price in enumerate(prices):
            writer.writerow([(date(2020, 1, 1) + timedelta(days=t)).isoformat(), price])
    return path


def features(prices, lookback=20):
    """At close t predict return t→t+1, with no prices after t in X."""
    prices = np.asarray(prices, dtype=float)
    if lookback < 5 or len(prices) <= lookback + 1:
        raise ValueError("At least lookback+2 prices and lookback >= 5 required")
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("Close prices must be finite and strictly positive")
    returns = prices[1:] / prices[:-1] - 1
    indices = np.arange(lookback, len(prices) - 1)
    x = np.array([
        [returns[t - 1], returns[t - 5:t].mean(), returns[t - lookback:t].mean(),
         returns[t - lookback:t].std(), prices[t] / prices[t - lookback:t + 1].mean() - 1]
        for t in indices
    ])
    return x, returns[indices], indices
