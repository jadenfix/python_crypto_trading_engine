"""Run an offline research pipeline; no broker integration or live orders."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .backtest import backtest
from .datasets import demo_prices, features, load_prices
from .models import train


def run(csv_path, output, model_type="ridge", initial_balance=10000.0,
        fee_bps=10.0, threshold=0.001, max_position=0.5, shorting=False):
    csv_path = Path(csv_path)
    output = Path(output)
    source_path = csv_path.resolve()
    for name in ("backtest_results.csv", "performance.json", "backtest_plot.png"):
        target = output / name
        if (target.resolve() == source_path or
                (target.exists() and csv_path.exists() and target.samefile(csv_path))):
            raise ValueError(f"Report {name} would overwrite the input CSV; choose another output directory")
    dates, prices = load_prices(csv_path)
    x, y, indices = features(prices)
    split = int(len(x) * 0.7)
    # Purge one training row: its target would otherwise end at the first test decision.
    model, parameters, cv_mse = train(x[:split - 1], y[:split - 1], model_type)
    predictions = model.predict(x[split:])
    result = backtest(predictions, y[split:], initial_balance, fee_bps, threshold,
                      max_position, shorting)
    baseline = backtest(np.ones(len(predictions)), y[split:], initial_balance,
                        fee_bps, threshold=0, max_position=max_position)
    output.mkdir(parents=True, exist_ok=True)
    rows = zip(dates[indices[split:]], dates[indices[split:] + 1], predictions,
               y[split:], result["positions"], result["turnover"], result["fees"],
               result["net_returns"], result["equity"])
    with (output / "backtest_results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["decision_date", "settlement_date", "prediction", "asset_return",
                         "position", "turnover", "fees", "net_return", "equity"])
        writer.writerows(rows)
    summary = {"model": model_type, "parameters": parameters, "training_cv_mse": cv_mse,
               "test_mse": float(np.mean((predictions - y[split:]) ** 2)),
               "training_samples": split - 1, "purged_samples": 1,
               "test_samples": len(predictions), "fee_bps": fee_bps,
               "max_position": max_position, "shorting": shorting,
               "strategy": result["metrics"], "buy_and_hold": baseline["metrics"]}
    (output / "performance.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#101820")
    axes[0].plot(result["equity"], color="#68e6be", label=f"{model_type} strategy")
    axes[0].plot(baseline["equity"], color="#90a4be", label="Same-exposure buy & hold", alpha=.85)
    axes[0].set_title("Chronological held-out backtest · after transaction costs", loc="left", pad=18)
    axes[0].set_ylabel("Equity ($)")
    axes[0].legend(frameon=False)
    axes[1].step(np.arange(len(predictions)), result["positions"], color="#68e6be")
    axes[1].set_ylabel("Exposure")
    axes[1].set_xlabel("Held-out daily observation")
    for ax in axes:
        ax.set_facecolor("#101820")
        ax.grid(alpha=.12)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "backtest_plot.png", dpi=160)
    plt.close(fig)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Daily consecutive date,close CSV")
    source.add_argument("--demo", action="store_true", help="Use seeded synthetic prices")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--model", choices=["ridge", "rf"], default="ridge")
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--signal-threshold", type=float, default=0.001)
    parser.add_argument("--max-position", type=float, default=0.5)
    parser.add_argument("--enable-shorting", action="store_true")
    args = parser.parse_args(argv)
    csv_path = demo_prices(args.output / "synthetic_prices.csv") if args.demo else args.csv
    try:
        summary = run(csv_path, args.output, args.model, args.initial_balance,
                      args.fee_bps, args.signal_threshold, args.max_position, args.enable_shorting)
    except ValueError as exc:
        parser.error(str(exc))
    summary["data_source"] = "synthetic_demo" if args.demo else "user_csv"
    (args.output / "performance.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
