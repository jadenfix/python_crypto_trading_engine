import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.backtest import backtest
from src.datasets import demo_prices, features, load_prices
from src.main import main, run

class BacktestTests(unittest.TestCase):
    def test_one_day_entry_and_liquidation_fees(self):
        r = backtest([1], [.1], initial_balance=100, fee_bps=100, max_position=1)
        self.assertAlmostEqual(r['equity'][0], 100 * .99 * 1.1 * .99)
        self.assertAlmostEqual(r['metrics']['total_fees'], 1 + 108.9 * .01)

    def test_weight_drift_rebalance(self):
        r = backtest([1, 1], [.1, 0], initial_balance=100, fee_bps=100, max_position=.5)
        first = 99.5 * 1.05
        drift = .5 * 1.1 / 1.05
        second = first * (1 - .01 * abs(.5 - drift)) * (1 - .01 * .5)
        self.assertAlmostEqual(r['equity'][1], second)

    def test_initial_drawdown(self):
        r = backtest([1], [-.1], fee_bps=0, max_position=1)
        self.assertAlmostEqual(r['metrics']['maximum_drawdown'], -.1)

    def test_flat_has_no_fees(self):
        r = backtest([0, 0], [.5, -.5])
        np.testing.assert_equal(r['equity'], [10000, 10000])
        self.assertEqual(r['metrics']['total_fees'], 0)

    def test_short_opt_in(self):
        self.assertEqual(backtest([-1], [-.1])['positions'][0], 0)
        self.assertGreater(backtest([-1], [-.1], shorting=True)['equity'][0], 10000)

    def test_invalid_parameters(self):
        for kw in ({'fee_bps': -1}, {'threshold': -1}, {'max_position': 2},
                   {'initial_balance': 0}, {'fee_bps': float('nan')}):
            with self.assertRaises(ValueError): backtest([1], [.1], **kw)
        with self.assertRaises(ValueError): backtest([1], [float('nan')])
        with self.assertRaises(ValueError): backtest([-1], [3], max_position=1, shorting=True)

class PipelineTests(unittest.TestCase):
    def test_features_do_not_use_future_prices(self):
        prices = np.linspace(100, 200, 100)
        x, _, indices = features(prices)
        changed = prices.copy()
        changed[51:] *= 2
        altered, _, _ = features(changed)
        np.testing.assert_equal(x[indices <= 50], altered[indices <= 50])

    def test_target_is_next_daily_return(self):
        prices = np.linspace(100, 200, 100)
        _, y, indices = features(prices)
        np.testing.assert_allclose(y, prices[indices + 1] / prices[indices] - 1)

    def test_duplicate_dates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = demo_prices(Path(tmp) / 'prices.csv', days=100)
            with path.open() as f: rows = list(csv.reader(f))
            rows[51][0] = rows[50][0]
            with path.open('w', newline='') as f: csv.writer(f).writerows(rows)
            with self.assertRaises(ValueError): load_prices(path)

    def test_end_to_end_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = demo_prices(Path(tmp) / 'prices.csv', days=160)
            a = run(path, Path(tmp) / 'first')
            b = run(path, Path(tmp) / 'second')
            self.assertEqual(a, b)
            output = Path(tmp) / 'first'
            self.assertGreater((output / 'backtest_plot.png').stat().st_size, 1000)
            with (output / 'backtest_results.csv').open() as f: rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), a['test_samples'])
            self.assertEqual(json.loads((output / 'performance.json').read_text()), a)
            self.assertEqual(a['purged_samples'], 1)

    def test_demo_labels_synthetic_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            main(['--demo', '--output', tmp])
            self.assertEqual(json.loads((Path(tmp) / 'performance.json').read_text())['data_source'], 'synthetic_demo')

if __name__ == '__main__': unittest.main()
