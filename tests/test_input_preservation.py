"""Input preservation checks using temporary, explicitly synthetic CSVs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.datasets import demo_prices
from src.main import run


REPORTS = ("backtest_results.csv", "performance.json", "backtest_plot.png")


class InputPreservationTests(unittest.TestCase):
    def test_direct_output_collisions_rejected(self):
        for name in REPORTS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                path = demo_prices(Path(tmp) / name, days=160)
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "overwrite.*input"):
                    run(path, tmp)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(list(Path(tmp).iterdir()), [path])

    def check_alias(self, kind):
        for name in REPORTS:
            with self.subTest(kind=kind, name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = demo_prices(root / "prices.csv", days=160)
                before = path.read_bytes()
                output = root / "reports"
                output.mkdir()
                target = output / name
                if kind == "symlink":
                    target.symlink_to(path)
                else:
                    os.link(path, target)
                with self.assertRaisesRegex(ValueError, "overwrite.*input"):
                    run(path, output)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(list(output.iterdir()), [target])
                self.assertTrue(target.samefile(path))

    def test_symlink_aliases_rejected(self):
        self.check_alias("symlink")

    def test_hardlink_aliases_rejected(self):
        self.check_alias("hardlink")

    def test_parent_directory_alias_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = demo_prices(root / "backtest_results.csv", days=160)
            (root / "nested").mkdir()
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "overwrite.*input"):
                run(root / "nested" / ".." / path.name, root)
            self.assertEqual(path.read_bytes(), before)

    def test_cli_collision_exits_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = demo_prices(Path(tmp) / "backtest_results.csv", days=160)
            before = path.read_bytes()
            result = subprocess.run([sys.executable, "-m", "src.main", "--csv", str(path),
                                     "--output", tmp], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("overwrite", result.stderr)
            self.assertEqual(path.read_bytes(), before)

    def test_same_directory_noncolliding_cli_still_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = demo_prices(Path(tmp) / "prices.csv", days=160)
            before = path.read_bytes()
            result = subprocess.run([sys.executable, "-m", "src.main", "--csv", str(path),
                                     "--output", tmp], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(path.read_bytes(), before)
            output = Path(tmp)
            summary = json.loads((output / "performance.json").read_text())
            self.assertEqual(summary["data_source"], "user_csv")
            for name in REPORTS:
                self.assertGreater((output / name).stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
