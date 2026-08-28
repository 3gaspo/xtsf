"""Static checks for tiered Git artifact publication."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TransferTierContractTest(unittest.TestCase):
    def test_publisher_defaults_to_lightweight_and_never_publishes_caches(self) -> None:
        publisher = (ROOT / "publish_job.sh").read_text(encoding="utf-8")

        self.assertIn('publish_size="lightweight"', publisher)
        self.assertIn('lightweight|detailed)', publisher)
        self.assertIn("**/window_metrics.csv", publisher)
        self.assertIn("**/per_user_date_metrics.csv", publisher)
        self.assertIn("**/setting_diagnostics_samples.csv", publisher)
        self.assertIn("**/criterion_loss.pdf", publisher)
        self.assertIn("**/example_prediction.pdf", publisher)
        self.assertIn("**/*.pt", publisher)
        self.assertIn("**/*.npy", publisher)
        self.assertIn("**/*.cbm", publisher)


if __name__ == "__main__":
    unittest.main()
