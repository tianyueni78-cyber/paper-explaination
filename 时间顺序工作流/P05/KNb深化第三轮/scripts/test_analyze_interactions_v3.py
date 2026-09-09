import unittest

from analyze_interactions_v3 import factorial_contrast, paired_contrast


class AnalyzeInteractionsV3Tests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        values = {
            "000": 0.00,
            "001": 0.10,
            "010": 0.20,
            "011": 0.35,
            "100": 0.40,
            "101": 0.70,
            "110": 0.65,
            "111": 1.05,
        }
        for instance in ("Mk01", "Mk02"):
            for configuration, gain in values.items():
                self.rows.append({
                    "state": f"{instance}-g0-p0",
                    "instance": instance,
                    "repeat": 0,
                    "configuration": configuration,
                    "gain": gain,
                    "decodes": 10,
                })

    def test_paired_contrast_uses_matching_experimental_units(self):
        result = paired_contrast(self.rows, "101", "100", bootstrap_samples=100)
        self.assertEqual(result["pairs"], 2)
        self.assertAlmostEqual(result["mean_gain_difference"], 0.30)
        self.assertEqual(result["positive_gain_instances"], 2)

    def test_factorial_contrast_reports_kb_interaction(self):
        result = factorial_contrast(self.rows, "K:b", bootstrap_samples=100)
        # (101-100) - (001-000) averaged over N=0/1.
        self.assertAlmostEqual(result["mean_gain_contrast"], 0.225)
        self.assertEqual(result["positive_gain_instances"], 2)


if __name__ == "__main__":
    unittest.main()
