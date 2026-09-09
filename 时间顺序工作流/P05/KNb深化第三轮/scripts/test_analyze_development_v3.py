import unittest

from analyze_development_v3 import paired_differences


class AnalyzeDevelopmentV3Tests(unittest.TestCase):
    def test_paired_difference_uses_same_state_and_repeat(self):
        rows = [
            {"state": "a", "repeat": 0, "configuration": "base", "gain": 1.0, "decodes": 10},
            {"state": "a", "repeat": 0, "configuration": "new", "gain": 1.2, "decodes": 12},
        ]
        result = paired_differences(rows, "new", "base")
        self.assertAlmostEqual(result[0]["gain_difference"], 0.2)
        self.assertEqual(result[0]["decode_difference"], 2)
        self.assertAlmostEqual(result[0]["efficiency_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
