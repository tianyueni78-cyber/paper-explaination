import unittest

from pointwise_analysis import objectives_at_prefix, paired_prefix_rows


class PointwiseAnalysisTests(unittest.TestCase):
    def test_prefix_uses_cumulative_decode_index_across_batches(self):
        row = {
            "trace": [
                {"decodes": 4, "total_decodes": 4, "candidates": [
                    {"decode_index": 1, "objective": [9.0, 10.0]},
                    {"decode_index": 4, "objective": [8.0, 10.0]},
                ]},
                {"decodes": 2, "total_decodes": 6, "candidates": [
                    {"decode_index": 2, "objective": [7.0, 10.0]},
                ]},
            ]
        }
        self.assertEqual(objectives_at_prefix(row, 4), [[9.0, 10.0], [8.0, 10.0]])
        self.assertEqual(objectives_at_prefix(row, 6), [[9.0, 10.0], [8.0, 10.0], [7.0, 10.0]])

    def test_pair_uses_smaller_real_decode_count(self):
        common = {
            "instance": "Mk01", "seed": 1, "generation": 0, "rep": 0,
            "base_objective": [10.0, 10.0], "trace": [],
        }
        rows = [
            dict(common, policy="fixed", decodes=7),
            dict(common, policy="feedback", decodes=5),
        ]
        result = paired_prefix_rows(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["common_decodes"], 5)

    def test_pair_rejects_missing_policy(self):
        rows = [{
            "instance": "Mk01", "seed": 1, "generation": 0, "rep": 0,
            "base_objective": [10.0, 10.0], "trace": [],
            "policy": "fixed", "decodes": 7,
        }]
        with self.assertRaises(ValueError):
            paired_prefix_rows(rows)


if __name__ == "__main__":
    unittest.main()
