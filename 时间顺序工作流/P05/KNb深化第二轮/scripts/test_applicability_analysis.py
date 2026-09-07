import unittest

from applicability_analysis import ROOT, load_rows, objective_metrics, paired_rows, summarize


class ApplicabilityAnalysisTests(unittest.TestCase):
    def test_load_rows_accepts_audited_legacy_filename(self):
        folder = ROOT.parent / "KNb深挖/runs/budget-corrected"
        rows, _ = load_rows(
            folder, "D3_raw.jsonl.gz", "D3_trajectories",
            "a6a2b88c60ba5d8557d65d84a9a801acf2d8155442980de8469c8876df3ad4eb",
        )
        self.assertEqual(len(rows), 648)

    def test_paired_rows_rejects_missing_pair(self):
        rows = [
            {"state": "s1", "fraction": 0.25, "rep": 0, "config": ["new"]},
            {"state": "s2", "fraction": 0.25, "rep": 0, "config": ["new"]},
            {"state": "s1", "fraction": 0.25, "rep": 0, "config": ["fixed"]},
        ]
        with self.assertRaises(ValueError):
            paired_rows(rows, ("new",), ("fixed",))

    def test_objective_metrics_reports_separate_extremes(self):
        row = {
            "gain": 0.2,
            "decodes": 4,
            "wall_seconds": 2.0,
            "trace": [{"candidates": [
                {"objective": [8.0, 12.0]},
                {"objective": [11.0, 7.0]},
            ]}],
        }
        got = objective_metrics(row, (10.0, 10.0))
        self.assertEqual(got["hv_gain"], 0.2)
        self.assertEqual(got["hv_per_decode"], 0.05)
        self.assertEqual(got["makespan_improvement"], 0.2)
        self.assertEqual(got["tec_improvement"], 0.3)
        self.assertEqual(got["decodes"], 4)

    def test_summarize_keeps_condition_and_metric_directions(self):
        pairs = [
            ({"instance": "A", "generation": 0}, {"hv_gain": 0.3, "decodes": 4},
             {"hv_gain": 0.1, "decodes": 4}),
            ({"instance": "B", "generation": 0}, {"hv_gain": 0.1, "decodes": 4},
             {"hv_gain": 0.2, "decodes": 4}),
        ]
        got = summarize(pairs, "generation", ("hv_gain", "decodes"))
        self.assertAlmostEqual(got["0"]["hv_gain"]["mean_difference"], 0.05)
        self.assertEqual(got["0"]["hv_gain"]["positive_pairs"], 1)
        self.assertEqual(got["0"]["decodes"]["mean_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
