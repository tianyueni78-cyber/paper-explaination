import unittest

from lifecycle_experiment import checkpoints, repository_root, stage_of, summarize_paired


class LifecycleExperimentTests(unittest.TestCase):
    def test_repository_root_is_paper_explaination(self):
        self.assertTrue((repository_root() / ".git").exists())

    def test_checkpoints_cover_predeclared_lifecycle(self):
        self.assertEqual(checkpoints(20), [0, 2, 4, 8, 12, 16, 19])

    def test_stage_boundary_exits_adaptation_at_twenty_percent(self):
        self.assertEqual(stage_of(0, 20), "early")
        self.assertEqual(stage_of(3, 20), "early")
        self.assertEqual(stage_of(4, 20), "middle")
        self.assertEqual(stage_of(13, 20), "middle")
        self.assertEqual(stage_of(14, 20), "late")

    def test_summary_uses_instance_as_direction_unit(self):
        pairs = [
            {"instance": "Mk01", "seed": 1, "rep": 0, "delta": 0.2},
            {"instance": "Mk01", "seed": 2, "rep": 0, "delta": 0.4},
            {"instance": "Mk02", "seed": 1, "rep": 0, "delta": -0.1},
            {"instance": "Mk02", "seed": 2, "rep": 0, "delta": -0.3},
        ]
        result = summarize_paired(pairs, "delta", bootstrap_samples=200, bootstrap_seed=7)
        self.assertAlmostEqual(result["mean_difference"], 0.05)
        self.assertEqual(result["positive_instances"], 1)
        self.assertEqual(result["instances"], 2)
        self.assertAlmostEqual(result["per_instance"]["Mk01"], 0.3)
        self.assertAlmostEqual(result["per_instance"]["Mk02"], -0.2)


if __name__ == "__main__":
    unittest.main()
