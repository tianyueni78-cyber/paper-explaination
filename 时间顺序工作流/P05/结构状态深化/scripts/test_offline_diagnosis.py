import importlib.util
import math
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("offline_diagnosis.py")


class OfflineDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("offline_diagnosis", PATH)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_axis_actions_change_only_one_dimension(self):
        actions = [(k, n, b) for k in (0.25, 0.5) for n in range(1, 7) for b in (4, 12)]
        self.assertEqual(self.module.axis_actions(actions, "K"), [(0.25, 3, 4), (0.5, 3, 4)])
        self.assertEqual(len(self.module.axis_actions(actions, "N")), 6)
        self.assertEqual(self.module.axis_actions(actions, "b"), [(0.25, 3, 4), (0.25, 3, 12)])

    def test_split_oracle_selects_and_evaluates_on_disjoint_repeats(self):
        rows = []
        for action, selection, evaluation in (
            ((0.25, 3, 4), 10.0, 0.0),
            ((0.5, 3, 4), 5.0, 9.0),
        ):
            for repeat in range(4):
                value = selection if repeat < 2 else evaluation
                rows.append(self.module.make_test_row("s", "I", action, repeat, value))
        result = self.module.split_oracle(rows, [(0.25, 3, 4), (0.5, 3, 4)])
        self.assertEqual(result[0]["selected_action"], (0.25, 3, 4))
        self.assertEqual(result[0]["value"], 0.0)

    def test_stability_reports_agreement_and_evaluation_regret(self):
        rows = []
        for action, values in (
            ((0.25, 3, 4), (4.0, 4.0, 1.0, 1.0)),
            ((0.5, 3, 4), (2.0, 2.0, 5.0, 5.0)),
        ):
            for repeat, value in enumerate(values):
                rows.append(self.module.make_test_row("s", "I", action, repeat, value))
        result = self.module.stability(rows, [(0.25, 3, 4), (0.5, 3, 4)])
        self.assertEqual(result["top1_agreement"], 0.0)
        self.assertEqual(result["mean_regret"], 4.0)

    def test_structural_features_are_named_and_finite(self):
        values = self.module.structural_summary(
            machine_scores=[4.0, 3.0, 1.0, 0.0],
            agv_scores=[3.0, 2.0, 1.0, 0.0],
            waits=[1.0, 3.0, 2.0, 0.0],
            flexible=[1, 1, 0, 0],
            agv_assignments=[0, 0, 1, 1],
        )
        self.assertGreaterEqual(len(values), 8)
        self.assertTrue(all(isinstance(name, str) and math.isfinite(value) for name, value in values.items()))

    def test_hierarchical_choice_uses_k_then_n_then_b(self):
        actions = [(k, n, b) for k in (0.25, 0.5) for n in (3, 5) for b in (4, 12)]
        values = {action: 0.0 for action in actions}
        values[(0.5, 5, 12)] = 9.0
        values[(0.25, 3, 4)] = 8.0
        self.assertEqual(self.module.hierarchical_choice(values), (0.5, 5, 12))


if __name__ == "__main__":
    unittest.main(verbosity=2)
