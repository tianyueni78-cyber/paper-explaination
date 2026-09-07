import copy
import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("analyze.py")


def row(instance, state, action, repeat, value):
    fraction, neighborhood, budget = action
    return {
        "instance": instance,
        "state": state,
        "fraction": fraction,
        "n": neighborhood,
        "b": budget,
        "repeat": repeat,
        "hv_gain_per_allowed_decode": value,
        "gain": value * budget,
        "gain_per_decode": value,
        "makespan_improvement": value,
        "tec_improvement": value,
        "decodes": budget,
    }


class AnalyzeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("conditional_analyze", PATH)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_split_repeat_oracle_selects_on_zero_one_and_scores_on_two_three(self):
        a = (0.25, 1, 4)
        b = (0.25, 2, 4)
        rows = []
        for repeat, value in enumerate((10.0, 10.0, 0.0, 0.0)):
            rows.append(row("I1", "s1", a, repeat, value))
        for repeat, value in enumerate((5.0, 5.0, 9.0, 9.0)):
            rows.append(row("I1", "s1", b, repeat, value))
        result = self.module.split_repeat_oracle(rows)
        self.assertEqual(result[0]["selected_action"], a)
        self.assertEqual(result[0]["hv_gain_per_allowed_decode"], 0.0)

    def test_leave_one_instance_fixed_never_uses_heldout_values_for_selection(self):
        a = (0.25, 1, 4)
        b = (0.25, 2, 4)
        rows = []
        for instance in ("A", "B"):
            for action, value in ((a, 1.0), (b, 2.0 if instance == "B" else 100.0)):
                for repeat in range(4):
                    rows.append(row(instance, f"{instance}-s", action, repeat, value))
        first = self.module.leave_one_instance_fixed(rows)
        selected_for_a = next(item for item in first if item["instance"] == "A")["selected_action"]
        self.assertEqual(selected_for_a, b)
        changed = copy.deepcopy(rows)
        for item in changed:
            if item["instance"] == "A":
                item["hv_gain_per_allowed_decode"] *= -1000
        second = self.module.leave_one_instance_fixed(changed)
        self.assertEqual(
            selected_for_a,
            next(item for item in second if item["instance"] == "A")["selected_action"],
        )

    def test_knn_action_does_not_change_when_test_outcomes_change(self):
        a = (0.25, 1, 4)
        b = (0.25, 2, 4)
        states = [
            {"state": "A-s", "instance": "A", "features": [0.0, 0.0]},
            {"state": "B-s", "instance": "B", "features": [1.0, 1.0]},
            {"state": "T-s", "instance": "T", "features": [0.1, 0.1]},
        ]
        rows = []
        for instance, state, winner in (("A", "A-s", a), ("B", "B-s", b), ("T", "T-s", b)):
            for action in (a, b):
                for repeat in range(4):
                    rows.append(row(instance, state, action, repeat, 10.0 if action == winner else 0.0))
        first = self.module.leave_one_instance_knn(states, rows, neighbors=1)
        selected = next(item for item in first if item["instance"] == "T")["selected_action"]
        changed = copy.deepcopy(rows)
        for item in changed:
            if item["instance"] == "T":
                item["hv_gain_per_allowed_decode"] = 9999.0
        second = self.module.leave_one_instance_knn(states, changed, neighbors=1)
        self.assertEqual(
            selected,
            next(item for item in second if item["instance"] == "T")["selected_action"],
        )

    def test_generation_summary_uses_paired_state_differences(self):
        baseline = [
            {"state": "s1", "generation": 0, "hv_gain_per_allowed_decode": 1.0},
            {"state": "s2", "generation": 0, "hv_gain_per_allowed_decode": 2.0},
            {"state": "s3", "generation": 1, "hv_gain_per_allowed_decode": 4.0},
        ]
        method = [
            {"state": "s1", "generation": 0, "hv_gain_per_allowed_decode": 2.0},
            {"state": "s2", "generation": 0, "hv_gain_per_allowed_decode": 5.0},
            {"state": "s3", "generation": 1, "hv_gain_per_allowed_decode": 3.0},
        ]
        self.assertEqual(
            self.module.summarize_by_generation(method, baseline),
            {"0": 2.0, "1": -1.0},
        )

    def test_action_dimension_frequencies_are_reported_separately(self):
        records = [
            {"selected_action": (0.25, 3, 4)},
            {"selected_action": (0.25, 3, 12)},
            {"selected_action": (0.50, 5, 4)},
        ]
        self.assertEqual(
            self.module.action_dimension_frequencies(records),
            {
                "K": {"0.25": 2, "0.5": 1},
                "N": {"3": 2, "5": 1},
                "b": {"4": 2, "12": 1},
            },
        )

    def test_metric_summary_is_instance_equal_weighted(self):
        baseline = [
            {"state": "a1", "instance": "A", "gain": 1.0},
            {"state": "a2", "instance": "A", "gain": 1.0},
            {"state": "b1", "instance": "B", "gain": 4.0},
        ]
        method = [
            {"state": "a1", "instance": "A", "gain": 2.0},
            {"state": "a2", "instance": "A", "gain": 4.0},
            {"state": "b1", "instance": "B", "gain": 3.0},
        ]
        self.assertEqual(
            self.module.summarize_metric_difference(method, baseline, "gain"),
            {
                "mean_difference": 0.5,
                "positive_instances": 1,
                "instance_differences": {"A": 2.0, "B": -1.0},
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
