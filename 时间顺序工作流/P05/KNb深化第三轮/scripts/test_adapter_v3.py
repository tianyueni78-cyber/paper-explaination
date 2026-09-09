import json
import unittest
from unittest.mock import patch

from adapter_v3 import discrete_state, region_context
from run_experiments import ROOT, old


class AdapterV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT.parent / "KNb深挖" / "runs" / "budget-corrected" / "states.json"
        state = json.loads(source.read_text(encoding="utf-8"))[0]
        cls.data = old.load_case(state["instance"])
        cls.parent = old.Chromosome(**{key: tuple(value) for key, value in state["chromosome"].items()})
        cls.schedule = old.decode_static(cls.data, cls.parent)

    def test_region_context_uses_no_hidden_complete_decode(self):
        with old.Meter() as meter:
            context = region_context(self.data, self.parent, self.schedule, minimum_coverage_gain=0.05)
        self.assertEqual(meter.count, 0)
        self.assertEqual(tuple(context["opportunities25"]), (1, 2, 3, 4, 5, 6))
        self.assertEqual(tuple(context["opportunities50"]), (1, 2, 3, 4, 5, 6))
        self.assertTrue(all(value >= 0 for value in context["opportunities25"].values()))
        self.assertLessEqual(max(context["opportunities25"].values()), 1.0)
        self.assertLessEqual(max(context["opportunities50"].values()), 1.0)
        self.assertGreaterEqual(context["coverage50"], context["coverage25"])
        self.assertTrue(set(context["feasible25"]) <= set(range(1, 7)))

    def test_state_uses_expand_dominant_opportunity_trend_and_budget(self):
        self.assertEqual(discrete_state(True, {1: 2, 2: 5, 3: 5}, [], 96, 96), (1, 2, 0, 2))
        self.assertEqual(discrete_state(False, {1: 2, 2: 0}, [(0.0, 4)], 40, 96), (0, 1, 2, 1))
        self.assertEqual(discrete_state(False, {}, [(0.1, 4)], 20, 96), (0, 0, 1, 0))

    def test_finite_sampling_zero_does_not_mask_n1_or_n2(self):
        with patch("adapter_v3._os_opportunity_rates", return_value=(0.0, 0.0)):
            context = region_context(
                self.data, self.parent, self.schedule, minimum_coverage_gain=0.05
            )
        self.assertIn(1, context["feasible25"])
        self.assertIn(2, context["feasible25"])
        self.assertIn(1, context["feasible50"])
        self.assertIn(2, context["feasible50"])


if __name__ == "__main__":
    unittest.main()
