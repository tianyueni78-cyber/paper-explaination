import unittest

from mechanisms_v3 import (
    FactorizedQ,
    confidence_action,
    decay_state_credit,
    feasible_neighborhoods,
    k_candidates,
    next_budget_tranche,
    state_change_score,
)


class MechanismsV3Tests(unittest.TestCase):
    def test_k50_requires_marginal_coverage_or_new_opportunity(self):
        self.assertEqual(k_candidates(0.70, 0.72, 8, 8, 0.05), (0.25,))
        self.assertEqual(k_candidates(0.70, 0.80, 8, 8, 0.05), (0.25, 0.5))
        self.assertEqual(k_candidates(0.70, 0.72, 8, 10, 0.05), (0.25, 0.5))

    def test_neighborhood_mask_uses_only_pre_action_opportunity(self):
        self.assertEqual(feasible_neighborhoods({1: 0, 2: 3, 3: 0, 4: 1, 5: 0, 6: 2}), (2, 4, 6))
        self.assertEqual(feasible_neighborhoods({n: 0 for n in range(1, 7)}), ())

    def test_budget_is_allocated_in_real_decode_tranches(self):
        self.assertEqual(next_budget_tranche(0, [], 20, 0.01), 4)
        self.assertEqual(next_budget_tranche(4, [(0.08, 4)], 20, 0.01), 4)
        self.assertEqual(next_budget_tranche(8, [(0.08, 4)], 20, 0.01, maximum=8), 0)
        self.assertEqual(next_budget_tranche(8, [(0.0, 4)], 20, 0.01), 0)
        self.assertEqual(next_budget_tranche(12, [(1.0, 4)], 20, 0.01), 0)
        self.assertEqual(next_budget_tranche(0, [], 2, 0.01), 2)

    def test_factorized_update_changes_joint_value_by_standard_td_amount(self):
        table = FactorizedQ()
        state = "s0"
        action = (0.25, 2, 4)
        self.assertEqual(table.value(state, action), 0.0)
        td = table.update(state, action, reward=1.0, next_state="s1", next_actions=[], alpha=0.5, gamma=0.9)
        self.assertEqual(td, 1.0)
        self.assertAlmostEqual(table.value(state, action), 0.5)

    def test_confidence_gate_requires_two_actions_and_sufficient_gap(self):
        values = {(0.25, 2, 4): 0.8, (0.5, 4, 4): 0.6}
        self.assertEqual(confidence_action(values, 0.1), (0.25, 2, 4))
        self.assertIsNone(confidence_action(values, 0.3))
        self.assertIsNone(confidence_action({(0.25, 2, 4): 0.8}, 0.0))

    def test_state_change_and_decay_affect_only_old_state(self):
        score = state_change_score({1, 2}, {2, 3}, (1, 0, 2), (0, 0, 4))
        self.assertGreater(score, 0.5)
        table = FactorizedQ()
        table.update("old", (0.25, 2, 4), 1.0, "next", [], 1.0, 0.0)
        table.update("keep", (0.25, 2, 4), 1.0, "next", [], 1.0, 0.0)
        decay_state_credit(table, "old", 0.25)
        self.assertAlmostEqual(table.value("old", (0.25, 2, 4)), 0.25)
        self.assertAlmostEqual(table.value("keep", (0.25, 2, 4)), 1.0)


if __name__ == "__main__":
    unittest.main()
