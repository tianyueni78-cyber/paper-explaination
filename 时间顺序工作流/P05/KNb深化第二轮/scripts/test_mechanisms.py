import unittest
from mechanisms import rank_region, state_id, choose_n, choose_b, remember


class MechanismTests(unittest.TestCase):
    def test_adjustability_changes_ranking(self):
        self.assertEqual(rank_region([10, 8, 0], [False, True, True], [False, True, False], 1), {1})

    def test_zero_scores_tie_by_operation(self):
        self.assertEqual(rank_region([0, 0, 0], [False]*3, [False]*3, 2), {0, 1})

    def test_state_is_majority_and_zero_wait_is_low(self):
        self.assertEqual(state_id([0], [0], [True], {0}), 1)
        self.assertEqual(state_id([3], [1], [False], {0}), 2)

    def test_initial_visits_and_round_robin_exploration(self):
        history = {n: [(n / 10, 4)] for n in range(6)}
        self.assertEqual(choose_n(history, [0]*6, 0, 0), (0, 0))
        self.assertEqual(choose_n(history, [1]*6, 7, 0), (5, 0))
        self.assertEqual(choose_n(history, [1]*6, 8, 2), (2, 3))

    def test_recent_cost_weighted_credit_and_depth_cap(self):
        history = {0: [(1, 4)], 1: [(0, 4)]}
        self.assertEqual(choose_b(history, 0, 96), 12)
        self.assertEqual(choose_b(history, 1, 96), 4)
        self.assertEqual(choose_b(history, 0, 2), 2)
        self.assertEqual(choose_b({}, 0, 96), 4)

    def test_window_drops_old_record_and_ignores_zero_cost(self):
        history = {0: [(1, 4), (2, 4), (3, 4)]}
        remember(history, 0, 0, 0)
        remember(history, 0, 4, 12)
        self.assertEqual(history[0], [(2, 4), (3, 4), (4, 12)])


if __name__ == '__main__':
    unittest.main()
