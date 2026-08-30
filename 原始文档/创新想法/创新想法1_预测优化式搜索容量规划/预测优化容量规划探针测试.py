import unittest


class CapacityPlanningProbeTests(unittest.TestCase):
    def test_choose_action_obeys_time_limit_then_maximizes_predicted_gain(self):
        from innovation_probe_capacity.probe_10_capacity_planning import choose_action

        rows = [
            {"action_id": "slow", "predicted_gain": 0.9, "predicted_time": 2.0},
            {"action_id": "fast", "predicted_gain": 0.6, "predicted_time": 0.5},
            {"action_id": "tiny", "predicted_gain": 0.2, "predicted_time": 0.1},
        ]

        self.assertEqual(choose_action(rows, time_limit=1.0)["action_id"], "fast")

    def test_choose_action_falls_back_to_fastest_when_predictions_miss_limit(self):
        from innovation_probe_capacity.probe_10_capacity_planning import choose_action

        rows = [
            {"action_id": "a", "predicted_gain": 0.9, "predicted_time": 2.0},
            {"action_id": "b", "predicted_gain": 0.2, "predicted_time": 1.5},
        ]

        self.assertEqual(choose_action(rows, time_limit=1.0)["action_id"], "b")


if __name__ == "__main__":
    unittest.main()
