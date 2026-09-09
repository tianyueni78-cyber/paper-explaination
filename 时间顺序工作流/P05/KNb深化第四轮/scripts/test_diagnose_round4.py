import unittest

from diagnose_round4 import factorial_interaction, trajectory_stats


def row(configuration, gain, action, state=(1, 3, 0, 2), mode="adaptive"):
    return {
        "configuration": configuration,
        "instance": "Mk01",
        "generation": 0,
        "repeat": 0,
        "gain": gain,
        "decodes": 3,
        "trace": [
            {
                "action": list(action),
                "state": list(state),
                "mode": mode,
                "q_updated": mode == "adaptive",
                "decodes": 3,
                "gain": gain,
                "tranches": [
                    {
                        "attempts": 5,
                        "outside": 1,
                        "noop": 1,
                        "duplicate": 0,
                        "evaluated": 3,
                        "decodes": 3,
                        "accepted": 1,
                        "gain": gain,
                    }
                ],
            }
        ],
    }


class TrajectoryStatsTests(unittest.TestCase):
    def test_counts_actions_failures_and_expand_choice(self):
        stats = trajectory_stats(row("100_q", 0.2, (0.5, 4, 8)))
        self.assertEqual(stats["decisions"], 1)
        self.assertEqual(stats["attempts"], 5)
        self.assertEqual(stats["outside"], 1)
        self.assertEqual(stats["noop"], 1)
        self.assertEqual(stats["evaluated"], 3)
        self.assertEqual(stats["k_counts"], {"0.5": 1})
        self.assertEqual(stats["n_counts"], {"4": 1})
        self.assertEqual(stats["b_counts"], {"8": 1})
        self.assertEqual(stats["mode_counts"], {"adaptive": 1})
        self.assertEqual(stats["expand_decisions"], 1)
        self.assertEqual(stats["expand_k50"], 1)

    def test_accepted_chromosome_object_counts_as_one_acceptance(self):
        sample = row("100_q", 0.2, (0.5, 4, 8))
        sample["trace"][0]["tranches"][0]["accepted"] = {"os": [0, 1]}
        stats = trajectory_stats(sample)
        self.assertEqual(stats["accepted"], 1)

    def test_zero_decode_decision_is_counted(self):
        sample = row("100_q", 0.0, (0.25, 3, 4), state=(0, 3, 0, 1), mode="fallback")
        sample["trace"][0]["decodes"] = 0
        sample["trace"][0]["tranches"] = []
        stats = trajectory_stats(sample)
        self.assertEqual(stats["zero_decode_decisions"], 1)
        self.assertEqual(stats["expand_decisions"], 0)


class FactorialInteractionTests(unittest.TestCase):
    def test_uses_matched_four_cell_difference(self):
        rows = [
            row("000_fixed", 1.0, (0.25, 3, 4)),
            row("001_q", 3.0, (0.25, 3, 8)),
            row("100_q", 2.0, (0.25, 3, 4)),
            row("101_q", 7.0, (0.25, 3, 8)),
        ]
        result = factorial_interaction(rows, "000_fixed", "001_q", "100_q", "101_q")
        self.assertEqual(result[0]["interaction"], (7.0 - 2.0) - (3.0 - 1.0))
        self.assertEqual(result[0]["unit"], "Mk01|g0|r0")


if __name__ == "__main__":
    unittest.main()
