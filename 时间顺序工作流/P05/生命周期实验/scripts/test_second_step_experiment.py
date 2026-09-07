import unittest

from second_step_experiment import merge_transition_points, summarize_generation_one, validate_protocol


class SecondStepExperimentTests(unittest.TestCase):
    def test_protocol_requires_generation_one_with_full_twenty_generation_context(self):
        protocol = {
            "version": "lifecycle-step2-g1-v1",
            "generations": 20,
            "target_generation": 1,
            "select_cutoff": False,
        }
        validate_protocol(protocol, partial=True)

    def test_protocol_rejects_cutoff_selection(self):
        protocol = {
            "version": "lifecycle-step2-g1-v1",
            "generations": 20,
            "target_generation": 1,
            "select_cutoff": True,
        }
        with self.assertRaises(ValueError):
            validate_protocol(protocol, partial=True)

    def test_merge_adds_only_generation_one_between_existing_endpoints(self):
        prior = {"0": {"hv_gain": {"mean_difference": 0.1}},
                 "2": {"hv_gain": {"mean_difference": -0.1}}}
        generation_one = {"hv_gain": {"mean_difference": 0.02}}
        merged = merge_transition_points(prior, generation_one)
        self.assertEqual(list(merged), ["0", "1", "2"])
        self.assertEqual(merged["1"], generation_one)

    def test_generation_one_summary_uses_instances_not_repeats_as_direction_units(self):
        pairs = [
            {"instance": "Mk01", "seed": 1, "hv_gain": 0.3},
            {"instance": "Mk01", "seed": 2, "hv_gain": 0.1},
            {"instance": "Mk02", "seed": 1, "hv_gain": -0.2},
            {"instance": "Mk02", "seed": 2, "hv_gain": -0.4},
        ]
        summary = summarize_generation_one(pairs, ("hv_gain",), 200, 7)
        self.assertEqual(summary["hv_gain"]["instances"], 2)
        self.assertEqual(summary["hv_gain"]["positive_instances"], 1)

if __name__ == "__main__":
    unittest.main()
