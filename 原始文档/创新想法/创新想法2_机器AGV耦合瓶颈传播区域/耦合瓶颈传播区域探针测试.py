import unittest


class CoupledRegionTests(unittest.TestCase):
    def test_coupled_score_rewards_downstream_propagation(self):
        from innovation_probe_coupling.probe_11_coupled_region import coupled_score

        self.assertGreater(
            coupled_score(transport=4.0, processing=2.0, successors=4),
            coupled_score(transport=6.0, processing=2.0, successors=0),
        )

    def test_select_region_uses_highest_scores(self):
        from innovation_probe_coupling.probe_11_coupled_region import select_region

        scores = {(1, 1): 2.0, (1, 2): 7.0, (2, 1): 5.0}
        self.assertEqual(select_region(scores, 2), ((1, 2), (2, 1)))


if __name__ == "__main__":
    unittest.main()
