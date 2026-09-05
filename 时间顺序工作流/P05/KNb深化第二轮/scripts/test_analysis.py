import unittest
from analysis import paired_summary, supplement_summary, three_way


class AnalysisTests(unittest.TestCase):
    def test_instance_weights_are_equal_not_candidate_counts(self):
        pairs = [('a', 1, 1.), ('a', 1, 1.), ('b', 1, 3.)]
        result = paired_summary(pairs)
        self.assertEqual(result['mean_difference'], 2.)
        self.assertEqual(result['per_instance'], {'a': 1., 'b': 3.})

    def test_constant_effect_has_constant_interval(self):
        result = paired_summary([(i, s, .25) for i in ('a', 'b') for s in (1, 2)])
        self.assertEqual(result['interval95'], [.25, .25])

    def test_no_improvements_does_not_report_perfect_recall(self):
        row = dict(n=1, objective=[1, 1], gain=0., changed=[0], decodes=1,
                   passes={'0.25': dict(A=True, L=False, K=False, R=False)})
        result = supplement_summary([row])['N1/0.25/K']
        self.assertIsNone(result['positive_recall'])
        self.assertEqual(result['evaluated_pool'], 1)

    def test_three_way_is_zero_for_additive_factor_effects(self):
        values = {(k, n, b): 2*k+3*n+5*b for k in (0, 1) for n in (0, 1) for b in (0, 1)}
        self.assertEqual(three_way(values), 0)


if __name__ == '__main__':
    unittest.main()
