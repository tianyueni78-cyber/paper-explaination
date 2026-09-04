import unittest
from diagnose import funnel, tail


class DiagnosisTests(unittest.TestCase):
    def test_funnel_uses_attempts_not_decodes_as_rejection_denominator(self):
        rows = [dict(attempts='10', outside='6', noop='1', duplicate='1', evaluated='2', decodes='12', gain='0.2')]
        result = funnel(rows)
        self.assertEqual(result['outside_rate'], 0.6)
        self.assertEqual(result['evaluated'], 2)
        self.assertEqual(result['decodes'], 12)

    def test_zero_attempts_is_missing_rate_not_success(self):
        row = dict(attempts='0', outside='0', noop='0', duplicate='0', evaluated='0', decodes='0', gain='0')
        self.assertIsNone(funnel([row])['outside_rate'])

    def test_tail_counts_only_cost_after_last_positive_gain(self):
        trace = [dict(gain=1, decodes=4), dict(gain=0, decodes=3), dict(gain=0, decodes=0)]
        self.assertEqual(tail(trace), {'tail_decodes': 3, 'tail_batches': 2, 'ever_improved': True})

    def test_no_gain_keeps_all_spend(self):
        self.assertEqual(tail([dict(gain=0, decodes=4)]), {'tail_decodes': 4, 'tail_batches': 1, 'ever_improved': False})


if __name__ == '__main__':
    unittest.main()
