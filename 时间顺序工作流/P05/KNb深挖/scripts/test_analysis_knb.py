import unittest
import analyze_knb as a

class AnalysisTests(unittest.TestCase):
    def test_instance_weight_not_candidate_weight(self):
        rows=[dict(instance='a',policy='p',gain=1.)]*10+[dict(instance='b',policy='p',gain=3.)]
        self.assertEqual(a.instance_mean(rows,'gain'),2.)

    def test_training_permutation_does_not_touch_validation(self):
        rows=[dict(split='explore',group=g) for g in 'ABC']+[dict(split='validation',group='L')]
        shuffled=a.shuffle_labels(rows)
        self.assertEqual(shuffled[-1]['group'],'L')
        self.assertEqual(rows[-1]['group'],'L')
        self.assertEqual(sorted(r['group'] for r in shuffled[:3]),list('ABC'))

if __name__=='__main__':unittest.main(verbosity=2)
