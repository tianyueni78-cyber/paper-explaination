import sys
import unittest
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'S0/scripts'))
import probe as old
import probe_knb as k

class MechanismTests(unittest.TestCase):
    def test_feedback_exploration_visits_all_channels(self):
        self.assertEqual([k.exploration_channel(2,s) for s in range(12,60,4)], [2,3,4,5,6,7,8,9,10,11,0,1])
    def test_wait_decomposition_and_unique_successors(self):
        # Break caught: charging transport duration as waiting / duplicate links.
        self.assertEqual(k.wait_pair(2,5,7,11), (3,4))
        self.assertEqual(k.propagate([7,2,3], [{1,2},{2},set()]), [12,5,3])

    def test_capture_preserves_a0_random_trajectory(self):
        data=old.load_case('Mk01')
        snapshots,cost,result=k.capture(data,101)
        baseline=old.q.run_qnsga2(data,population_size=20,generations=20,seed=101)
        self.assertEqual(result,baseline)
        self.assertEqual([x[0] for x in snapshots],[0,9,19])
        self.assertGreater(cost,0)

    def test_continuous_budget_replay_and_internal_count(self):
        data=old.load_case('Mk01')
        parent=old.hybrid_population(data,20,len(data.agv.speeds),old.random.Random(1)).chromosomes[0]
        schedule=old.decode_static(data,parent)
        zero=k.walk(data,parent,schedule,'fixed',3,0,123)
        self.assertEqual(zero['decodes'],0)
        a=k.walk(data,parent,schedule,'fixed',3,12,123)
        b=k.walk(data,parent,schedule,'fixed',3,12,123)
        self.assertEqual(a['gain'],b['gain'])
        self.assertEqual(a['decodes'],sum(t['decodes'] for t in a['trace']))
        self.assertLessEqual(a['decodes'],12)
        for t in a['trace']:
            self.assertEqual(t['decodes'],t['evaluated']+t['attempts'])
        self.assertGreaterEqual(a['gain'],0)
        self.assertIs(old.nb.decode_static,old.decode_static)

    def test_training_excludes_validation_labels(self):
        rows=[dict(split='explore',group='A',fraction=.25,n=1,gain_per_budget=1),
              dict(split='explore',group='A',fraction=.25,n=2,gain_per_budget=2),
              dict(split='validation',group='A',fraction=.25,n=1,gain_per_budget=100)]
        fitted=k.fit(rows)
        self.assertEqual(fitted['conditional']['A|0.25'],2)
        self.assertEqual(fitted['global_n'],2)

if __name__=='__main__': unittest.main(verbosity=2)
