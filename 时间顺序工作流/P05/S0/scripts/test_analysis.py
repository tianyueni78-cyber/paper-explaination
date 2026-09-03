import importlib.util
import unittest
from pathlib import Path

class AnalysisTests(unittest.TestCase):
    def setUp(self):
        p=Path(__file__).with_name('analyze.py')
        self.assertTrue(p.exists(),'缺少分析器')
        spec=importlib.util.spec_from_file_location('analysis',p)
        self.a=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.a)
    def test_selection_uses_training_repeats_only(self):
        rows=[dict(state='a',action='x',rep=0,utility=1.),dict(state='a',action='x',rep=2,utility=99.)]
        self.assertEqual(self.a.action_means(rows,{0,1}),{'a':{'x':1.}})
    def test_nearest_state_prediction(self):
        states=[{'state':'a','features':[0.]},{'state':'b','features':[10.]}]
        self.assertEqual(self.a.predict(states,{'features':[1.]},{'a':{'x':2.,'y':1.},'b':{'x':0.,'y':3.}},1,1),'x')
    def test_tied_actions_deterministic(self):
        self.assertEqual(self.a.best({'z':1.,'a':1.}),'a')

if __name__=='__main__': unittest.main(verbosity=2)
