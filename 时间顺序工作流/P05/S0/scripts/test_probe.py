"""测试破坏点：漏记内部解码、OS越界漏检、HV重复计入和父解污染。"""
import importlib.util
import random
import unittest
from pathlib import Path

PATH = Path(__file__).with_name('probe.py')

class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.exists(), '缺少隔离探针实现')
        spec = importlib.util.spec_from_file_location('probe', PATH)
        self.p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.p)

    def test_hv_union_not_sum(self):
        self.assertAlmostEqual(self.p.hv([(1,1),(.8,1),(1,.8),(.8,1)]), .05)

    def test_os_displacement_region(self):
        C = self.p.Chromosome
        a = C((0,1,0,1),(0,)*4,(0,)*4,(0,)*4,(0,)*4)
        b = C((1,0,0,1),(0,)*4,(0,)*4,(0,)*4,(0,)*4)
        self.assertEqual(self.p.touched(a,b,(2,2)), {0,2})

    def test_internal_decode_budget_and_restore(self):
        p=self.p
        data=p.load_case('Mk01')
        c=p.hybrid_population(data,10,4,random.Random(11)).chromosomes[0]
        original=p.nb.decode_static
        with p.Meter(1) as meter:
            p.nb.apply_neighborhood(data,c,3,random.Random(2))
            self.assertEqual(meter.count,1)
            with self.assertRaises(p.BudgetExhausted): meter.decode(data,c)
        self.assertIs(p.nb.decode_static,original)

    def test_trial_repeat_and_parent_unchanged(self):
        p=self.p; data=p.load_case('Mk01')
        c=p.hybrid_population(data,10,4,random.Random(11)).chromosomes[0]
        before=c.to_matlab_row()
        schedule=p.decode_static(data,c)
        x=p.trial(data,c,schedule,set(range(c.operation_count)),4,4,17)
        y=p.trial(data,c,schedule,set(range(c.operation_count)),4,4,17)
        for key in ('gain','decodes','attempts','evaluated','candidates'):
            self.assertEqual(x[key],y[key])
        self.assertEqual(c.to_matlab_row(),before)
        self.assertLessEqual(x['decodes'],4)
        self.assertEqual(x['decodes'],x['evaluated'])

    def test_training_excludes_heldout(self):
        rows=[{'instance':'A','state':'a'},{'instance':'B','state':'b'}]
        self.assertEqual(self.p.training_states(rows,'A'),[rows[1]])

if __name__=='__main__': unittest.main(verbosity=2)
