import unittest
import audit_knb as a

class AuditTests(unittest.TestCase):
    def test_json_roundtrip_preserves_trace_values(self):
        self.assertTrue(a.same_trace([{'os':(1,2)}],[{'os':[1,2]}]))
        self.assertFalse(a.same_trace([{'os':(1,2)}],[{'os':[2,1]}]))
    def test_independent_hv_rectangle_union(self):
        self.assertAlmostEqual(a.area([(1.,1.),(.9,.9)]),.04)

    def test_budget_rejects_omitted_internal_decode(self):
        row=dict(decodes=1,evaluated=1,attempts=1,n=4,candidates=[{}],outside=0,noop=0,duplicate=0)
        with self.assertRaises(AssertionError):a.check_counts(row,4)

if __name__=='__main__':unittest.main(verbosity=2)
