import importlib.util
import unittest
from pathlib import Path

class AuditTests(unittest.TestCase):
    def test_reject_omitted_internal_decodes(self):
        p=Path(__file__).with_name('audit.py')
        self.assertTrue(p.exists(),'缺少审计器')
        spec=importlib.util.spec_from_file_location('audit',p)
        a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
        row={'b':4,'decodes':1,'n':4,'attempts':2,'evaluated':1,'candidates':[{}]}
        with self.assertRaises(AssertionError): a.check_counts(row)
        row['decodes']=3
        a.check_counts(row)

if __name__=='__main__':unittest.main(verbosity=2)
