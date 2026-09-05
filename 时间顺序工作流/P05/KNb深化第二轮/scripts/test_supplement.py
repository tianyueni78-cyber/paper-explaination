import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import supplement
from run_experiments import ROOT, old


class SupplementTests(unittest.TestCase):
    def test_output_open_failure_is_preserved_in_manifest(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)/'run'
            with patch.object(supplement.sys, 'argv', ['supplement', '--output', str(output)]), patch.object(supplement.gzip, 'open', side_effect=OSError('disk failure')):
                with self.assertRaisesRegex(OSError, 'disk failure'):
                    supplement.main()
            self.assertEqual(json.loads((output/'manifest.json').read_text())['status'], 'failed')

    @classmethod
    def setUpClass(cls):
        state = json.loads((ROOT.parent/'KNb深挖/runs/budget-corrected/states.json').read_text(encoding='utf-8'))[0]
        cls.data = old.load_case(state['instance'])
        cls.parent = old.Chromosome(**{k: tuple(v) for k, v in state['chromosome'].items()})
        cls.schedule = old.decode_static(cls.data, cls.parent)

    def test_unfiltered_proposals_are_evaluated_and_internal_calls_charged(self):
        rows = list(supplement.proposals(self.data, self.parent, self.schedule, 'fixture'))
        self.assertEqual(len(rows), 192)
        for row in rows:
            self.assertEqual(row['decodes'], int(bool(row['changed'])) + int(row['n'] in (4, 6)))
            self.assertEqual(row['objective'] is not None, bool(row['changed']))
        self.assertTrue(any(row['objective'] is not None and not row['passes']['0.25']['K'] for row in rows))

    def test_proposals_repeat_without_learning_or_parent_changes(self):
        first = list(supplement.proposals(self.data, self.parent, self.schedule, 'fixture'))
        second = list(supplement.proposals(self.data, self.parent, self.schedule, 'fixture'))
        self.assertGreater(len(first), 0)
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
