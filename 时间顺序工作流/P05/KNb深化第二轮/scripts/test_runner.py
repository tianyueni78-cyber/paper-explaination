import json
import unittest
from unittest.mock import patch
from run_experiments import ROOT, old, batch, walk


class RunnerTests(unittest.TestCase):
    def test_last_batch_crossing_deadline_is_not_success(self):
        import run_experiments as runner
        clock = [0]
        original = runner.batch
        def slow_batch(*args, **kwargs):
            result = original(*args, **kwargs)
            clock[0] = 2
            return result
        with patch.object(runner.time, 'perf_counter', side_effect=lambda: clock[0]), patch.object(runner, 'batch', side_effect=slow_batch):
            with self.assertRaises(TimeoutError) as caught:
                walk(self.data, self.parent, self.schedule, ('K', 'fixed', '4'), 1.0, 91, 1, deadline=1)
        self.assertEqual(caught.exception.partial_result['decodes'], 1)
        self.assertEqual(len(caught.exception.partial_result['trace']), 1)

    def test_deadline_retains_partial_record(self):
        with self.assertRaises(TimeoutError) as caught:
            walk(self.data, self.parent, self.schedule, ('K', 'fixed', '4'), 0.5, 0, 24, deadline=0)
        self.assertEqual(caught.exception.partial_result['decodes'], 0)

    def test_validation_failure_retains_consumed_decode_and_restores_meter(self):
        original = old.nb.decode_static
        with patch.object(old, 'validate_schedule', side_effect=ValueError('injected')):
            with self.assertRaises(ValueError) as caught:
                walk(self.data, self.parent, self.schedule, ('K', 'fixed', '4'), 1.0, 91, 24)
        self.assertEqual(caught.exception.partial_result['decodes'], 1)
        self.assertIs(old.nb.decode_static, original)

    def test_empty_region_rejects_every_candidate(self):
        _, _, result = batch(self.data, self.parent, self.schedule, set(), 2, 1, 91,
            [(1, 1)], (self.schedule.makespan, self.schedule.machine_energy), {self.parent})
        self.assertEqual(result['evaluated'], 0)
        self.assertEqual(result['decodes'], 0)
        self.assertEqual(result['outside'] + result['noop'], result['attempts'])

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            walk(self.data, self.parent, self.schedule, ('K', 'typo', 'recent'), 0.5, 0, 1)

    def test_changed_protocol_constants_are_rejected(self):
        import run_experiments as runner
        protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
        protocol['window'] = 9
        with self.assertRaises(ValueError):
            runner.validate_protocol(protocol)

    @classmethod
    def setUpClass(cls):
        state = json.loads((ROOT.parent / 'KNb深挖/runs/budget-corrected/states.json').read_text(encoding='utf-8'))[0]
        cls.data = old.load_case(state['instance'])
        cls.parent = old.Chromosome(**{key: tuple(value) for key, value in state['chromosome'].items()})
        cls.schedule = old.decode_static(cls.data, cls.parent)

    def test_all_neighborhoods_charge_internal_decode_and_preserve_legality(self):
        base = (self.schedule.makespan, self.schedule.machine_energy)
        for n in range(6):
            with self.subTest(n=n):
                parent, schedule, result = batch(self.data, self.parent, self.schedule,
                    set(range(self.parent.operation_count)), n, 4, 91, [(1, 1)], base, {self.parent})
                self.assertGreater(result['decodes'], 0)
                self.assertLessEqual(result['decodes'], 4)
                self.assertEqual(result['decodes'], result['evaluated'] + (result['attempts'] if n in (3, 5) else 0))
                old.validate_schedule(self.data, parent, schedule)
                self.assertEqual(parent.empty_speed, self.parent.empty_speed)

    def test_internal_neighborhood_cannot_start_with_one_remaining(self):
        _, _, result = batch(self.data, self.parent, self.schedule, {0}, 3, 1, 0,
                              [(1, 1)], (self.schedule.makespan, self.schedule.machine_energy), {self.parent})
        self.assertEqual(result.get('attempts'), 0)

    def test_walk_repeats_exactly_and_resets_learning(self):
        cfg = ('K', 'state', 'recent')
        first = walk(self.data, self.parent, self.schedule, cfg, 0.5, 18, budget=24)
        second = walk(self.data, self.parent, self.schedule, cfg, 0.5, 18, budget=24)
        self.assertGreater(first['decodes'], 0)
        self.assertEqual(first['trace'], second['trace'])
        self.assertLessEqual(first['decodes'], 24)
        self.assertEqual(first['decodes'], sum(row['decodes'] for row in first['trace']))


if __name__ == '__main__':
    unittest.main()
