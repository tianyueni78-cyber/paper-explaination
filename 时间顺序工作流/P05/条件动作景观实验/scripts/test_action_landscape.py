import importlib.util
import csv
import math
import random
import tempfile
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("action_landscape.py")


class ActionLandscapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("action_landscape", PATH)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_protocol_has_24_ordered_actions_and_disjoint_splits(self):
        protocol = self.module.load_protocol()
        action_list = self.module.actions(protocol)
        self.assertEqual(len(action_list), 24)
        self.assertEqual(action_list[0], (0.25, 1, 4))
        self.assertEqual(action_list[-1], (0.5, 6, 12))
        self.assertEqual(len(action_list), len(set(action_list)))
        self.assertEqual(action_list.count((0.25, 3, 4)), 1)
        split_sets = [set(value["instances"]) for value in protocol["splits"].values()]
        self.assertFalse(split_sets[0] & split_sets[1])
        self.assertFalse(split_sets[0] & split_sets[2])
        self.assertFalse(split_sets[1] & split_sets[2])

    def test_action_seed_shares_budget_prefix_but_isolates_other_axes(self):
        seed = self.module.action_seed("Mk01-1401-0", 0.25, 3, 0)
        self.assertEqual(seed, self.module.action_seed("Mk01-1401-0", 0.25, 3, 0))
        variants = {
            self.module.action_seed("Mk01-1401-1", 0.25, 3, 0),
            self.module.action_seed("Mk01-1401-0", 0.5, 3, 0),
            self.module.action_seed("Mk01-1401-0", 0.25, 4, 0),
            self.module.action_seed("Mk01-1401-0", 0.25, 3, 1),
        }
        self.assertNotIn(seed, variants)
        self.assertEqual(len(variants), 4)

    def test_state_features_are_twelve_finite_predecision_values(self):
        data = self.module.old.load_case("Mk01")
        chromosome = self.module.old.hybrid_population(
            data, 10, len(data.agv.speeds), random.Random(91)
        ).chromosomes[0]
        schedule = self.module.old.decode_static(data, chromosome)
        values = self.module.state_features(data, schedule, 7, 20)
        self.assertEqual(len(values), 12)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertAlmostEqual(values[0], 7 / 19)

    def test_smoke_run_writes_complete_non_overwriting_grid(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            output = Path(temporary) / "smoke"
            self.module.run(output, "development", smoke=True)
            with (output / "trials.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3 * 24 * 4)
            self.assertTrue(all(int(row["decodes"]) <= int(row["b"]) for row in rows))
            self.assertEqual(len({row["state"] for row in rows}), 3)
            self.assertTrue((output / "raw_trials.jsonl.gz").exists())
            self.assertTrue((output / "states.json").exists())
            with self.assertRaises(FileExistsError):
                self.module.run(output, "development", smoke=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
