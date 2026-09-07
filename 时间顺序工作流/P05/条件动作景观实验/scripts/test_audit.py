import copy
import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("audit.py")
ACTIONS = [(fraction, n, b) for fraction in (0.25, 0.5) for n in range(1, 7) for b in (4, 12)]


def fixtures():
    rows = []
    raw = []
    for fraction, n, b in ACTIONS:
        for repeat in range(4):
            item = {
                "state": "s1",
                "instance": "Mk01",
                "fraction": fraction,
                "n": n,
                "b": b,
                "repeat": repeat,
                "decodes": b,
            }
            rows.append(item)
            raw.append({**item, "candidates": []})
    manifest = {
        "status": "completed",
        "states": 1,
        "trials": len(rows),
        "source_hashes": {"a.py": "same"},
        "source_unchanged": True,
    }
    return manifest, rows, raw


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("conditional_audit", PATH)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_complete_grid_passes(self):
        manifest, rows, raw = fixtures()
        result = self.module.audit_records(manifest, rows, raw, {"a.py": "same"})
        self.assertTrue(result["passed"])

    def test_missing_or_duplicate_pair_fails(self):
        manifest, rows, raw = fixtures()
        missing = self.module.audit_records(manifest, rows[:-1], raw[:-1], {"a.py": "same"})
        self.assertFalse(missing["passed"])
        duplicated = self.module.audit_records(
            manifest, rows + [copy.deepcopy(rows[0])], raw + [copy.deepcopy(raw[0])], {"a.py": "same"}
        )
        self.assertFalse(duplicated["passed"])

    def test_over_budget_and_source_change_fail(self):
        manifest, rows, raw = fixtures()
        rows[0]["decodes"] = 5
        raw[0]["decodes"] = 5
        result = self.module.audit_records(manifest, rows, raw, {"a.py": "changed"})
        self.assertFalse(result["passed"])
        self.assertGreater(result["over_budget"], 0)
        self.assertFalse(result["source_hashes_match"])

    def test_budget_prefix_mismatch_fails(self):
        manifest, rows, raw = fixtures()
        pair = [
            item for item in raw
            if item["fraction"] == 0.25 and item["n"] == 1 and item["repeat"] == 0
        ]
        next(item for item in pair if item["b"] == 4)["candidates"] = [
            {"chromosome": {"os": [1]}, "objective": [1, 1], "decode_index": 1}
        ]
        result = self.module.audit_records(manifest, rows, raw, {"a.py": "same"})
        self.assertFalse(result["passed"])
        self.assertEqual(result["prefix_mismatches"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
