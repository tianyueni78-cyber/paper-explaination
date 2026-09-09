import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import runner_v3
from runner_v3 import ROOT, choose_joint, old, walk
from mechanisms_v3 import FactorizedQ


class RunnerV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT.parent / "KNb深挖" / "runs" / "budget-corrected" / "states.json"
        state = json.loads(source.read_text(encoding="utf-8"))[0]
        cls.data = old.load_case(state["instance"])
        cls.parent = old.Chromosome(**{key: tuple(value) for key, value in state["chromosome"].items()})
        cls.schedule = old.decode_static(cls.data, cls.parent)
        cls.protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))

    def test_joint_choice_covers_unvisited_factors_before_confidence_gate(self):
        actions = [(0.25, 2, 4), (0.5, 4, 8)]
        selected, mode = choose_joint(FactorizedQ(), "s", actions, {"K": set(), "N": set(), "b": set()}, 1e-6)
        self.assertEqual((selected, mode), ((0.25, 2, 4), "cold_start"))
        covered = {"K": {0.25}, "N": {2}, "b": {4}}
        selected, mode = choose_joint(FactorizedQ(), "s", actions, covered, 1e-6)
        self.assertEqual((selected, mode), ((0.5, 4, 8), "cold_start"))

    def test_walk_is_repeatable_and_charges_every_decode(self):
        first = walk(self.data, self.parent, self.schedule, seed=91, budget=24, protocol=self.protocol)
        second = walk(self.data, self.parent, self.schedule, seed=91, budget=24, protocol=self.protocol)
        self.assertGreater(first["decodes"], 0)
        self.assertLessEqual(first["decodes"], 24)
        self.assertEqual(first["decodes"], sum(row["decodes"] for row in first["trace"]))
        self.assertEqual(first["trace"], second["trace"])
        self.assertEqual(first["q_terms"], second["q_terms"])
        self.assertTrue(all(row["actual_cap"] <= row["action"][2] for row in first["trace"]))
        self.assertEqual(first["q_updates"], sum(row["decodes"] > 0 for row in first["trace"]))

    def test_unchanged_parent_reuses_action_pre_context(self):
        original = runner_v3.region_context
        with patch.object(runner_v3, "region_context", wraps=original) as measured:
            result = walk(self.data, self.parent, self.schedule, seed=91, budget=24, protocol=self.protocol)
        self.assertLessEqual(measured.call_count, len(result["trace"]) + 1)

    def test_disabled_modules_are_exact_fixed_control(self):
        with patch.object(runner_v3, "region_context", side_effect=AssertionError("fixed不得计算创新机会特征")):
            result = walk(
                self.data,
                self.parent,
                self.schedule,
                seed=91,
                budget=12,
                protocol=self.protocol,
                modules=(False, False, False),
                controller="fixed",
            )
        self.assertTrue(all(tuple(row["action"]) == (0.25, 3, 4) for row in result["trace"]))
        self.assertEqual(result["q_updates"], 0)

    def test_rule_controller_uses_enabled_modules_without_q_updates(self):
        result = walk(
            self.data,
            self.parent,
            self.schedule,
            seed=91,
            budget=12,
            protocol=self.protocol,
            modules=(True, True, True),
            controller="rule",
        )
        self.assertEqual(result["q_updates"], 0)
        self.assertTrue(all(row["mode"] == "rule" for row in result["trace"]))
        self.assertEqual(result["decodes"], 12)

    def test_smoke_command_writes_completed_auditable_manifest(self):
        directory = ROOT / "_test_smoke_output"
        if directory.exists():
            shutil.rmtree(directory)
        output = directory / "smoke"
        try:
            with patch.object(runner_v3.sys, "argv", ["runner_v3", "--stage", "smoke", "--output", str(output)]):
                runner_v3.main()
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertFalse(manifest["performance_claims"])
            self.assertTrue(manifest["source_unchanged"])
            self.assertEqual(manifest["search_decodes"], result["decodes"])
        finally:
            if directory.exists():
                shutil.rmtree(directory)


if __name__ == "__main__":
    unittest.main()
