import random
import json
import subprocess
import sys
import time
import unittest
from pathlib import Path

from python_baseline.dfjspt.data import load_dynamic_experiment_input
from python_baseline.dfjspt.decoder import decode_static
from python_baseline.dfjspt.dynamic import DynamicEvent, _decode_dynamic
from python_baseline.dfjspt.initialization import hybrid_population

from llm_operator_mvp.evaluation import run_case, validate_operator
from llm_operator_mvp.operators import CANDIDATES


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "python_baseline" / "data"


def slow_operator(data, chromosome, rng):
    del data, rng
    time.sleep(1)
    return chromosome


class LLMOperatorMVPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dynamic_experiment_input(
            DATA / "brandimarte" / "Mk02.fjs",
            DATA / "resources" / "static_algorithm_comparison.json",
            DATA / "resources" / "dynamic_event_profiles.json",
            "order_cancellation",
        )
        cls.chromosome = hybrid_population(
            cls.data, 10, len(cls.data.agv.speeds), random.Random(7)
        ).chromosomes[0]

    def test_every_candidate_returns_a_valid_decodable_chromosome(self):
        original = decode_static(self.data, self.chromosome)
        event = DynamicEvent("order_cancellation", 50, 2)
        for index, candidate in enumerate(CANDIDATES):
            with self.subTest(candidate=candidate.candidate_id):
                result = candidate.operator(
                    self.data, self.chromosome, random.Random(index)
                )
                self.assertIsNot(result, self.chromosome)
                result.validate(
                    self.data.instance, self.data.agv.count, len(self.data.agv.speeds)
                )
                schedule = _decode_dynamic(self.data, result, original, event)
                self.assertGreater(schedule.makespan, 0)
                self.assertGreater(schedule.machine_energy, 0)

    def test_timeout_is_reported_without_stopping_validation(self):
        result = validate_operator(
            slow_operator, self.data, self.chromosome, seed=1, timeout_seconds=0.05
        )
        self.assertEqual(result["status"], "timeout")

    def test_run_case_uses_equal_decode_budget_for_all_operators(self):
        rows = run_case(
            self.data,
            DynamicEvent("order_cancellation", 50, 2),
            instance="Mk02",
            seed=7,
            decode_budget=2,
        )
        self.assertEqual(len(rows), 10)
        self.assertEqual({row["source"] for row in rows}, {"baseline", "llm"})
        self.assertTrue(all(row["decode_count"] <= 2 for row in rows))
        self.assertTrue(all(row["decode_budget"] == 2 for row in rows))

    def test_cli_smoke_writes_results_and_candidate_records(self):
        output = ROOT / ".codex_tmp" / "llm_mvp_test.csv"
        candidates = ROOT / ".codex_tmp" / "llm_mvp_candidates_test.json"
        output.unlink(missing_ok=True)
        candidates.unlink(missing_ok=True)
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_llm_operator_mvp.py"),
                "--instance", "Mk02", "--seed", "7", "--decode-budget", "1",
                "--output", str(output), "--candidates-output", str(candidates),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(output.read_text("utf-8").splitlines()), 11)
        records = json.loads(candidates.read_text("utf-8"))
        self.assertEqual(len(records), 4)
        self.assertTrue(all(item["validation_result"]["decode"] for item in records))


if __name__ == "__main__":
    unittest.main()
