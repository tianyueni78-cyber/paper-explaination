"""运行 LLM 离线邻域 MVP；不调用在线 LLM 或修改 baseline。"""

from __future__ import annotations

import argparse
import ast
import csv
import inspect
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_operator_mvp.evaluation import run_case, validate_operator
from llm_operator_mvp.operators import CANDIDATES
from python_baseline.dfjspt.data import load_dynamic_experiment_input
from python_baseline.dfjspt.decoder import decode_static
from python_baseline.dfjspt.dynamic import DynamicEvent, _decode_dynamic
from python_baseline.dfjspt.initialization import hybrid_population


DATA = ROOT / "python_baseline" / "data"


def _load(instance: str):
    return load_dynamic_experiment_input(
        DATA / "brandimarte" / f"{instance}.fjs",
        DATA / "resources" / "static_algorithm_comparison.json",
        DATA / "resources" / "dynamic_event_profiles.json",
        "order_cancellation",
    )


def _candidate_records(data, event):
    base = hybrid_population(
        data, 10, len(data.agv.speeds), random.Random(0)
    ).chromosomes[0]
    original = decode_static(data, base)
    records = []
    for item in CANDIDATES:
        code = inspect.getsource(item.operator)
        result = validate_operator(
            item.operator, data, base, seed=item.seed, timeout_seconds=5
        )
        validation = {
            "syntax_import": False,
            "interface": len(inspect.signature(item.operator).parameters) == 3,
            "chromosome": result["status"] == "valid",
            "decode": False,
            "status": result["status"],
            "error": result.get("error", ""),
        }
        try:
            ast.parse(code)
            validation["syntax_import"] = True
            if validation["chromosome"]:
                _decode_dynamic(data, result["chromosome"], original, event)
                validation["decode"] = True
                validation["status"] = "valid"
        except Exception as error:
            validation["status"] = "failed"
            validation["error"] = f"{type(error).__name__}: {error}"
        records.append({
            "id": item.candidate_id,
            "code": code,
            "generation_operator": item.generation_operator,
            "parents": list(item.parents),
            "model": item.model,
            "seed": item.seed,
            "validation_result": validation,
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 离线邻域算子 MVP")
    parser.add_argument("--instance", action="append", choices=("Mk02", "Mk07"))
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--decode-budget", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "llm_operator_mvp_results.csv"
    )
    parser.add_argument(
        "--candidates-output", type=Path,
        default=ROOT / "results" / "llm_operator_candidates.json",
    )
    args = parser.parse_args()
    instances = args.instance or ["Mk02", "Mk07"]
    seeds = args.seed or [11, 22, 33, 44, 55]
    event = DynamicEvent("order_cancellation", 50, 2)

    first_data = _load(instances[0])
    records = _candidate_records(first_data, event)
    args.candidates_output.parent.mkdir(parents=True, exist_ok=True)
    args.candidates_output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), "utf-8"
    )

    rows = []
    for instance in instances:
        data = _load(instance)
        for seed in seeds:
            print(f"run {instance}, seed={seed}, budget={args.decode_budget}", flush=True)
            rows.extend(run_case(
                data, event, instance=instance, seed=seed,
                decode_budget=args.decode_budget,
            ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
