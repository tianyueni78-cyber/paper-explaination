"""第三轮开发集单项与交互实验入口。"""

import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from contract_v3 import validate_protocol
from runner_v3 import ROOT, old, walk


def configurations(stage):
    fixed = ("000_fixed", (False, False, False), "fixed")
    if stage == "N":
        return [fixed, ("010_rule", (False, True, False), "rule"), ("010_q", (False, True, False), "q")]
    if stage == "K":
        return [fixed, ("100_rule", (True, False, False), "rule"), ("100_q", (True, False, False), "q")]
    if stage == "b":
        return [fixed, ("001_rule", (False, False, True), "rule"), ("001_q", (False, False, True), "q")]
    if stage == "Q":
        return [fixed, ("111_rule", (True, True, True), "rule"), ("111_q", (True, True, True), "q")]
    if stage == "interaction":
        output = []
        for k in (False, True):
            for n in (False, True):
                for b in (False, True):
                    name = f"{int(k)}{int(n)}{int(b)}"
                    output.append((f"{name}_{'fixed' if name == '000' else 'q'}", (k, n, b), "fixed" if name == "000" else "q"))
        return output
    raise ValueError(stage)


def _dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["configuration"]].append(row)
    return {
        name: {
            "trajectories": len(records),
            "mean_gain": fmean(record["gain"] for record in records),
            "mean_decodes": fmean(record["decodes"] for record in records),
            "mean_q_updates": fmean(record["q_updates"] for record in records),
            "zero_decode_decisions": sum(
                step["decodes"] == 0 for record in records for step in record["trace"]
            ),
        }
        for name, records in sorted(groups.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("N", "K", "b", "Q", "interaction"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    validate_protocol(protocol)
    source = ROOT.parent / "KNb深挖" / "runs" / "budget-corrected"
    states_path = source / "states.json"
    states = [
        state for state in json.loads(states_path.read_text(encoding="utf-8"))
        if state["seed"] == protocol["development_parent_seed"]
    ]
    if len(states) != 18 or any(state["instance"] not in protocol["instances"] for state in states):
        raise ValueError("开发父解集合不是冻结的6实例×3阶段")
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    hashes = source_manifest["source_hashes"]
    if hashes != {name: old.sha(old.BASE / name) for name in hashes}:
        raise ValueError("冻结A0来源已经变化")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "stage": args.stage,
        "performance_claims": False,
        "validation_data_opened": False,
        "protocol_sha256": old.sha(ROOT / "protocol.json"),
        "states_sha256": old.sha(states_path),
        "scripts": {path.name: old.sha(path) for path in Path(__file__).parent.glob("*.py")},
        "source_commit": source_manifest["source_commit"],
        "source_hashes": hashes,
        "parents": len(states),
        "repeats": protocol["development_repeats"],
        "acquisition_decodes": 0,
        "search_decodes": 0,
        "trajectories": 0,
    }
    _dump(args.output / "manifest.json", manifest)
    rows = []
    started = time.perf_counter()
    try:
        with gzip.open(args.output / "results.jsonl.gz", "wt", encoding="utf-8") as raw:
            for state in states:
                if time.perf_counter() - started > protocol["timeout_seconds"]:
                    raise TimeoutError("第三轮开发实验超过冻结时间上限")
                data = old.load_case(state["instance"])
                parent = old.Chromosome(**{key: tuple(value) for key, value in state["chromosome"].items()})
                schedule = old.decode_static(data, parent)
                manifest["acquisition_decodes"] += 1
                for repeat in protocol["development_repeats"]:
                    seed = old.seed_of("knb3-development", state["state"], repeat)
                    for name, modules, controller in configurations(args.stage):
                        result = walk(data, parent, schedule, seed, protocol["development_budget"], protocol, modules, controller)
                        row = {
                            "state": state["state"],
                            "instance": state["instance"],
                            "generation": state["generation"],
                            "repeat": repeat,
                            "configuration": name,
                            "modules": modules,
                            "controller": controller,
                            **result,
                        }
                        raw.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows.append(row)
                        manifest["search_decodes"] += result["decodes"]
                        manifest["trajectories"] += 1
                raw.flush()
                print(state["state"], manifest["trajectories"], flush=True)
        _dump(args.output / "summary.json", _summarize(rows))
        if hashes != {name: old.sha(old.BASE / name) for name in hashes}:
            raise ValueError("运行过程中冻结A0来源发生变化")
        manifest.update(status="completed", source_unchanged=True)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["seconds"] = time.perf_counter() - started
        for name in ("results.jsonl.gz", "summary.json"):
            if (args.output / name).exists():
                manifest[f"{name}_sha256"] = old.sha(args.output / name)
        _dump(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
