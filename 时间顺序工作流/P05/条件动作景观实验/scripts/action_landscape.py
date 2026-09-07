"""条件动作景观采集。"""

from __future__ import annotations

import json
import argparse
import csv
import gzip
import math
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
P05 = ROOT.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(P05 / "KNb深挖" / "scripts"))
sys.path.insert(0, str(P05 / "S0" / "scripts"))
import probe as old
import probe_knb as deep


def load_protocol() -> dict:
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict) -> None:
    if protocol["fractions"] != [0.25, 0.5]:
        raise ValueError("K比例必须固定为0.25和0.5")
    if protocol["neighborhoods"] != [1, 2, 3, 4, 5, 6]:
        raise ValueError("邻域必须固定为N1至N6")
    if protocol["budgets"] != [4, 12] or protocol["repeats"] != [0, 1, 2, 3]:
        raise ValueError("预算或重复与冻结协议不一致")
    split_sets = [set(value["instances"]) for value in protocol["splits"].values()]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("开发、验证和封存实例必须互斥")


def actions(protocol: dict) -> list[tuple[float, int, int]]:
    return [
        (fraction, neighborhood, budget)
        for fraction in protocol["fractions"]
        for neighborhood in protocol["neighborhoods"]
        for budget in protocol["budgets"]
    ]


def action_seed(state_id: str, fraction: float, neighborhood: int, repeat: int) -> int:
    return old.seed_of("conditional-landscape", state_id, fraction, neighborhood, repeat)


def _top_share(values: list[float], fraction: float = 0.25) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    count = max(1, math.ceil(len(values) * fraction))
    return sum(sorted(values, reverse=True)[:count]) / total


def state_features(data, schedule, generation: int, generations: int) -> list[float]:
    machine_scores, agv_scores, load_features = old.scores(data, schedule)
    waits, _, details = deep.wait_features(data, schedule)
    operation_count = data.instance.operation_count
    scale = max(schedule.makespan * operation_count, 1.0)
    mean_wait = fmean(waits) if waits else 0.0
    a25_count = max(1, math.ceil(operation_count * 0.25))
    a25 = sorted(range(operation_count), key=lambda i: (-agv_scores[i], i))[:a25_count]
    operations = [operation for job in data.instance.jobs for operation in job.operations]
    flexible_fraction = sum(len(operations[index].options) > 1 for index in a25) / len(a25)
    return [
        generation / max(generations - 1, 1),
        *load_features,
        sum(row["transport_wait"] for row in details) / scale,
        sum(row["machine_wait"] for row in details) / scale,
        max(waits, default=0.0) / (mean_wait or 1.0),
        _top_share(machine_scores),
        _top_share(agv_scores),
        _top_share(waits),
        flexible_fraction,
    ]


def capture(data, seed: int, generations: int, population_size: int):
    positions = {
        generation: random.Random(
            old.seed_of("conditional-capture", seed, generation)
        ).randrange(population_size)
        for generation in range(generations)
    }
    snapshots = []
    calls = 0
    original = old.q.apply_neighborhood

    def observe(current_data, chromosome, neighborhood, rng):
        nonlocal calls
        generation, position = divmod(calls, population_size)
        if positions[generation] == position:
            snapshots.append((generation, position, chromosome))
        calls += 1
        return original(current_data, chromosome, neighborhood, rng)

    old.q.apply_neighborhood = observe
    try:
        with old.Meter() as meter:
            result = old.q.run_qnsga2(
                data,
                population_size=population_size,
                generations=generations,
                seed=seed,
            )
    finally:
        old.q.apply_neighborhood = original
    if calls != generations * population_size or len(snapshots) != generations:
        raise RuntimeError("A0父解采集数量与协议不一致")
    return snapshots, meter.count, result


def _endpoint_improvements(base: tuple[float, float], candidates: list[dict]) -> tuple[float, float]:
    if not candidates:
        return 0.0, 0.0
    return (
        max(0.0, (base[0] - min(row["objective"][0] for row in candidates)) / base[0]),
        max(0.0, (base[1] - min(row["objective"][1] for row in candidates)) / base[1]),
    )


def run(output: Path, split: str, smoke: bool = False) -> None:
    protocol = load_protocol()
    if split not in {"development", "validation"}:
        raise ValueError("只能运行development或validation")
    output.mkdir(parents=True, exist_ok=False)
    split_config = protocol["splits"][split]
    instances = split_config["instances"][:1] if smoke else split_config["instances"]
    optimizer_seeds = split_config["optimizer_seeds"][:1] if smoke else split_config["optimizer_seeds"]
    selected_generations = {0, 10, 19} if smoke else set(range(protocol["generations"]))
    baseline_sources = list((old.BASE / "paper_static_baseline" / "dfjspt").glob("*.py"))
    baseline_sources += [
        old.BASE / "paper_static_baseline" / "data" / "resources" / "static_algorithm_comparison.json",
        *[
            old.BASE / "paper_static_baseline" / "data" / "brandimarte" / f"{name}.fjs"
            for name in instances
        ],
    ]
    source_hashes = {str(path.relative_to(old.BASE)): old.sha(path) for path in baseline_sources}
    command = " ".join(sys.argv)
    manifest = {
        "status": "running",
        "version": protocol["version"],
        "split": split,
        "smoke": smoke,
        "instances": instances,
        "optimizer_seeds": optimizer_seeds,
        "generations": sorted(selected_generations),
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "baseline_repo": str(old.BASE),
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=old.BASE, text=True
        ).strip(),
        "source_hashes": source_hashes,
        "protocol_hash": old.sha(ROOT / "protocol.json"),
        "script_hash": old.sha(Path(__file__)),
        "acquisition_decodes": 0,
        "trial_decodes": 0,
        "states": 0,
        "trials": 0,
    }
    old.dump(output / "manifest.json", manifest)
    states = []
    rows = []
    started = time.perf_counter()
    try:
        with gzip.open(output / "raw_trials.jsonl.gz", "wt", encoding="utf-8") as raw:
            for instance in instances:
                data = old.load_case(instance)
                for optimizer_seed in optimizer_seeds:
                    snapshots, cost, _ = capture(
                        data,
                        optimizer_seed,
                        protocol["generations"],
                        protocol["population_size"],
                    )
                    manifest["acquisition_decodes"] += cost
                    for generation, population_index, parent in snapshots:
                        if generation not in selected_generations:
                            continue
                        if time.perf_counter() - started > protocol["timeout_seconds"]:
                            raise TimeoutError("达到冻结的1,800秒硬超时")
                        schedule = old.decode_static(data, parent)
                        manifest["acquisition_decodes"] += 1
                        state_id = f"{instance}-{optimizer_seed}-{generation}"
                        feature_values = state_features(
                            data, schedule, generation, protocol["generations"]
                        )
                        states.append(
                            {
                                "state": state_id,
                                "instance": instance,
                                "optimizer_seed": optimizer_seed,
                                "generation": generation,
                                "population_index": population_index,
                                "features": feature_values,
                                "chromosome": asdict(parent),
                                "base_objective": [schedule.makespan, schedule.machine_energy],
                            }
                        )
                        machine_scores, agv_scores, _ = old.scores(data, schedule)
                        del machine_scores
                        for fraction, neighborhood, budget in actions(protocol):
                            k = math.ceil(parent.operation_count * fraction)
                            for repeat in protocol["repeats"]:
                                stream = action_seed(
                                    state_id, fraction, neighborhood, repeat
                                )
                                region_started = time.perf_counter()
                                selected = old.region(
                                    [], agv_scores, k, protocol["region_group"], stream
                                )
                                region_seconds = time.perf_counter() - region_started
                                result = old.trial(
                                    data,
                                    parent,
                                    schedule,
                                    selected,
                                    neighborhood - 1,
                                    budget,
                                    stream,
                                )
                                base = (schedule.makespan, schedule.machine_energy)
                                makespan_gain, tec_gain = _endpoint_improvements(
                                    base, result["candidates"]
                                )
                                item = {
                                    "state": state_id,
                                    "instance": instance,
                                    "optimizer_seed": optimizer_seed,
                                    "generation": generation,
                                    "population_index": population_index,
                                    "fraction": fraction,
                                    "n": neighborhood,
                                    "b": budget,
                                    "repeat": repeat,
                                    "action": f"K{fraction}-N{neighborhood}-b{budget}",
                                    "rng_seed": stream,
                                    "region_seconds": region_seconds,
                                    "makespan_improvement": makespan_gain,
                                    "tec_improvement": tec_gain,
                                    "hv_gain_per_allowed_decode": result["gain"] / budget,
                                    **result,
                                }
                                raw.write(
                                    json.dumps(
                                        {**item, "region": sorted(selected)},
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                                rows.append(
                                    {key: value for key, value in item.items() if key != "candidates"}
                                )
                                manifest["trial_decodes"] += result["decodes"]
                        print(
                            split,
                            state_id,
                            len(rows),
                            round(time.perf_counter() - started, 1),
                            flush=True,
                        )
        baseline_after = {
            str(path.relative_to(old.BASE)): old.sha(path) for path in baseline_sources
        }
        if source_hashes != baseline_after:
            raise RuntimeError("冻结A0来源在运行期间发生变化")
        manifest.update(
            status="completed",
            source_unchanged=True,
            states=len(states),
            trials=len(rows),
            seconds=time.perf_counter() - started,
        )
    except BaseException as error:
        manifest.update(
            status="failed",
            error=repr(error),
            states=len(states),
            trials=len(rows),
            seconds=time.perf_counter() - started,
        )
        raise
    finally:
        old.dump(output / "states.json", states)
        old.csv_write(output / "trials.csv", rows)
        old.dump(output / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.output, args.split, args.smoke)


if __name__ == "__main__":
    main()
