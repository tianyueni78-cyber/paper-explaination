"""固定K/N下的反馈预算生命周期实验；A0只读，输出拒绝覆盖。"""
import argparse
import gzip
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "KNb深挖/scripts"))
import probe_knb as previous

old = previous.old


def repository_root():
    return Path(__file__).resolve().parents[4]


def checkpoints(generations):
    if generations != 20:
        raise ValueError("当前冻结协议只支持20代")
    return [0, 2, 4, 8, 12, 16, 19]


def stage_of(generation, generations):
    progress = generation / generations
    if progress < 0.2:
        return "early"
    if progress < 0.7:
        return "middle"
    return "late"


def _seed_means(rows, metric):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["instance"], row["seed"])].append(row[metric])
    return {key: fmean(values) for key, values in grouped.items()}


def summarize_paired(rows, metric, bootstrap_samples=10000, bootstrap_seed=20260907):
    seed_means = _seed_means(rows, metric)
    by_instance = defaultdict(list)
    for (instance, _), value in seed_means.items():
        by_instance[instance].append(value)
    per_instance = {instance: fmean(values) for instance, values in sorted(by_instance.items())}
    instances = sorted(by_instance)
    rng = random.Random(bootstrap_seed)
    sampled = []
    for _ in range(bootstrap_samples):
        outer = [rng.choice(instances) for _ in instances]
        values = []
        for instance in outer:
            seeds = by_instance[instance]
            values.append(fmean(rng.choice(seeds) for _ in seeds))
        sampled.append(fmean(values))
    sampled.sort()
    low = sampled[int(0.025 * bootstrap_samples)]
    high = sampled[min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))]
    return {
        "mean_difference": fmean(per_instance.values()),
        "ci95": [low, high],
        "positive_instances": sum(value > 0 for value in per_instance.values()),
        "zero_instances": sum(value == 0 for value in per_instance.values()),
        "instances": len(per_instance),
        "per_instance": per_instance,
    }


def validate_protocol(protocol):
    expected = {
        "version": "lifecycle-local-v1",
        "scope": "fresh-seed-lifecycle-validation-before-integrated-run",
        "instances": [f"Mk{i:02}" for i in range(1, 7)],
        "optimizer_seeds": [404, 505, 606, 707, 808, 909, 1001, 1102, 1203, 1304],
        "local_repeats": [0, 1, 2],
        "population_size": 20,
        "generations": 20,
        "checkpoints": [0, 2, 4, 8, 12, 16, 19],
        "early_fraction": 0.2,
        "late_fraction": 0.7,
        "policies": ["fixed", "feedback"],
        "fixed_action": {"fraction": 0.25, "neighborhood": "N3"},
        "local_decode_budget": 96,
        "primary_metric": "hv_gain",
        "bootstrap_samples": 10000,
        "bootstrap_seed": 20260907,
        "minimum_positive_instances": 4,
        "timeout_seconds": 1800,
        "integrated_run_requires_local_gate": True,
    }
    if protocol != expected:
        raise ValueError("协议与实现不一致，禁止静默改变实验")


def capture(data, seed, generations=20, population_size=20):
    wanted = checkpoints(generations)
    positions = {g: random.Random(old.seed_of("lifecycle-capture", seed, g)).randrange(population_size)
                 for g in wanted}
    snapshots = []
    count = 0
    original = old.q.apply_neighborhood

    def observe(d, chromosome, neighborhood, rng):
        nonlocal count
        generation, index = divmod(count, population_size)
        if positions.get(generation) == index:
            snapshots.append((generation, index, chromosome))
        count += 1
        return original(d, chromosome, neighborhood, rng)

    old.q.apply_neighborhood = observe
    try:
        with old.Meter() as meter:
            old.q.run_qnsga2(data, population_size=population_size,
                             generations=generations, seed=seed)
    finally:
        old.q.apply_neighborhood = original
    if len(snapshots) != len(wanted) or count != generations * population_size:
        raise RuntimeError("生命周期快照捕获不完整")
    return snapshots, meter.count


def load_protocol():
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    validate_protocol(protocol)
    return protocol


def run(output):
    protocol = load_protocol()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source_manifest_path = ROOT.parent / "KNb深挖/runs/budget-corrected/manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_hashes = source_manifest["source_hashes"]
    current = {name: old.sha(old.BASE / name) for name in source_hashes}
    if current != source_hashes:
        raise ValueError("冻结A0来源发生变化")
    selection_path = ROOT.parent / "KNb深挖/runs/budget-corrected/selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["fixed_action"] != 2:
        raise ValueError("固定强基线不再是K=25%、N3")
    manifest = {
        "status": "running",
        "protocol_sha256": old.sha(ROOT / "protocol.json"),
        "script_sha256": old.sha(Path(__file__)),
        "source_commit": source_manifest["source_commit"],
        "source_hashes": source_hashes,
        "source_manifest_sha256": old.sha(source_manifest_path),
        "selection_sha256": old.sha(selection_path),
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root(), text=True).strip(),
        "acquisition_decodes": 0,
        "search_decodes": 0,
        "trajectories": 0,
    }
    old.dump(output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        with gzip.open(output / "trajectories.jsonl.gz", "wt", encoding="utf-8") as raw:
            for instance in protocol["instances"]:
                data = old.load_case(instance)
                for optimizer_seed in protocol["optimizer_seeds"]:
                    snapshots, cost = capture(data, optimizer_seed, protocol["generations"],
                                              protocol["population_size"])
                    manifest["acquisition_decodes"] += cost
                    for generation, index, parent in snapshots:
                        schedule = old.decode_static(data, parent)
                        manifest["acquisition_decodes"] += 1
                        identity = {
                            "instance": instance,
                            "seed": optimizer_seed,
                            "generation": generation,
                            "stage": stage_of(generation, protocol["generations"]),
                            "population_index": index,
                            "parent": asdict(parent),
                            "base_objective": [schedule.makespan, schedule.machine_energy],
                        }
                        for rep in protocol["local_repeats"]:
                            local_seed = old.seed_of("lifecycle", instance, optimizer_seed, generation, rep)
                            for policy in protocol["policies"]:
                                if time.perf_counter() - started > protocol["timeout_seconds"]:
                                    raise TimeoutError("生命周期实验超过30分钟上限")
                                result = previous.walk(data, parent, schedule, policy,
                                                       selection["fixed_action"],
                                                       protocol["local_decode_budget"], local_seed)
                                row = dict(**identity, rep=rep, local_seed=local_seed, **result)
                                raw.write(json.dumps(row) + "\n")
                                manifest["search_decodes"] += result["decodes"]
                                manifest["trajectories"] += 1
                    print(instance, optimizer_seed, manifest["trajectories"], flush=True)
        if {name: old.sha(old.BASE / name) for name in source_hashes} != source_hashes:
            raise ValueError("运行期间冻结A0来源发生变化")
        manifest.update(status="completed", source_unchanged=True)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["seconds"] = time.perf_counter() - started
        raw_path = output / "trajectories.jsonl.gz"
        if raw_path.exists():
            manifest["raw_sha256"] = old.sha(raw_path)
        old.dump(output / "manifest.json", manifest)


def metrics(row):
    base = row["base_objective"]
    objectives = [candidate["objective"] for step in row["trace"] for candidate in step["candidates"]]
    best_makespan = min([base[0], *(value[0] for value in objectives)])
    best_tec = min([base[1], *(value[1] for value in objectives)])
    return {
        "hv_gain": row["gain"],
        "hv_per_decode": row["gain"] / row["decodes"] if row["decodes"] else 0.0,
        "makespan_improvement": (base[0] - best_makespan) / base[0],
        "tec_improvement": (base[1] - best_tec) / base[1],
        "decodes": row["decodes"],
        "wall_seconds": row["wall_seconds"],
    }


def paired_differences(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["instance"], row["seed"], row["generation"], row["rep"])
        if row["policy"] in grouped[key]:
            raise ValueError("配对键重复")
        grouped[key][row["policy"]] = row
    result = []
    for key, policies in sorted(grouped.items()):
        if set(policies) != {"fixed", "feedback"}:
            raise ValueError("反馈与固定配对缺失")
        feedback, fixed = metrics(policies["feedback"]), metrics(policies["fixed"])
        item = {
            "instance": key[0], "seed": key[1], "generation": key[2], "rep": key[3],
            "stage": policies["fixed"]["stage"],
        }
        for metric in feedback:
            item[metric] = feedback[metric] - fixed[metric]
        result.append(item)
    return result


def analyze(run_folder):
    protocol = load_protocol()
    manifest = json.loads((run_folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("未完成运行不能分析")
    raw_path = run_folder / "trajectories.jsonl.gz"
    if old.sha(raw_path) != manifest["raw_sha256"]:
        raise ValueError("原始数据哈希不一致")
    with gzip.open(raw_path, "rt", encoding="utf-8") as raw:
        rows = [json.loads(line) for line in raw]
    if len(rows) != manifest["trajectories"]:
        raise ValueError("轨迹数量与manifest不一致")
    pairs = paired_differences(rows)
    summaries = {}
    for stage in ("early", "middle", "late"):
        subset = [row for row in pairs if row["stage"] == stage]
        summaries[stage] = {
            metric: summarize_paired(subset, metric, protocol["bootstrap_samples"],
                                     protocol["bootstrap_seed"] + index)
            for index, metric in enumerate(("hv_gain", "hv_per_decode", "makespan_improvement",
                                            "tec_improvement", "decodes", "wall_seconds"))
        }
    by_key = defaultdict(dict)
    for row in pairs:
        by_key[(row["instance"], row["seed"], row["rep"])].setdefault(row["stage"], []).append(row["hv_gain"])
    interactions = []
    for (instance, seed, rep), stages in by_key.items():
        later = stages.get("middle", []) + stages.get("late", [])
        if not stages.get("early") or not later:
            raise ValueError("生命周期阶段数据缺失")
        interactions.append({"instance": instance, "seed": seed, "rep": rep,
                             "delta": fmean(stages["early"]) - fmean(later)})
    interaction = summarize_paired(interactions, "delta", protocol["bootstrap_samples"],
                                   protocol["bootstrap_seed"] + 99)
    early = summaries["early"]
    checks = {
        "early_hv_mean_positive": early["hv_gain"]["mean_difference"] > 0,
        "early_hv_positive_instances": early["hv_gain"]["positive_instances"] >= protocol["minimum_positive_instances"],
        "early_hv_ci_above_zero": early["hv_gain"]["ci95"][0] > 0,
        "early_efficiency_positive": early["hv_per_decode"]["mean_difference"] > 0,
        "early_efficiency_positive_instances": early["hv_per_decode"]["positive_instances"] >= protocol["minimum_positive_instances"],
        "makespan_no_clear_systematic_harm": early["makespan_improvement"]["ci95"][1] >= 0,
        "tec_no_clear_systematic_harm": early["tec_improvement"]["ci95"][1] >= 0,
        "lifecycle_interaction_positive": interaction["mean_difference"] > 0,
        "lifecycle_interaction_ci_above_zero": interaction["ci95"][0] > 0,
    }
    result = {
        "status": "analyzed",
        "claim_scope": "local_lifecycle_gate_only",
        "source_commit": manifest["source_commit"],
        "pairs": len(pairs),
        "summaries": summaries,
        "early_minus_later_hv": interaction,
        "gate_checks": checks,
        "local_gate_passed": all(checks.values()),
        "next_action": "run_integrated_20_percent_rule" if all(checks.values()) else "stop_before_integrated_run",
    }
    old.dump(run_folder / "analysis.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "run":
        run(args.output)
    else:
        analyze(args.output)


if __name__ == "__main__":
    main()
