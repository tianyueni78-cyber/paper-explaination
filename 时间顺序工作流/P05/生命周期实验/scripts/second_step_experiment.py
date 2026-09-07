"""补采第1代并在共同真实解码前缀下定位0—10%转折。"""
import argparse
import gzip
import json
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from lifecycle_experiment import ROOT, old, summarize_paired
from pointwise_analysis import METRICS, paired_prefix_rows, sign_change_intervals
import probe_knb as previous


EXPECTED = {
    "version": "lifecycle-step2-g1-v1",
    "scope": "exploratory-localization-between-generation-zero-and-two",
    "instances": [f"Mk{i:02}" for i in range(1, 7)],
    "optimizer_seeds": [404, 505, 606, 707, 808, 909, 1001, 1102, 1203, 1304],
    "local_repeats": [0, 1, 2],
    "population_size": 20,
    "generations": 20,
    "target_generation": 1,
    "policies": ["fixed", "feedback"],
    "fixed_action": {"fraction": 0.25, "neighborhood": "N3"},
    "local_decode_budget": 96,
    "common_real_decode_prefix": True,
    "primary_metric": "hv_gain",
    "bootstrap_samples": 10000,
    "bootstrap_seed": 20260917,
    "source_raw_sha256": "c8e0d817882880a85f0140bdccea91e5ec15661e1331abab5d08ddee65afdcee",
    "source_pointwise_sha256": "dd25220457a06dfc325d5a10b0918c76395ffe3ddcb77cc1ed937b4a74fdc62e",
    "select_cutoff": False,
    "timeout_seconds": 1800,
}


def validate_protocol(protocol, partial=False):
    if partial:
        required = {"version": EXPECTED["version"], "generations": 20,
                    "target_generation": 1, "select_cutoff": False}
        if any(protocol.get(key) != value for key, value in required.items()):
            raise ValueError("第二步必须在完整20代语境中仅补第1代，且不得选择切点")
        return
    if protocol != EXPECTED:
        raise ValueError("第二步协议与实现不一致，禁止静默改变实验")


def load_protocol():
    protocol = json.loads((ROOT / "second_step_protocol.json").read_text(encoding="utf-8"))
    validate_protocol(protocol)
    return protocol


def merge_transition_points(prior, generation_one):
    return {key: ({"1": generation_one} | prior)[key]
            for key in sorted([*prior.keys(), "1"], key=int) if key in {"0", "1", "2"}}


def summarize_generation_one(pairs, metrics, bootstrap_samples, bootstrap_seed):
    return {metric: summarize_paired(pairs, metric, bootstrap_samples,
                                     bootstrap_seed + index)
            for index, metric in enumerate(metrics)}


def capture_generation(data, seed, target, generations, population_size):
    position = random.Random(old.seed_of("lifecycle-capture", seed, target)).randrange(population_size)
    snapshot = []
    count = 0
    original = old.q.apply_neighborhood

    def observe(d, chromosome, neighborhood, rng):
        nonlocal count
        generation, index = divmod(count, population_size)
        if generation == target and index == position:
            snapshot.append((index, chromosome))
        count += 1
        return original(d, chromosome, neighborhood, rng)

    old.q.apply_neighborhood = observe
    try:
        with old.Meter() as meter:
            old.q.run_qnsga2(data, population_size=population_size,
                             generations=generations, seed=seed)
    finally:
        old.q.apply_neighborhood = original
    if len(snapshot) != 1 or count != generations * population_size:
        raise RuntimeError("第1代快照捕获不完整")
    return snapshot[0], meter.count


def source_context():
    manifest_path = ROOT.parent / "KNb深挖/runs/budget-corrected/manifest.json"
    selection_path = ROOT.parent / "KNb深挖/runs/budget-corrected/selection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    current = {name: old.sha(old.BASE / name) for name in manifest["source_hashes"]}
    if current != manifest["source_hashes"]:
        raise ValueError("冻结A0来源发生变化")
    if selection["fixed_action"] != 2:
        raise ValueError("固定强基线不再是K=25%、N3")
    return manifest_path, manifest, selection_path, selection


def run(output):
    protocol = load_protocol()
    if output.exists():
        raise FileExistsError(output)
    formal = ROOT / "runs/formal-v1"
    if old.sha(formal / "trajectories.jsonl.gz") != protocol["source_raw_sha256"]:
        raise ValueError("第一步原始轨迹发生变化")
    if old.sha(formal / "pointwise_analysis.json") != protocol["source_pointwise_sha256"]:
        raise ValueError("第一步逐点分析发生变化")
    manifest_path, source, selection_path, selection = source_context()
    output.mkdir(parents=True)
    manifest = {
        "status": "running",
        "protocol_sha256": old.sha(ROOT / "second_step_protocol.json"),
        "script_sha256": old.sha(Path(__file__)),
        "source_commit": source["source_commit"],
        "source_hashes": source["source_hashes"],
        "source_manifest_sha256": old.sha(manifest_path),
        "selection_sha256": old.sha(selection_path),
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4],
            text=True).strip(),
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
                    (index, parent), cost = capture_generation(
                        data, optimizer_seed, protocol["target_generation"],
                        protocol["generations"], protocol["population_size"])
                    manifest["acquisition_decodes"] += cost
                    schedule = old.decode_static(data, parent)
                    manifest["acquisition_decodes"] += 1
                    identity = {
                        "instance": instance,
                        "seed": optimizer_seed,
                        "generation": protocol["target_generation"],
                        "stage": "transition-localization",
                        "population_index": index,
                        "parent": asdict(parent),
                        "base_objective": [schedule.makespan, schedule.machine_energy],
                    }
                    for rep in protocol["local_repeats"]:
                        local_seed = old.seed_of("lifecycle", instance, optimizer_seed,
                                                 protocol["target_generation"], rep)
                        for policy in protocol["policies"]:
                            if time.perf_counter() - started > protocol["timeout_seconds"]:
                                raise TimeoutError("第二步实验超过30分钟上限")
                            result = previous.walk(data, parent, schedule, policy,
                                                   selection["fixed_action"],
                                                   protocol["local_decode_budget"], local_seed)
                            raw.write(json.dumps(dict(**identity, rep=rep,
                                                     local_seed=local_seed, **result)) + "\n")
                            manifest["search_decodes"] += result["decodes"]
                            manifest["trajectories"] += 1
                    print(instance, optimizer_seed, manifest["trajectories"], flush=True)
        if {name: old.sha(old.BASE / name) for name in source["source_hashes"]} != source["source_hashes"]:
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


def analyze(run_folder):
    protocol = load_protocol()
    manifest = json.loads((run_folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("未完成运行不能分析")
    raw_path = run_folder / "trajectories.jsonl.gz"
    if old.sha(raw_path) != manifest["raw_sha256"]:
        raise ValueError("第二步原始轨迹哈希不一致")
    with gzip.open(raw_path, "rt", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    if len(rows) != 360 or len(rows) != manifest["trajectories"]:
        raise ValueError("第二步必须包含360条第1代轨迹")
    pairs = paired_prefix_rows(rows)
    point = {
        "progress": 0.05,
        "pairs": len(pairs),
        "common_decodes": {
            "mean": fmean(row["common_decodes"] for row in pairs),
            "min": min(row["common_decodes"] for row in pairs),
            "max": max(row["common_decodes"] for row in pairs),
            "discarded_fixed": sum(row["discarded_fixed_decodes"] for row in pairs),
            "discarded_feedback": sum(row["discarded_feedback_decodes"] for row in pairs),
        },
        **summarize_generation_one(pairs, METRICS, protocol["bootstrap_samples"],
                                   protocol["bootstrap_seed"]),
    }
    prior_file = ROOT / "runs/formal-v1/pointwise_analysis.json"
    prior = json.loads(prior_file.read_text(encoding="utf-8"))["points"]
    points = merge_transition_points(prior, point)
    result = {
        "status": "exploratory_transition_localization",
        "verification_status": "analyzed",
        "claim_scope": "generation-one-between-preexisting-zero-and-two",
        "source_raw_sha256": manifest["raw_sha256"],
        "source_pointwise_sha256": old.sha(prior_file),
        "new_trajectories": len(rows),
        "paired_prefixes": len(pairs),
        "new_true_decodes": manifest["acquisition_decodes"] + manifest["search_decodes"],
        "metric_direction": "feedback_minus_fixed_higher_is_better",
        "points": points,
        "observed_sign_changes": {
            metric: sign_change_intervals(points, metric) for metric in METRICS
        },
        "cutoff_selected": False,
        "confirmation_claim_allowed": False,
    }
    old.dump(run_folder / "analysis.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output) if args.mode == "run" else analyze(args.output)


if __name__ == "__main__":
    main()
