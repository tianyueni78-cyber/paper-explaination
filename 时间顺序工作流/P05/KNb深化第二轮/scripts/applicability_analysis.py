"""在既有探索数据中定位KNb具体机制的条件化适用范围。"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from run_experiments import ROOT, old


METRICS = (
    "hv_gain", "hv_per_decode", "makespan_improvement", "tec_improvement",
    "decodes", "wall_seconds",
)


def objective_metrics(row, base):
    objectives = [tuple(candidate["objective"])
                  for step in row["trace"] for candidate in step["candidates"]]
    best_makespan = min([base[0], *(value[0] for value in objectives)])
    best_tec = min([base[1], *(value[1] for value in objectives)])
    decodes = row["decodes"]
    return {
        "hv_gain": row["gain"],
        "hv_per_decode": row["gain"] / decodes if decodes else 0.0,
        "makespan_improvement": (base[0] - best_makespan) / base[0],
        "tec_improvement": (base[1] - best_tec) / base[1],
        "decodes": decodes,
        "wall_seconds": row["wall_seconds"],
    }


def paired_rows(rows, left, right, field="config"):
    def selected(value):
        result = {}
        for row in rows:
            actual = tuple(row[field]) if field == "config" else row[field]
            if actual != value:
                continue
            key = (row["state"], row.get("fraction"), row["rep"])
            if key in result:
                raise ValueError("配对键重复")
            result[key] = row
        return result

    left_rows, right_rows = selected(left), selected(right)
    if not left_rows or left_rows.keys() != right_rows.keys():
        raise ValueError("配对样本缺失")
    return [(left_rows[key], right_rows[key]) for key in sorted(left_rows)]


def summarize(pairs, dimension, metrics=METRICS):
    groups = defaultdict(list)
    for context, left, right in pairs:
        groups[str(context.get(dimension))].append((context, left, right))
    result = {}
    for group, values in sorted(groups.items()):
        result[group] = {}
        for metric in metrics:
            differences = [left[metric] - right[metric] for _, left, right in values]
            by_instance = defaultdict(list)
            for context, left, right in values:
                by_instance[context["instance"]].append(left[metric] - right[metric])
            result[group][metric] = {
                "mean_difference": fmean(differences),
                "positive_pairs": sum(value > 0 for value in differences),
                "zero_pairs": sum(value == 0 for value in differences),
                "pairs": len(differences),
                "per_instance": {key: fmean(items) for key, items in sorted(by_instance.items())},
            }
    return result


def load_rows(folder, raw_name="trajectories.jsonl.gz", count_key="trajectories", expected_hash=None):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError(f"未完成运行不能用于适用范围分析：{folder.name}")
    raw_path = folder / raw_name
    expected_hash = expected_hash or manifest["raw_sha256"]
    if old.sha(raw_path) != expected_hash:
        raise ValueError("原始数据哈希不一致")
    with gzip.open(raw_path, "rt", encoding="utf-8") as raw:
        rows = [json.loads(line) for line in raw]
    if len(rows) != manifest[count_key]:
        raise ValueError("轨迹数与manifest不一致")
    return rows, manifest


def base_objectives():
    states_path = ROOT.parent / "KNb深挖/runs/budget-corrected/states.json"
    states = json.loads(states_path.read_text(encoding="utf-8"))
    result = {}
    for state in states:
        data = old.load_case(state["instance"])
        chromosome = old.Chromosome(**{key: tuple(value) for key, value in state["chromosome"].items()})
        schedule = old.decode_static(data, chromosome)
        result[state["state"]] = (schedule.makespan, schedule.machine_energy)
    return result


def metric_pairs(rows, left, right, bases, field="config"):
    result = []
    for left_row, right_row in paired_rows(rows, left, right, field):
        context = {key: left_row.get(key) for key in ("state", "instance", "generation", "fraction")}
        result.append((context, objective_metrics(left_row, bases[left_row["state"]]),
                       objective_metrics(right_row, bases[right_row["state"]])))
    return result


def validate_source(manifests):
    commits = {manifest["source_commit"] for manifest in manifests}
    hashes = [manifest["source_hashes"] for manifest in manifests]
    if len(commits) != 1 or any(value != hashes[0] for value in hashes[1:]):
        raise ValueError("各运行使用的A0来源不一致")
    current = {name: old.sha(old.BASE / name) for name in hashes[0]}
    if current != hashes[0]:
        raise ValueError("当前A0来源与运行记录不一致")
    return commits.pop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    folders = {
        "D3": ROOT.parent / "KNb深挖/runs/budget-corrected",
        "N": ROOT / "runs/N-v1",
        "b": ROOT / "runs/b-v1",
        "interaction": ROOT / "runs/interaction-v1",
    }
    audit = json.loads((folders["D3"] / "audit.json").read_text(encoding="utf-8"))
    loaded = {
        "D3": load_rows(folders["D3"], "D3_raw.jsonl.gz", "D3_trajectories",
                        audit["artifacts"]["D3_raw.jsonl.gz"]),
        **{name: load_rows(folder) for name, folder in folders.items() if name != "D3"},
    }
    source_commit = validate_source([item[1] for item in loaded.values()])
    bases = base_objectives()
    contrasts = {
        "D3_feedback_minus_fixed": metric_pairs(loaded["D3"][0], "feedback", "fixed", bases, "policy"),
        "R2_recent_minus_fixed4": metric_pairs(loaded["b"][0], ("A", "global", "recent"),
                                                ("A", "global", "4"), bases),
        "R2_state_minus_fixedN3": metric_pairs(loaded["N"][0], ("A", "state", "4"),
                                                ("A", "fixed", "4"), bases),
        "R2_state_minus_global": metric_pairs(loaded["N"][0], ("A", "state", "4"),
                                               ("A", "global", "4"), bases),
        "R2_111_minus_000": metric_pairs(loaded["interaction"][0], ("K", "state", "recent"),
                                          ("A", "fixed", "4"), bases),
    }
    output = {
        "status": "exploratory",
        "source_commit": source_commit,
        "diagnostic_parent_decodes": len(bases),
        "metric_direction": {
            "higher_is_better": list(METRICS[:4]),
            "lower_is_better": ["decodes", "wall_seconds"],
        },
        "single_objective_extremes_may_be_different_candidates": True,
        "contrasts": {name: {dimension: summarize(pairs, dimension)
                              for dimension in ("generation", "fraction", "instance")}
                      for name, pairs in contrasts.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
