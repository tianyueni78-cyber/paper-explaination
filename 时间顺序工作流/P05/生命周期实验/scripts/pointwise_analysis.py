"""按共同真实解码前缀生成无预设阶段标签的生命周期曲线。"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from lifecycle_experiment import ROOT, old, summarize_paired


METRICS = ("hv_gain", "hv_per_decode", "makespan_improvement", "tec_improvement")


def objectives_at_prefix(row, budget):
    objectives = []
    for step in row["trace"]:
        start = step["total_decodes"] - step["decodes"]
        for candidate in step["candidates"]:
            if start + candidate["decode_index"] <= budget:
                objectives.append(candidate["objective"])
    return objectives


def prefix_metrics(row, budget):
    base = row["base_objective"]
    objectives = objectives_at_prefix(row, budget)
    points = [(1.0, 1.0)] + [(value[0] / base[0], value[1] / base[1]) for value in objectives]
    gain = max(0.0, old.hv(points) - old.hv([(1.0, 1.0)]))
    best_makespan = min([base[0], *(value[0] for value in objectives)])
    best_tec = min([base[1], *(value[1] for value in objectives)])
    return {
        "hv_gain": gain,
        "hv_per_decode": gain / budget if budget else 0.0,
        "makespan_improvement": (base[0] - best_makespan) / base[0],
        "tec_improvement": (base[1] - best_tec) / base[1],
    }


def paired_prefix_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["instance"], row["seed"], row["generation"], row["rep"])
        if row["policy"] in grouped[key]:
            raise ValueError("配对键重复")
        grouped[key][row["policy"]] = row
    result = []
    for key, policies in sorted(grouped.items()):
        if set(policies) != {"fixed", "feedback"}:
            raise ValueError("反馈与固定策略配对缺失")
        common = min(policies["fixed"]["decodes"], policies["feedback"]["decodes"])
        fixed = prefix_metrics(policies["fixed"], common)
        feedback = prefix_metrics(policies["feedback"], common)
        item = {
            "instance": key[0], "seed": key[1], "generation": key[2], "rep": key[3],
            "common_decodes": common,
            "discarded_fixed_decodes": policies["fixed"]["decodes"] - common,
            "discarded_feedback_decodes": policies["feedback"]["decodes"] - common,
        }
        for metric in METRICS:
            item[metric] = feedback[metric] - fixed[metric]
            item[f"feedback_{metric}"] = feedback[metric]
            item[f"fixed_{metric}"] = fixed[metric]
        result.append(item)
    return result


def sign_change_intervals(points, metric):
    ordered = sorted((int(generation), summary[metric]["mean_difference"])
                     for generation, summary in points.items())
    return [[left[0], right[0]] for left, right in zip(ordered, ordered[1:])
            if (left[1] <= 0 < right[1]) or (left[1] >= 0 > right[1])]


def analyze(run_folder, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((run_folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("未完成运行不能分析")
    raw_path = run_folder / "trajectories.jsonl.gz"
    if old.sha(raw_path) != manifest["raw_sha256"]:
        raise ValueError("原始轨迹哈希不一致")
    current = {name: old.sha(old.BASE / name) for name in manifest["source_hashes"]}
    if current != manifest["source_hashes"]:
        raise ValueError("冻结A0来源发生变化")
    with gzip.open(raw_path, "rt", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    if len(rows) != manifest["trajectories"]:
        raise ValueError("轨迹数量与manifest不一致")
    pairs = paired_prefix_rows(rows)
    points = {}
    for generation in sorted({row["generation"] for row in pairs}):
        subset = [row for row in pairs if row["generation"] == generation]
        points[str(generation)] = {
            "progress": generation / 20,
            "pairs": len(subset),
            "common_decodes": {
                "mean": fmean(row["common_decodes"] for row in subset),
                "min": min(row["common_decodes"] for row in subset),
                "max": max(row["common_decodes"] for row in subset),
                "discarded_fixed": sum(row["discarded_fixed_decodes"] for row in subset),
                "discarded_feedback": sum(row["discarded_feedback_decodes"] for row in subset),
            },
            **{metric: summarize_paired(subset, metric, 10000, 20260907 + generation * 10 + index)
               for index, metric in enumerate(METRICS)},
        }
    result = {
        "status": "exploratory_reanalysis",
        "claim_scope": "pointwise_curve_without_cutoff_selection",
        "source_commit": manifest["source_commit"],
        "source_raw_sha256": manifest["raw_sha256"],
        "trajectories_reused": len(rows),
        "paired_prefixes": len(pairs),
        "new_schedule_decodes": 0,
        "metric_direction": "feedback_minus_fixed_higher_is_better",
        "points": points,
        "observed_sign_changes": {
            "hv_gain": sign_change_intervals(points, "hv_gain"),
            "hv_per_decode": sign_change_intervals(points, "hv_per_decode"),
            "makespan_improvement": sign_change_intervals(points, "makespan_improvement"),
            "tec_improvement": sign_change_intervals(points, "tec_improvement"),
        },
        "cutoff_selected": False,
        "confirmation_claim_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    old.dump(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.output)


if __name__ == "__main__":
    main()
