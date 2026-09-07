"""条件动作景观分析。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean


PRIMARY = "hv_gain_per_allowed_decode"
METRICS = (
    PRIMARY,
    "gain",
    "gain_per_decode",
    "makespan_improvement",
    "tec_improvement",
    "decodes",
)


def action_key(row: dict) -> tuple[float, int, int]:
    return float(row["fraction"]), int(row["n"]), int(row["b"])


def _number(row: dict, key: str) -> float:
    value = row[key]
    if value in (None, "", "None"):
        return 0.0
    return float(value)


def _choose(rows: list[dict], repeats: set[int] | None = None) -> tuple[float, int, int]:
    values = defaultdict(list)
    for row in rows:
        if repeats is None or int(row["repeat"]) in repeats:
            values[action_key(row)].append(_number(row, PRIMARY))
    if not values:
        raise ValueError("没有可用于动作选择的数据")
    return min(values, key=lambda action: (-fmean(values[action]), action))


def _evaluate(rows: list[dict], action: tuple[float, int, int], repeats=(2, 3)) -> dict:
    selected = [
        row
        for row in rows
        if action_key(row) == action and int(row["repeat"]) in set(repeats)
    ]
    if not selected:
        raise ValueError(f"动作{action}缺少评价重复")
    result = {metric: fmean(_number(row, metric) for row in selected) for metric in METRICS}
    first = selected[0]
    return {
        "state": first["state"],
        "instance": first["instance"],
        "generation": int(first.get("generation", 0)),
        "selected_action": action,
        **result,
    }


def _state_groups(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["state"]].append(row)
    return grouped


def split_repeat_oracle(rows: list[dict]) -> list[dict]:
    results = []
    for state_rows in _state_groups(rows).values():
        selected = _choose(state_rows, {0, 1})
        results.append(_evaluate(state_rows, selected))
    return sorted(results, key=lambda row: row["state"])


def posthoc_oracle(rows: list[dict]) -> list[dict]:
    results = []
    for state_rows in _state_groups(rows).values():
        selected = _choose(state_rows, {2, 3})
        results.append(_evaluate(state_rows, selected))
    return sorted(results, key=lambda row: row["state"])


def declared_fixed(rows: list[dict], action=(0.25, 3, 4)) -> list[dict]:
    return sorted(
        [_evaluate(state_rows, action) for state_rows in _state_groups(rows).values()],
        key=lambda row: row["state"],
    )


def leave_one_instance_fixed(rows: list[dict]) -> list[dict]:
    results = []
    instances = sorted({row["instance"] for row in rows})
    grouped = _state_groups(rows)
    for heldout in instances:
        training = [row for row in rows if row["instance"] != heldout]
        selected = _choose(training)
        for state_rows in grouped.values():
            if state_rows[0]["instance"] == heldout:
                results.append(_evaluate(state_rows, selected))
    return sorted(results, key=lambda row: row["state"])


def _standardizer(vectors: list[list[float]]):
    means = [fmean(column) for column in zip(*vectors)]
    scales = []
    for index, mean in enumerate(means):
        variance = fmean((row[index] - mean) ** 2 for row in vectors)
        scales.append(math.sqrt(variance) or 1.0)
    return means, scales


def _distance(left: list[float], right: list[float], means, scales) -> float:
    return sum(
        (((left[index] - means[index]) / scales[index]) -
         ((right[index] - means[index]) / scales[index])) ** 2
        for index in range(len(means))
    )


def leave_one_instance_knn(
    states: list[dict], rows: list[dict], neighbors: int = 3, shuffled: bool = False
) -> list[dict]:
    state_map = {state["state"]: state for state in states}
    grouped = _state_groups(rows)
    instances = sorted({state["instance"] for state in states})
    results = []
    for heldout in instances:
        training_states = [state for state in states if state["instance"] != heldout]
        means, scales = _standardizer([state["features"] for state in training_states])
        action_means = {}
        for state in training_states:
            by_action = defaultdict(list)
            for row in grouped[state["state"]]:
                by_action[action_key(row)].append(_number(row, PRIMARY))
            action_means[state["state"]] = {
                action: fmean(values) for action, values in by_action.items()
            }
        if shuffled:
            for state in training_states:
                actions = sorted(action_means[state["state"]])
                values = [action_means[state["state"]][action] for action in actions]
                random.Random(f"conditional-shuffle-{state['state']}").shuffle(values)
                action_means[state["state"]] = dict(zip(actions, values))
        for test_state in (state for state in states if state["instance"] == heldout):
            nearest = sorted(
                training_states,
                key=lambda state: (
                    _distance(test_state["features"], state["features"], means, scales),
                    state["state"],
                ),
            )[: min(neighbors, len(training_states))]
            available = sorted(action_means[nearest[0]["state"]])
            selected = min(
                available,
                key=lambda action: (
                    -fmean(action_means[state["state"]][action] for state in nearest),
                    action,
                ),
            )
            results.append(_evaluate(grouped[test_state["state"]], selected))
    return sorted(results, key=lambda row: row["state"])


def summarize_difference(method: list[dict], baseline: list[dict], samples=10000, seed=20260908) -> dict:
    baseline_by_state = {row["state"]: row for row in baseline}
    by_instance = defaultdict(list)
    for row in method:
        by_instance[row["instance"]].append(
            row[PRIMARY] - baseline_by_state[row["state"]][PRIMARY]
        )
    instance_differences = {
        instance: fmean(values) for instance, values in sorted(by_instance.items())
    }
    values = list(instance_differences.values())
    rng = random.Random(seed)
    bootstrap = sorted(
        fmean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    lower = bootstrap[math.floor(0.025 * (samples - 1))]
    upper = bootstrap[math.floor(0.975 * (samples - 1))]
    return {
        "mean_difference": fmean(values),
        "interval": [lower, upper],
        "positive_instances": sum(value > 0 for value in values),
        "instance_differences": instance_differences,
    }


def summarize_metric_difference(
    method: list[dict], baseline: list[dict], metric: str
) -> dict:
    baseline_by_state = {row["state"]: row for row in baseline}
    by_instance = defaultdict(list)
    for row in method:
        by_instance[row["instance"]].append(
            float(row[metric]) - float(baseline_by_state[row["state"]][metric])
        )
    instance_differences = {
        instance: fmean(values) for instance, values in sorted(by_instance.items())
    }
    return {
        "mean_difference": fmean(instance_differences.values()),
        "positive_instances": sum(value > 0 for value in instance_differences.values()),
        "instance_differences": instance_differences,
    }


def action_frequencies(records: list[dict]) -> dict[str, int]:
    counts = Counter(str(tuple(row["selected_action"])) for row in records)
    return dict(sorted(counts.items()))


def action_dimension_frequencies(records: list[dict]) -> dict[str, dict[str, int]]:
    dimensions = {"K": Counter(), "N": Counter(), "b": Counter()}
    for row in records:
        fraction, neighborhood, budget = row["selected_action"]
        dimensions["K"][str(float(fraction))] += 1
        dimensions["N"][str(int(neighborhood))] += 1
        dimensions["b"][str(int(budget))] += 1
    return {name: dict(sorted(counts.items())) for name, counts in dimensions.items()}


def summarize_by_generation(method: list[dict], baseline: list[dict]) -> dict[str, float]:
    baseline_by_state = {row["state"]: row for row in baseline}
    differences = defaultdict(list)
    for row in method:
        differences[str(row["generation"])].append(
            row[PRIMARY] - baseline_by_state[row["state"]][PRIMARY]
        )
    return {generation: fmean(values) for generation, values in sorted(differences.items(), key=lambda item: int(item[0]))}


def marginal_budget(rows: list[dict]) -> dict:
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["state"], float(row["fraction"]), int(row["n"]), int(row["repeat"]))
        grouped[key][int(row["b"])] = row
    differences = defaultdict(list)
    for key, pair in grouped.items():
        if set(pair) != {4, 12}:
            continue
        instance = pair[4]["instance"]
        extra_decodes = _number(pair[12], "decodes") - _number(pair[4], "decodes")
        extra_gain = _number(pair[12], "gain") - _number(pair[4], "gain")
        differences[instance].append(extra_gain / extra_decodes if extra_decodes > 0 else 0.0)
    per_instance = {instance: fmean(values) for instance, values in sorted(differences.items())}
    return {
        "mean_marginal_hv_per_true_decode": fmean(per_instance.values()),
        "positive_instances": sum(value > 0 for value in per_instance.values()),
        "per_instance": per_instance,
    }


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze_run(run_folder: Path) -> dict:
    rows = read_rows(run_folder / "trials.csv")
    states = json.loads((run_folder / "states.json").read_text(encoding="utf-8"))
    fixed = leave_one_instance_fixed(rows)
    methods = {
        "declared_fixed": declared_fixed(rows),
        "fold_fixed": fixed,
        "posthoc_oracle": posthoc_oracle(rows),
        "split_repeat_oracle": split_repeat_oracle(rows),
        "knn": leave_one_instance_knn(states, rows),
        "shuffled_knn": leave_one_instance_knn(states, rows, shuffled=True),
    }
    comparisons = {
        name: summarize_difference(records, fixed)
        for name, records in methods.items()
        if name != "fold_fixed"
    }
    gate = {
        "oracle_passed": comparisons["split_repeat_oracle"]["mean_difference"] > 0
        and comparisons["split_repeat_oracle"]["positive_instances"] >= 4,
        "prediction_passed": comparisons["knn"]["mean_difference"] > 0
        and comparisons["knn"]["positive_instances"] >= 4,
    }
    result = {
        "rows": len(rows),
        "states": len(states),
        "comparisons_to_fold_fixed": comparisons,
        "action_frequencies": {
            name: action_frequencies(records) for name, records in methods.items()
        },
        "action_dimension_frequencies": {
            name: action_dimension_frequencies(records) for name, records in methods.items()
        },
        "generation_differences_to_fold_fixed": {
            name: summarize_by_generation(records, fixed)
            for name, records in methods.items()
            if name != "fold_fixed"
        },
        "metric_differences_to_fold_fixed": {
            name: {
                metric: summarize_metric_difference(records, fixed, metric)
                for metric in METRICS
            }
            for name, records in methods.items()
            if name != "fold_fixed"
        },
        "marginal_budget": marginal_budget(rows),
        "gate": gate,
        "records": methods,
    }
    (run_folder / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_run(args.run)
    print(json.dumps({key: result[key] for key in ("rows", "states", "gate")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
