"""只使用已冻结开发动作景观的结构状态离线诊断。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
P05 = ROOT.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(P05 / "S0" / "scripts"))
sys.path.insert(0, str(P05 / "KNb深挖" / "scripts"))
import probe as old
import probe_knb as deep
from paper_static_baseline.dfjspt.chromosome import Chromosome


PRIMARY = "hv_gain_per_allowed_decode"
ALL_ACTIONS = [(k, n, b) for k in (0.25, 0.5) for n in range(1, 7) for b in (4, 12)]


def make_test_row(state, instance, action, repeat, value):
    k, n, b = action
    return {
        "state": state,
        "instance": instance,
        "fraction": k,
        "n": n,
        "b": b,
        "repeat": repeat,
        PRIMARY: value,
    }


def action_key(row):
    return float(row["fraction"]), int(row["n"]), int(row["b"])


def axis_actions(actions, axis):
    if axis == "K":
        return [action for action in actions if action[1:] == (3, 4)]
    if axis == "N":
        return [action for action in actions if action[0] == 0.25 and action[2] == 4]
    if axis == "b":
        return [action for action in actions if action[:2] == (0.25, 3)]
    if axis == "joint":
        return list(actions)
    raise ValueError("axis必须为K、N、b或joint")


def _groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["state"]].append(row)
    return grouped


def _action_means(rows, repeats, allowed):
    values = defaultdict(list)
    for row in rows:
        action = action_key(row)
        if action in allowed and int(row["repeat"]) in repeats:
            values[action].append(float(row[PRIMARY]))
    return {action: fmean(items) for action, items in values.items()}


def split_oracle(rows, allowed):
    output = []
    for state_rows in _groups(rows).values():
        selection = _action_means(state_rows, {0, 1}, set(allowed))
        evaluation = _action_means(state_rows, {2, 3}, set(allowed))
        selected = min(selection, key=lambda action: (-selection[action], action))
        output.append(
            {
                "state": state_rows[0]["state"],
                "instance": state_rows[0]["instance"],
                "selected_action": selected,
                "value": evaluation[selected],
            }
        )
    return sorted(output, key=lambda row: row["state"])


def stability(rows, allowed):
    agreements = []
    regrets = []
    for state_rows in _groups(rows).values():
        selection = _action_means(state_rows, {0, 1}, set(allowed))
        evaluation = _action_means(state_rows, {2, 3}, set(allowed))
        selected = min(selection, key=lambda action: (-selection[action], action))
        evaluated = min(evaluation, key=lambda action: (-evaluation[action], action))
        agreements.append(float(selected == evaluated))
        regrets.append(evaluation[evaluated] - evaluation[selected])
    return {"top1_agreement": fmean(agreements), "mean_regret": fmean(regrets)}


def _top(values, fraction):
    count = max(1, math.ceil(len(values) * fraction))
    return set(sorted(range(len(values)), key=lambda index: (-values[index], index))[:count])


def _jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def structural_summary(machine_scores, agv_scores, waits, flexible, agv_assignments):
    count = len(agv_scores)
    cut25 = max(1, math.ceil(count * 0.25))
    ordered = sorted(agv_scores, reverse=True)
    scale = max(sum(abs(value) for value in agv_scores), 1.0)
    boundary = (ordered[cut25 - 1] - ordered[cut25]) / scale if cut25 < count else 0.0
    a25, a50 = _top(agv_scores, 0.25), _top(agv_scores, 0.5)
    m25, w25 = _top(machine_scores, 0.25), _top(waits, 0.25)
    second = a50 - a25
    second_share = sum(agv_scores[index] for index in second) / scale
    return {
        "agv_boundary_gap_25": boundary,
        "agv_second_quartile_share": second_share,
        "machine_agv_top25_jaccard": _jaccard(m25, a25),
        "wait_agv_top25_jaccard": _jaccard(w25, a25),
        "flexible_fraction_a25": sum(flexible[index] for index in a25) / len(a25),
        "flexible_fraction_a50": sum(flexible[index] for index in a50) / len(a50),
        "agv_diversity_a25": len({agv_assignments[index] for index in a25}) / len(a25),
        "agv_diversity_a50": len({agv_assignments[index] for index in a50}) / len(a50),
    }


def enrich_states(states):
    data_cache = {}
    output = []
    for state in states:
        instance = state["instance"]
        data = data_cache.setdefault(instance, old.load_case(instance))
        values = state["chromosome"]
        chromosome = Chromosome(**{name: tuple(segment) for name, segment in values.items()})
        schedule = old.decode_static(data, chromosome)
        machine_scores, agv_scores, _ = old.scores(data, schedule)
        waits, _, _ = deep.wait_features(data, schedule)
        operations = [operation for job in data.instance.jobs for operation in job.operations]
        structural = structural_summary(
            machine_scores,
            agv_scores,
            waits,
            [int(len(operation.options) > 1) for operation in operations],
            chromosome.agv,
        )
        output.append({**state, "structural": structural, "augmented": [*state["features"], *structural.values()]})
    return output


def hierarchical_choice(values):
    k_values = defaultdict(list)
    for (k, _, _), value in values.items():
        k_values[k].append(value)
    selected_k = min(k_values, key=lambda key: (-fmean(k_values[key]), key))
    n_values = defaultdict(list)
    for (k, n, _), value in values.items():
        if k == selected_k:
            n_values[n].append(value)
    selected_n = min(n_values, key=lambda key: (-fmean(n_values[key]), key))
    candidates = {action: value for action, value in values.items() if action[:2] == (selected_k, selected_n)}
    return min(candidates, key=lambda action: (-candidates[action], action))


def _standardize(train):
    means = [fmean(column) for column in zip(*train)]
    scales = []
    for index, mean in enumerate(means):
        scale = math.sqrt(fmean((row[index] - mean) ** 2 for row in train))
        scales.append(scale or 1.0)
    return means, scales


def _distance(left, right, means, scales):
    return sum((((left[i] - means[i]) - (right[i] - means[i])) / scales[i]) ** 2 for i in range(len(means)))


def predict(states, rows, feature_key, hierarchical=False, neighbors=3):
    grouped = _groups(rows)
    output = []
    for heldout in sorted({state["instance"] for state in states}):
        train = [state for state in states if state["instance"] != heldout]
        means, scales = _standardize([state[feature_key] for state in train])
        train_values = {}
        for state in train:
            train_values[state["state"]] = _action_means(grouped[state["state"]], {0, 1, 2, 3}, set(ALL_ACTIONS))
        for test in (state for state in states if state["instance"] == heldout):
            nearest = sorted(train, key=lambda item: (_distance(test[feature_key], item[feature_key], means, scales), item["state"]))[:neighbors]
            values = {action: fmean(train_values[item["state"]][action] for item in nearest) for action in ALL_ACTIONS}
            selected = hierarchical_choice(values) if hierarchical else min(values, key=lambda action: (-values[action], action))
            evaluation = _action_means(grouped[test["state"]], {2, 3}, set(ALL_ACTIONS))
            output.append({"state": test["state"], "instance": heldout, "selected_action": selected, "value": evaluation[selected]})
    return sorted(output, key=lambda row: row["state"])


def fold_fixed(states, rows):
    grouped = _groups(rows)
    output = []
    for heldout in sorted({state["instance"] for state in states}):
        training = [row for row in rows if row["instance"] != heldout]
        values = _action_means(training, {0, 1, 2, 3}, set(ALL_ACTIONS))
        selected = min(values, key=lambda action: (-values[action], action))
        for state in (item for item in states if item["instance"] == heldout):
            evaluation = _action_means(grouped[state["state"]], {2, 3}, {selected})
            output.append({"state": state["state"], "instance": heldout, "selected_action": selected, "value": evaluation[selected]})
    return sorted(output, key=lambda row: row["state"])


def comparison(method, baseline, samples=10000):
    base = {row["state"]: row["value"] for row in baseline}
    by_instance = defaultdict(list)
    for row in method:
        by_instance[row["instance"]].append(row["value"] - base[row["state"]])
    per_instance = {key: fmean(values) for key, values in sorted(by_instance.items())}
    values = list(per_instance.values())
    rng = random.Random(20260908)
    bootstrap = sorted(fmean(rng.choice(values) for _ in values) for _ in range(samples))
    return {
        "mean_difference": fmean(values),
        "interval": [bootstrap[249], bootstrap[9749]],
        "positive_instances": sum(value > 0 for value in values),
        "instance_differences": per_instance,
    }


def _read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows = _read_rows(source / "trials.csv")
    states = json.loads((source / "states.json").read_text(encoding="utf-8"))
    enriched = enrich_states(states)
    baseline = fold_fixed(enriched, rows)
    axis = {}
    for name in ("K", "N", "b", "joint"):
        allowed = axis_actions(ALL_ACTIONS, name)
        oracle = split_oracle(rows, allowed)
        axis[name] = {
            "actions": allowed,
            "stability": stability(rows, allowed),
            "oracle_vs_fold_fixed": comparison(oracle, baseline),
        }
    predictors = {
        "flat_original": predict(enriched, rows, "features"),
        "flat_structural": predict(enriched, rows, "augmented"),
        "hierarchical_structural": predict(enriched, rows, "augmented", hierarchical=True),
    }
    result = {
        "version": "structural-state-offline-v1",
        "source_hashes": {name: _sha(source / name) for name in ("trials.csv", "states.json", "manifest.json")},
        "states": len(states),
        "trials": len(rows),
        "validation_data_opened": False,
        "structural_feature_names": list(enriched[0]["structural"]),
        "axis": axis,
        "predictors_vs_fold_fixed": {name: comparison(records, baseline) for name, records in predictors.items()},
        "seconds": time.perf_counter() - started,
    }
    (output / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "features.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["state", "instance", *result["structural_feature_names"]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for state in enriched:
            writer.writerow({"state": state["state"], "instance": state["instance"], **state["structural"]})
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["method", "state", "instance", "selected_action", "value"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, records in predictors.items():
            for row in records:
                writer.writerow({"method": name, **row})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.output)
    print(json.dumps({"states": result["states"], "trials": result["trials"], "predictors": result["predictors_vs_fold_fixed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
