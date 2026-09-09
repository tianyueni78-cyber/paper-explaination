"""第三轮析因交互和强简单对照分析。"""

import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean


def load_rows(run_dir):
    with gzip.open(Path(run_dir) / "results.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _unit(row):
    return row["state"], row["repeat"]


def _efficiency(row):
    return row["gain"] / max(1, row["decodes"])


def _interval(values, seed=20260908, samples=10000):
    rng = random.Random(seed)
    boot = sorted(fmean(rng.choice(values) for _ in values) for _ in range(samples))
    low = max(0, round(samples * 0.025) - 1)
    high = min(samples - 1, round(samples * 0.975) - 1)
    return [boot[low], boot[high]]


def _summarize_records(records, label, bootstrap_samples):
    by_instance = defaultdict(list)
    for row in records:
        by_instance[row["instance"]].append(row)
    instance_gain = {
        name: fmean(row["gain"] for row in group)
        for name, group in sorted(by_instance.items())
    }
    instance_efficiency = {
        name: fmean(row["efficiency"] for row in group)
        for name, group in sorted(by_instance.items())
    }
    return {
        "contrast": label,
        "pairs": len(records),
        "mean_gain_difference": fmean(row["gain"] for row in records),
        "mean_decode_difference": fmean(row["decodes"] for row in records),
        "mean_efficiency_difference": fmean(row["efficiency"] for row in records),
        "instance_gain_differences": instance_gain,
        "gain_interval": _interval(list(instance_gain.values()), samples=bootstrap_samples),
        "positive_gain_instances": sum(value > 0 for value in instance_gain.values()),
        "instance_efficiency_differences": instance_efficiency,
        "efficiency_interval": _interval(
            list(instance_efficiency.values()), seed=20260909, samples=bootstrap_samples
        ),
        "positive_efficiency_instances": sum(value > 0 for value in instance_efficiency.values()),
    }


def paired_contrast(rows, treatment, baseline, bootstrap_samples=10000):
    indexed = {(_unit(row), row["configuration"]): row for row in rows}
    records = []
    treatment_units = sorted(unit for unit, name in indexed if name == treatment)
    for unit in treatment_units:
        new = indexed[unit, treatment]
        old = indexed[unit, baseline]
        records.append({
            "instance": new.get("instance", new["state"].split("-")[0]),
            "gain": new["gain"] - old["gain"],
            "decodes": new["decodes"] - old["decodes"],
            "efficiency": _efficiency(new) - _efficiency(old),
        })
    return _summarize_records(records, f"{treatment}-{baseline}", bootstrap_samples)


def _factor_weights(name):
    if name == "K:N":
        return {"000": 1, "001": 1, "010": -1, "011": -1,
                "100": -1, "101": -1, "110": 1, "111": 1}
    if name == "K:b":
        return {"000": 1, "001": -1, "010": 1, "011": -1,
                "100": -1, "101": 1, "110": -1, "111": 1}
    if name == "N:b":
        return {"000": 1, "001": -1, "010": -1, "011": 1,
                "100": 1, "101": -1, "110": -1, "111": 1}
    if name == "K:N:b":
        return {"000": -1, "001": 1, "010": 1, "011": -1,
                "100": 1, "101": -1, "110": -1, "111": 1}
    raise ValueError(f"unknown factorial contrast: {name}")


def factorial_contrast(rows, name, bootstrap_samples=10000):
    canonical = []
    for row in rows:
        copied = dict(row)
        copied["configuration"] = copied["configuration"].split("_")[0]
        canonical.append(copied)
    indexed = {(_unit(row), row["configuration"]): row for row in canonical}
    weights = _factor_weights(name)
    scale = 2 if name.count(":") == 1 else 1
    units = sorted({unit for unit, configuration in indexed if configuration == "000"})
    records = []
    for unit in units:
        selected = {configuration: indexed[unit, configuration] for configuration in weights}
        first = selected["000"]
        records.append({
            "instance": first.get("instance", first["state"].split("-")[0]),
            "gain": sum(weights[key] * selected[key]["gain"] for key in weights) / scale,
            "decodes": sum(weights[key] * selected[key]["decodes"] for key in weights) / scale,
            "efficiency": sum(weights[key] * _efficiency(selected[key]) for key in weights) / scale,
        })
    summary = _summarize_records(records, name, bootstrap_samples)
    summary["mean_gain_contrast"] = summary.pop("mean_gain_difference")
    summary["mean_decode_contrast"] = summary.pop("mean_decode_difference")
    summary["mean_efficiency_contrast"] = summary.pop("mean_efficiency_difference")
    return summary


def _canonical_interaction_rows(rows):
    output = []
    for row in rows:
        copied = dict(row)
        copied["configuration"] = copied["configuration"].split("_")[0]
        output.append(copied)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interaction-run", type=Path, required=True)
    parser.add_argument("--k-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    interaction = _canonical_interaction_rows(load_rows(args.interaction_run))
    k_rows = load_rows(args.k_run)
    k50 = []
    for row in k_rows:
        if row["configuration"] == "100_rule":
            copied = dict(row)
            copied["configuration"] = "K50_fixed"
            k50.append(copied)
    combined = interaction + k50

    configurations = ["001", "010", "011", "100", "101", "110", "111"]
    result = {
        "versus_k25_n3_b4": {
            name: paired_contrast(interaction, name, "000") for name in configurations
        },
        "incremental_module_effects": {
            "b_given_K": paired_contrast(interaction, "101", "100"),
            "N_given_K": paired_contrast(interaction, "110", "100"),
            "N_given_Kb": paired_contrast(interaction, "111", "101"),
            "b_given_KN": paired_contrast(interaction, "111", "110"),
            "K_given_Nb": paired_contrast(interaction, "111", "011"),
        },
        "factorial_interactions": {
            name: factorial_contrast(interaction, name)
            for name in ("K:N", "K:b", "N:b", "K:N:b")
        },
        "versus_fixed_k50": {
            name: paired_contrast(combined, name, "K50_fixed")
            for name in ("100", "101", "110", "111")
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
