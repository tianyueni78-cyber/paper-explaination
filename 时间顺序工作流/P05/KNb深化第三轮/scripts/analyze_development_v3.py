"""第三轮开发实验的配对分析。"""

import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean


def paired_differences(rows, treatment, baseline):
    indexed = {(row["state"], row["repeat"], row["configuration"]): row for row in rows}
    output = []
    for state, repeat, configuration in sorted(indexed):
        if configuration != treatment:
            continue
        new = indexed[state, repeat, treatment]
        old = indexed[state, repeat, baseline]
        output.append({
            "state": state,
            "instance": new.get("instance", state.split("-")[0]),
            "repeat": repeat,
            "gain_difference": new["gain"] - old["gain"],
            "decode_difference": new["decodes"] - old["decodes"],
            "efficiency_difference": new["gain"] / max(1, new["decodes"]) - old["gain"] / max(1, old["decodes"]),
        })
    return output


def _interval(values, seed=20260908, samples=10000):
    rng = random.Random(seed)
    boot = sorted(fmean(rng.choice(values) for _ in values) for _ in range(samples))
    return [boot[249], boot[9749]]


def summarize(rows, treatment, baseline="000_fixed"):
    pairs = paired_differences(rows, treatment, baseline)
    by_instance = defaultdict(list)
    for row in pairs:
        by_instance[row["instance"]].append(row)
    instance_gain = {name: fmean(row["gain_difference"] for row in records) for name, records in sorted(by_instance.items())}
    instance_efficiency = {name: fmean(row["efficiency_difference"] for row in records) for name, records in sorted(by_instance.items())}
    gain_values = list(instance_gain.values())
    efficiency_values = list(instance_efficiency.values())
    return {
        "treatment": treatment,
        "baseline": baseline,
        "pairs": len(pairs),
        "mean_gain_difference": fmean(row["gain_difference"] for row in pairs),
        "mean_decode_difference": fmean(row["decode_difference"] for row in pairs),
        "mean_efficiency_difference": fmean(row["efficiency_difference"] for row in pairs),
        "instance_gain_differences": instance_gain,
        "gain_interval": _interval(gain_values),
        "positive_gain_instances": sum(value > 0 for value in gain_values),
        "instance_efficiency_differences": instance_efficiency,
        "efficiency_interval": _interval(efficiency_values, seed=20260909),
        "positive_efficiency_instances": sum(value > 0 for value in efficiency_values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.input / "results.jsonl.gz", "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    treatments = sorted({row["configuration"] for row in rows} - {"000_fixed"})
    result = {name: summarize(rows, name) for name in treatments}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
