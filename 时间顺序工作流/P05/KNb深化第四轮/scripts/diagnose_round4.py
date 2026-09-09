import argparse
import csv
import gzip
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit_key(row):
    return row["instance"], int(row["generation"]), int(row["repeat"])


def unit_label(key):
    return f"{key[0]}|g{key[1]}|r{key[2]}"


def trajectory_stats(row):
    totals = Counter()
    k_counts = Counter()
    n_counts = Counter()
    b_counts = Counter()
    mode_counts = Counter()
    states = Counter()
    expand_decisions = 0
    expand_k50 = 0

    for decision in row.get("trace", []):
        totals["decisions"] += 1
        decodes = int(decision.get("decodes", 0))
        if decodes == 0:
            totals["zero_decode_decisions"] += 1
        action = decision.get("action", [None, None, None])
        k_counts[str(action[0])] += 1
        n_counts[str(action[1])] += 1
        b_counts[str(action[2])] += 1
        mode_counts[str(decision.get("mode", "unknown"))] += 1
        if decision.get("q_updated", False):
            totals["q_updates"] += 1
        state = tuple(decision.get("state", []))
        states[str(state)] += 1
        if state and int(state[0]) == 1:
            expand_decisions += 1
            if float(action[0]) == 0.5:
                expand_k50 += 1
        for tranche in decision.get("tranches", []):
            for field in ("attempts", "outside", "noop", "duplicate", "evaluated"):
                totals[field] += int(tranche.get(field, 0))
            totals["accepted"] += int(bool(tranche.get("accepted")))

    decisions = totals["decisions"]
    return {
        "gain": float(row.get("gain", 0.0)),
        "decodes": int(row.get("decodes", 0)),
        "decisions": decisions,
        "zero_decode_decisions": totals["zero_decode_decisions"],
        "attempts": totals["attempts"],
        "outside": totals["outside"],
        "noop": totals["noop"],
        "duplicate": totals["duplicate"],
        "evaluated": totals["evaluated"],
        "accepted": totals["accepted"],
        "q_updates": totals["q_updates"],
        "unique_states": len(states),
        "visits_per_state": decisions / len(states) if states else 0.0,
        "k_counts": dict(k_counts),
        "n_counts": dict(n_counts),
        "b_counts": dict(b_counts),
        "mode_counts": dict(mode_counts),
        "expand_decisions": expand_decisions,
        "expand_k50": expand_k50,
    }


def _safe_rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def row_metrics(row):
    stats = trajectory_stats(row)
    stats.update(
        {
            "gain_per_decode": _safe_rate(stats["gain"], stats["decodes"]),
            "outside_rate": _safe_rate(stats["outside"], stats["attempts"]),
            "noop_rate": _safe_rate(stats["noop"], stats["attempts"]),
            "duplicate_rate": _safe_rate(stats["duplicate"], stats["attempts"]),
            "evaluation_rate": _safe_rate(stats["evaluated"], stats["attempts"]),
            "zero_decode_rate": _safe_rate(stats["zero_decode_decisions"], stats["decisions"]),
        }
    )
    return stats


def summarize_configuration(rows):
    metrics = [row_metrics(row) for row in rows]
    additive = ("gain", "decodes", "decisions", "attempts", "outside", "noop", "duplicate", "evaluated", "accepted")
    totals = {name: sum(item[name] for item in metrics) for name in additive}
    return {
        "rows": len(rows),
        "mean_gain": statistics.fmean(item["gain"] for item in metrics),
        "mean_decodes": statistics.fmean(item["decodes"] for item in metrics),
        "gain_per_decode": _safe_rate(totals["gain"], totals["decodes"]),
        "outside_rate": _safe_rate(totals["outside"], totals["attempts"]),
        "noop_rate": _safe_rate(totals["noop"], totals["attempts"]),
        "duplicate_rate": _safe_rate(totals["duplicate"], totals["attempts"]),
        "evaluation_rate": _safe_rate(totals["evaluated"], totals["attempts"]),
        "acceptance_rate": _safe_rate(totals["accepted"], totals["evaluated"]),
        "zero_decode_rate": _safe_rate(
            sum(item["zero_decode_decisions"] for item in metrics), totals["decisions"]
        ),
    }


def _bootstrap_ci(values, seed=20260909, samples=20000):
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(draw))
    means.sort()
    return [means[int(0.025 * samples)], means[int(0.975 * samples)]]


def summarize_instance_effects(records, value_field):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["instance"]].append(float(record[value_field]))
    instance_effects = {key: statistics.fmean(values) for key, values in sorted(grouped.items())}
    values = list(instance_effects.values())
    return {
        "mean": statistics.fmean(values) if values else None,
        "positive_instances": sum(value > 0 for value in values),
        "zero_instances": sum(value == 0 for value in values),
        "instances": len(values),
        "bootstrap_95_ci": _bootstrap_ci(values),
        "by_instance": instance_effects,
    }


def paired_difference(rows, baseline, treatment, metric):
    by_config = defaultdict(dict)
    for row in rows:
        by_config[row["configuration"]][unit_key(row)] = row_metrics(row)
    common = sorted(set(by_config[baseline]) & set(by_config[treatment]))
    records = []
    for key in common:
        records.append(
            {
                "unit": unit_label(key),
                "instance": key[0],
                "generation": key[1],
                "repeat": key[2],
                "difference": by_config[treatment][key][metric] - by_config[baseline][key][metric],
            }
        )
    return {"metric": metric, "units": len(records), "effect": summarize_instance_effects(records, "difference")}


def factorial_interaction(rows, c00, c01, c10, c11):
    by_config = defaultdict(dict)
    for row in rows:
        by_config[row["configuration"]][unit_key(row)] = float(row["gain"])
    common = sorted(set.intersection(*(set(by_config[name]) for name in (c00, c01, c10, c11))))
    records = []
    for key in common:
        off_effect = by_config[c01][key] - by_config[c00][key]
        on_effect = by_config[c11][key] - by_config[c10][key]
        records.append(
            {
                "unit": unit_label(key),
                "instance": key[0],
                "generation": key[1],
                "repeat": key[2],
                "b_effect_k_off": off_effect,
                "b_effect_k_on": on_effect,
                "interaction": on_effect - off_effect,
            }
        )
    return records


def action_diagnostics(rows, configurations):
    chosen = [row for row in rows if row["configuration"] in configurations]
    totals = Counter()
    k_counts = Counter()
    n_counts = Counter()
    b_counts = Counter()
    mode_counts = Counter()
    state_counts = Counter()
    n_outcomes = defaultdict(Counter)
    per_trajectory = []

    for row in chosen:
        stats = trajectory_stats(row)
        per_trajectory.append(stats)
        totals["decisions"] += stats["decisions"]
        totals["q_updates"] += stats["q_updates"]
        totals["expand_decisions"] += stats["expand_decisions"]
        totals["expand_k50"] += stats["expand_k50"]
        totals["zero_decode_decisions"] += stats["zero_decode_decisions"]
        k_counts.update(stats["k_counts"])
        n_counts.update(stats["n_counts"])
        b_counts.update(stats["b_counts"])
        mode_counts.update(stats["mode_counts"])
        for decision in row.get("trace", []):
            action = decision.get("action", [None, None, None])
            n_key = str(action[1])
            n_outcomes[n_key]["decisions"] += 1
            n_outcomes[n_key]["decodes"] += int(decision.get("decodes", 0))
            n_outcomes[n_key]["positive_decisions"] += float(decision.get("gain", 0.0)) > 0
            n_outcomes[n_key]["gain"] += float(decision.get("gain", 0.0))
            state_counts[str(tuple(decision.get("state", [])))] += 1

    def shares(counter):
        total = sum(counter.values())
        return {key: value / total for key, value in sorted(counter.items())}

    n_summary = {}
    for key, values in sorted(n_outcomes.items()):
        n_summary[key] = {
            "decisions": values["decisions"],
            "decision_share": _safe_rate(values["decisions"], totals["decisions"]),
            "positive_decision_rate": _safe_rate(values["positive_decisions"], values["decisions"]),
            "gain_per_decode": _safe_rate(values["gain"], values["decodes"]),
        }

    return {
        "configurations": sorted(configurations),
        "trajectories": len(chosen),
        "decisions": totals["decisions"],
        "mode_shares": shares(mode_counts),
        "k_shares": shares(k_counts),
        "n_shares": shares(n_counts),
        "b_shares": shares(b_counts),
        "n_outcomes_descriptive": n_summary,
        "q_update_rate": _safe_rate(totals["q_updates"], totals["decisions"]),
        "zero_decode_rate": _safe_rate(totals["zero_decode_decisions"], totals["decisions"]),
        "expand_state_decisions": totals["expand_decisions"],
        "k50_rate_when_expand_available": _safe_rate(totals["expand_k50"], totals["expand_decisions"]),
        "unique_states_global": len(state_counts),
        "mean_unique_states_per_trajectory": statistics.fmean(x["unique_states"] for x in per_trajectory),
        "mean_visits_per_state_per_trajectory": statistics.fmean(x["visits_per_state"] for x in per_trajectory),
        "state_visit_distribution": dict(sorted(state_counts.items())),
    }


def build_report(k_rows, interaction_rows, sources):
    k_configs = defaultdict(list)
    for row in k_rows:
        k_configs[row["configuration"]].append(row)
    interaction_configs = defaultdict(list)
    for row in interaction_rows:
        interaction_configs[row["configuration"]].append(row)

    source_metrics = ("gain", "decodes", "gain_per_decode", "outside_rate", "noop_rate", "duplicate_rate", "evaluation_rate", "zero_decode_rate")
    k_source = {
        "configuration_summaries": {
            name: summarize_configuration(rows) for name, rows in sorted(k_configs.items())
        },
        "K50_rule_minus_K25_fixed": [
            paired_difference(k_rows, "000_fixed", "100_rule", metric) for metric in source_metrics
        ],
    }

    kb_records = factorial_interaction(
        interaction_rows, "000_fixed", "001_q", "100_q", "101_q"
    )
    kb = {
        "interpretation_boundary": "This is K-module x b-module, not pure K50 x b.",
        "configuration_summaries": {
            name: summarize_configuration(interaction_configs[name])
            for name in ("000_fixed", "001_q", "100_q", "101_q")
        },
        "interaction_effect": summarize_instance_effects(kb_records, "interaction"),
        "b_effect_k_off": summarize_instance_effects(kb_records, "b_effect_k_off"),
        "b_effect_k_on": summarize_instance_effects(kb_records, "b_effect_k_on"),
        "actual_action_distribution": action_diagnostics(
            interaction_rows, {"001_q", "100_q", "101_q"}
        ),
    }

    n_effects = {}
    for baseline, treatment in (("100_q", "110_q"), ("101_q", "111_q")):
        name = f"{treatment}_minus_{baseline}"
        n_effects[name] = {
            metric: paired_difference(interaction_rows, baseline, treatment, metric)
            for metric in ("gain", "decodes", "gain_per_decode", "outside_rate", "noop_rate", "duplicate_rate")
        }
    n_diagnostic = {
        "paired_effects": n_effects,
        "N_off": action_diagnostics(interaction_rows, {"100_q", "101_q"}),
        "N_on": action_diagnostics(interaction_rows, {"110_q", "111_q"}),
    }

    q_diagnostic = {
        name: action_diagnostics(interaction_rows, {name})
        for name in ("100_q", "101_q", "110_q", "111_q")
    }

    return {
        "status": "exploratory_failure_attribution",
        "independent_unit": "instance after averaging generation and repeat",
        "source_files": sources,
        "F1_K50_source": k_source,
        "F2_K_module_x_b_module": kb,
        "F3_N_negative_effect": n_diagnostic,
        "F4_Q_control": q_diagnostic,
    }


def write_summary_csv(report, path):
    rows = []
    for comparison in report["F1_K50_source"]["K50_rule_minus_K25_fixed"]:
        effect = comparison["effect"]
        rows.append(
            {
                "question": "F1_K50_rule_minus_K25_fixed",
                "metric": comparison["metric"],
                "mean": effect["mean"],
                "positive_instances": effect["positive_instances"],
                "instances": effect["instances"],
                "ci_low": effect["bootstrap_95_ci"][0],
                "ci_high": effect["bootstrap_95_ci"][1],
            }
        )
    for metric in ("interaction_effect", "b_effect_k_off", "b_effect_k_on"):
        effect = report["F2_K_module_x_b_module"][metric]
        rows.append(
            {
                "question": "F2_K_module_x_b_module",
                "metric": metric,
                "mean": effect["mean"],
                "positive_instances": effect["positive_instances"],
                "instances": effect["instances"],
                "ci_low": effect["bootstrap_95_ci"][0],
                "ci_high": effect["bootstrap_95_ci"][1],
            }
        )
    for contrast, metrics in report["F3_N_negative_effect"]["paired_effects"].items():
        for metric, comparison in metrics.items():
            effect = comparison["effect"]
            rows.append(
                {
                    "question": f"F3_{contrast}",
                    "metric": metric,
                    "mean": effect["mean"],
                    "positive_instances": effect["positive_instances"],
                    "instances": effect["instances"],
                    "ci_low": effect["bootstrap_95_ci"][0],
                    "ci_high": effect["bootstrap_95_ci"][1],
                }
            )
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--third-round", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    k_path = args.third_round / "runs" / "K-v1" / "results.jsonl.gz"
    interaction_path = args.third_round / "runs" / "interaction-v2" / "results.jsonl.gz"
    sources = {
        "K-v1": {"path": str(k_path), "sha256": sha256(k_path)},
        "interaction-v2": {"path": str(interaction_path), "sha256": sha256(interaction_path)},
    }
    report = build_report(load_rows(k_path), load_rows(interaction_path), sources)
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "failure_attribution.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    write_summary_csv(report, args.output / "effect_summary.csv")


if __name__ == "__main__":
    main()
