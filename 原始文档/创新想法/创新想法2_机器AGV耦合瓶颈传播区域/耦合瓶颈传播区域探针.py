"""E11探针：直接负载Top-5、耦合传播Top-5与随机Top-5的公平比较。"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path
from statistics import fmean

from innovation_probe.probe_02_agv_bottleneck import (
    diagnose_agv_bottleneck,
    evaluate_candidates,
    generate_as_candidates,
    load_mk05_case,
    select_bottleneck_region,
    summarize_results,
)
from python_baseline.dfjspt.dynamic import _decode_dynamic


OperationKey = tuple[int, int]
OUTPUT = Path(__file__).resolve().parent / "outputs"


def coupled_score(transport: float, processing: float, successors: int) -> float:
    """传播代理：直接运输影响乘以下游长度，再加当前加工占用。"""
    return transport * (1 + successors) + processing


def select_region(scores: dict[OperationKey, float], size: int) -> tuple[OperationKey, ...]:
    if len(scores) < size:
        raise ValueError("合法候选池不足以生成指定区域")
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return tuple(key for key, _ in ranked[:size])


def processing_durations(schedule) -> dict[OperationKey, float]:
    return {
        (block.job, block.opera): block.end - block.start
        for table in schedule.machine_tables
        for block in table
    }


def build_coupled_scores(data, schedule, diagnosis) -> dict[OperationKey, float]:
    durations = processing_durations(schedule)
    operation_counts = tuple(data.instance.operation_counts)
    scores = {}
    for key, transport in diagnosis.contributions.items():
        job, operation = key
        successors = operation_counts[job - 1] - operation
        scores[key] = coupled_score(
            transport=transport,
            processing=durations.get(key, 0.0),
            successors=successors,
        )
    return scores


def run_probe(seeds=range(10), region_size: int = 5, budget: int = 50):
    data, chromosome, schedule, event, plan = load_mk05_case()
    residual = plan.rescheduled_operations
    diagnosis = diagnose_agv_bottleneck(schedule, residual, event.time)
    legal_pool = frozenset(diagnosis.contributions)
    direct_region = select_bottleneck_region(diagnosis, residual, region_size)
    coupled_region = select_region(build_coupled_scores(data, schedule, diagnosis), region_size)
    operation_counts = tuple(data.instance.operation_counts)
    base_schedule = _decode_dynamic(data, chromosome, schedule, event)
    original_objective = (base_schedule.makespan, base_schedule.machine_energy)
    rows = []

    for seed in seeds:
        random_region = tuple(random.Random(seed).sample(sorted(legal_pool), region_size))
        for group, region in (
            ("direct_load", direct_region),
            ("coupled_propagation", coupled_region),
            ("random", random_region),
        ):
            candidates = generate_as_candidates(
                chromosome, operation_counts, region, data.agv.count, budget, seed
            )
            results, elapsed = evaluate_candidates(
                data, candidates, chromosome, schedule, event
            )
            summary = summarize_results(results, original_objective, elapsed)
            summary.update(
                {
                    "seed": seed,
                    "group": group,
                    "region": " ".join(f"J{job}O{operation}" for job, operation in region),
                }
            )
            rows.append(summary)

    grouped = {
        group: [row for row in rows if row["group"] == group]
        for group in ("direct_load", "coupled_propagation", "random")
    }
    summary = []
    for group, group_rows in grouped.items():
        summary.append(
            {
                "group": group,
                "seed_count": len(group_rows),
                "mean_improvement_rate": fmean(
                    row["dominant_improvement_rate"] for row in group_rows
                ),
                "mean_best_makespan": fmean(row["best_makespan"] for row in group_rows),
                "mean_best_energy": fmean(
                    row["best_energy_objective"] for row in group_rows
                ),
                "mean_wall_time": fmean(row["wall_time"] for row in group_rows),
            }
        )
    direct = next(row for row in summary if row["group"] == "direct_load")
    coupled = next(row for row in summary if row["group"] == "coupled_propagation")
    coupled["go"] = (
        coupled["mean_improvement_rate"] > direct["mean_improvement_rate"]
        and coupled["mean_best_makespan"] <= direct["mean_best_makespan"]
    )
    return direct_region, coupled_region, rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if getattr(sys.stdout, "encoding", "").lower() != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    direct_region, coupled_region, rows, summary = run_probe()
    write_csv(OUTPUT / "probe_11_seed_results.csv", rows)
    write_csv(OUTPUT / "probe_11_summary.csv", summary)
    print("直接负载Top-5：", direct_region)
    print("耦合传播Top-5：", coupled_region)
    for row in summary:
        print(
            row["group"],
            "改善率=", round(row["mean_improvement_rate"], 4),
            "Makespan=", round(row["mean_best_makespan"], 2),
            "耗时=", round(row["mean_wall_time"], 4),
            "Go=", row.get("go", ""),
        )


if __name__ == "__main__":
    main()
