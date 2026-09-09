"""把冻结A0排程转换为第三轮动作前状态；本模块不得调用完整解码。"""

import math
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECOND = ROOT.parent / "KNb深化第二轮" / "scripts"
sys.path.insert(0, str(SECOND))
import run_experiments as second

from mechanisms_v3 import feasible_neighborhoods, k_candidates


old = second.old


def _operation_offsets(counts):
    offsets = []
    total = 0
    for count in counts:
        offsets.append(total)
        total += count
    return offsets


def _table_operations(data, tables):
    offsets = _operation_offsets(data.instance.operation_counts)
    return [
        {offsets[block.job - 1] + block.opera - 1 for block in table if block.job}
        for table in tables
    ]


def _os_opportunity_rates(data, chromosome, selected, n1_samples=64, n2_samples=64):
    counts = data.instance.operation_counts
    size = chromosome.operation_count
    n1_valid = 0
    n1_rng = random.Random(old.seed_of("n1-opportunity", chromosome.os, tuple(sorted(selected))))
    if size >= 3:
        for _ in range(n1_samples):
            child = old.nb.n1_reinsert_reversed_pair(
                chromosome, n1_rng.randrange(size - 1), n1_rng.randrange(1, size - 1)
            )
            changed = old.touched(chromosome, child, counts)
            n1_valid += int(bool(changed) and changed <= selected)
    rng = random.Random(old.seed_of("n2-opportunity", chromosome.os, tuple(sorted(selected))))
    n2_valid = 0
    for _ in range(n2_samples if size >= 3 else 0):
        positions = tuple(rng.sample(range(size), 2))
        insert_after = rng.randrange(1, size - 1)
        child = old.nb.n2_remove_and_reinsert(chromosome, positions, insert_after)
        changed = old.touched(chromosome, child, counts)
        n2_valid += int(bool(changed) and changed <= selected)
    return (
        n1_valid / n1_samples if n1_samples and size >= 3 else 0.0,
        n2_valid / n2_samples if n2_samples and size >= 3 else 0.0,
    )


def opportunity_profile(data, chromosome, schedule, selected, n1_samples=64, n2_samples=64):
    """计算0—1动作前可执行机会率；正值不承诺候选一定改善。"""
    operations = [operation for job in data.instance.jobs for operation in job.operations]
    flexible = {index for index, operation in enumerate(operations) if len(operation.options) > 1}
    machine_operations = _table_operations(data, schedule.machine_tables)
    loads = [sum(block.job != 0 for block in table) for table in schedule.machine_tables]
    maximum_machine = max(range(len(loads)), key=lambda index: (loads[index], -index))
    agv_alternatives = max(0, data.agv.count - 1)
    denominator = max(1, len(selected))
    n1_rate, n2_rate = _os_opportunity_rates(data, chromosome, selected, n1_samples, n2_samples)
    return {
        1: n1_rate,
        2: n2_rate,
        3: len(selected & flexible) / denominator,
        4: len(selected & flexible & machine_operations[maximum_machine]) / denominator,
        5: float(bool(selected) and agv_alternatives > 0),
        6: float(bool(selected) and agv_alternatives > 0),
    }


def _coverage(values, selected):
    total = sum(max(0.0, value) for value in values)
    if total == 0:
        return len(selected) / len(values) if values else 0.0
    return sum(max(0.0, values[index]) for index in selected) / total


def _conservative_feasible(opportunities, operation_count, selected):
    """抽样率只描述机会强弱，不得把N1/N2的抽样零当成不可行。"""
    feasible = set(feasible_neighborhoods(opportunities))
    if operation_count >= 3 and len(selected) >= 2:
        feasible.update((1, 2))
    return tuple(sorted(feasible))


def region_context(data, chromosome, schedule, minimum_coverage_gain):
    """生成A25/A50覆盖、机会向量和保守掩码，全程零新增解码。"""
    _, agv_scores, _ = old.scores(data, schedule)
    operation_count = chromosome.operation_count
    count25 = max(1, math.ceil(operation_count * 0.25))
    count50 = max(1, math.ceil(operation_count * 0.50))
    region25 = old.region([0.0] * operation_count, agv_scores, count25, "A", 0)
    region50 = old.region([0.0] * operation_count, agv_scores, count50, "A", 0)
    opportunities25 = opportunity_profile(data, chromosome, schedule, region25)
    opportunities50 = opportunity_profile(data, chromosome, schedule, region50)
    coverage25 = _coverage(agv_scores, region25)
    coverage50 = _coverage(agv_scores, region50)
    candidates = k_candidates(
        coverage25,
        coverage50,
        sum(opportunities25.values()),
        sum(opportunities50.values()),
        minimum_coverage_gain,
    )
    return {
        "coverage25": coverage25,
        "coverage50": coverage50,
        "region25": region25,
        "region50": region50,
        "opportunities25": opportunities25,
        "opportunities50": opportunities50,
        "feasible25": _conservative_feasible(opportunities25, operation_count, region25),
        "feasible50": _conservative_feasible(opportunities50, operation_count, region50),
        "k_candidates": candidates,
    }


def discrete_state(expand, opportunities, recent_batches, remaining, total_budget):
    feasible = [(count, n) for n, count in opportunities.items() if count > 0]
    dominant = min((n for count, n in feasible if count == max(row[0] for row in feasible)), default=0)
    if not recent_batches:
        trend = 0
    else:
        gain, decodes = recent_batches[-1]
        trend = 1 if decodes > 0 and gain / decodes > 0 else 2
    ratio = remaining / total_budget if total_budget else 0.0
    budget_level = 2 if ratio > 2 / 3 else 1 if ratio > 1 / 3 else 0
    return int(bool(expand)), dominant, trend, budget_level
