"""四个离线生成的候选邻域，对应 EoH 的 i1/e1/e2/m1。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from python_baseline.dfjspt.chromosome import Chromosome
from python_baseline.dfjspt.data import ExperimentInput


Operator = Callable[[ExperimentInput, Chromosome, random.Random], Chromosome]


def _copy(
    chromosome: Chromosome, *, os=None, ms=None, agv=None, empty=None, loaded=None
) -> Chromosome:
    return Chromosome(
        tuple(chromosome.os if os is None else os),
        tuple(chromosome.ms if ms is None else ms),
        tuple(chromosome.agv if agv is None else agv),
        tuple(chromosome.empty_speed if empty is None else empty),
        tuple(chromosome.loaded_speed if loaded is None else loaded),
    )


def i1_machine_speed_pair(
    data: ExperimentInput, chromosome: Chromosome, rng: random.Random
) -> Chromosome:
    """Purpose: 联动修改候选机器与载货速度；Input/Output: 标准三参数/新染色体。"""
    del rng
    choices = []
    flat = 0
    for job in data.instance.jobs:
        for operation in job.operations:
            if len(operation.options) > 1:
                times = [option.processing_time for option in operation.options]
                choices.append((max(times) - min(times), flat, times))
            flat += 1
    if not choices:
        loaded = list(chromosome.loaded_speed)
        loaded[0] = max(range(len(data.agv.speeds)), key=data.agv.speeds.__getitem__)
        return _copy(chromosome, loaded=loaded)
    _, position, times = max(choices)
    ms = list(chromosome.ms)
    best = min(range(len(times)), key=times.__getitem__)
    ms[position] = best if best != ms[position] else (best + 1) % len(times)
    loaded = list(chromosome.loaded_speed)
    loaded[position] = max(range(len(data.agv.speeds)), key=data.agv.speeds.__getitem__)
    return _copy(chromosome, ms=ms, loaded=loaded)


def e1_order_agv_exchange(
    data: ExperimentInput, chromosome: Chromosome, rng: random.Random
) -> Chromosome:
    """Purpose: 将 OS 交换与 AGV 重分配组合；Input/Output: 标准三参数/新染色体。"""
    os = list(chromosome.os)
    pairs = [(a, b) for a in range(len(os)) for b in range(a + 1, len(os)) if os[a] != os[b]]
    if pairs:
        first, second = rng.choice(pairs)
        os[first], os[second] = os[second], os[first]
    agv = list(chromosome.agv)
    position = rng.randrange(len(agv))
    agv[position] = (agv[position] + 1 + rng.randrange(data.agv.count - 1)) % data.agv.count
    return _copy(chromosome, os=os, agv=agv)


def e2_machine_transport_backbone(
    data: ExperimentInput, chromosome: Chromosome, rng: random.Random
) -> Chromosome:
    """Purpose: 从机器/运输共同骨架生成协同扰动；Input/Output: 标准三参数/新染色体。"""
    candidates = []
    flat = 0
    for job in data.instance.jobs:
        for operation in job.operations:
            if len(operation.options) > 1:
                times = [option.processing_time for option in operation.options]
                candidates.append((sum(times) / len(times), flat, times))
            flat += 1
    position = max(candidates)[1] if candidates else rng.randrange(chromosome.operation_count)
    ms = list(chromosome.ms)
    if candidates:
        times = next(row[2] for row in candidates if row[1] == position)
        alternatives = [index for index in range(len(times)) if index != ms[position]]
        ms[position] = min(alternatives, key=times.__getitem__)
    agv = list(chromosome.agv)
    agv[position] = (agv[position] + 1) % data.agv.count
    empty = list(chromosome.empty_speed)
    empty[position] = min(range(len(data.agv.speeds)), key=data.agv.speeds.__getitem__)
    return _copy(chromosome, ms=ms, agv=agv, empty=empty)


def m1_rewrite_transport_gene(
    data: ExperimentInput, chromosome: Chromosome, rng: random.Random
) -> Chromosome:
    """Purpose: 将单 AS 改写扩展为同位运输三基因改写；Input/Output: 标准三参数/新染色体。"""
    position = rng.randrange(chromosome.operation_count)
    agv = list(chromosome.agv)
    empty = list(chromosome.empty_speed)
    loaded = list(chromosome.loaded_speed)
    agv[position] = (agv[position] + 1 + rng.randrange(data.agv.count - 1)) % data.agv.count
    empty[position] = (empty[position] + 1 + rng.randrange(len(data.agv.speeds) - 1)) % len(data.agv.speeds)
    loaded[position] = (loaded[position] + 1 + rng.randrange(len(data.agv.speeds) - 1)) % len(data.agv.speeds)
    return _copy(chromosome, agv=agv, empty=empty, loaded=loaded)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    generation_operator: str
    parents: tuple[str, ...]
    model: str
    seed: int
    operator: Operator


CANDIDATES = (
    Candidate("llm_i1_machine_speed", "i1", (), "OpenAI Codex GPT-5", 101, i1_machine_speed_pair),
    Candidate("llm_e1_order_agv", "e1", ("N1", "N2", "N5"), "OpenAI Codex GPT-5", 102, e1_order_agv_exchange),
    Candidate("llm_e2_machine_transport", "e2", ("N3", "N4", "N5", "N6"), "OpenAI Codex GPT-5", 103, e2_machine_transport_backbone),
    Candidate("llm_m1_transport_rewrite", "m1", ("N5", "N6"), "OpenAI Codex GPT-5", 104, m1_rewrite_transport_gene),
)
