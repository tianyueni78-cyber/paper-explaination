"""候选的最小隔离验证与动态离线评价。"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import random
import tempfile
import time
from hashlib import sha256
from pathlib import Path

from python_baseline.dfjspt.decoder import decode_static
from python_baseline.dfjspt.dynamic import DynamicEvent, _decode_dynamic
from python_baseline.dfjspt.initialization import hybrid_population
from python_baseline.dfjspt.metrics import hypervolume_2d, normalize_groups
from python_baseline.dfjspt.multiobjective import pareto_indices
from python_baseline.dfjspt.neighborhoods import apply_neighborhood

from .operators import CANDIDATES


def _operator_worker(result_path, operator, data, chromosome, seed):
    try:
        result = operator(data, chromosome, random.Random(seed))
        result.validate(data.instance, data.agv.count, len(data.agv.speeds))
        payload = ("valid", result)
    except Exception as error:
        payload = ("failed", f"{type(error).__name__}: {error}")
    Path(result_path).write_bytes(pickle.dumps(payload))


def validate_operator(operator, data, chromosome, *, seed: int, timeout_seconds: float):
    """Purpose: 隔离接口/合法性异常与超时；Input: 算子和样本；Output: 状态字典。"""
    context = multiprocessing.get_context("spawn")
    temp_root = Path.cwd() / ".codex_tmp"
    temp_root.mkdir(exist_ok=True)
    handle, name = tempfile.mkstemp(suffix=".pkl", dir=temp_root)
    os.close(handle)
    result_path = Path(name)
    try:
        process = context.Process(
            target=_operator_worker,
            args=(str(result_path), operator, data, chromosome, seed),
        )
        process.start()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(2)
            return {"status": "timeout", "error": "operator timeout"}
        if not result_path.stat().st_size:
            return {"status": "failed", "error": "operator process returned no result"}
        status, payload = pickle.loads(result_path.read_bytes())
    finally:
        result_path.unlink(missing_ok=True)
    if status == "failed":
        return {"status": status, "error": payload}
    return {"status": status, "chromosome": payload}


def _seed(seed: int, operator_id: str) -> int:
    return int.from_bytes(sha256(f"{seed}:{operator_id}".encode()).digest()[:8], "big")


def _operators():
    rows = [
        (f"N{action + 1}", "baseline", lambda data, chromosome, rng, action=action:
         apply_neighborhood(data, chromosome, action, rng))
        for action in range(6)
    ]
    rows.extend((item.candidate_id, "llm", item.operator) for item in CANDIDATES)
    return rows


def run_case(data, event: DynamicEvent, *, instance: str, seed: int, decode_budget: int):
    """Purpose: 同预算比较十个算子；Input: 动态实例/事件；Output: 十行指标。"""
    if decode_budget <= 0:
        raise ValueError("decode_budget 必须为正")
    base = hybrid_population(
        data, 10, len(data.agv.speeds), random.Random(seed)
    ).chromosomes[0]
    original = decode_static(data, base)
    rows = []
    fronts = []
    for operator_id, source, operator in _operators():
        rng = random.Random(_seed(seed, operator_id))
        current = base
        current_schedule = _decode_dynamic(data, current, original, event)
        current_objective = (current_schedule.makespan, current_schedule.machine_energy)
        objectives = []
        feasible = changed = errors = decodes = 0
        started = time.perf_counter()
        for _ in range(decode_budget):
            try:
                candidate = operator(data, current, rng)
                candidate.validate(data.instance, data.agv.count, len(data.agv.speeds))
                if candidate != current:
                    changed += 1
                schedule = _decode_dynamic(data, candidate, original, event)
                decodes += 1
                feasible += 1
                objective = (schedule.makespan, schedule.machine_energy)
                objectives.append(objective)
                # 与原 QNSGA-II 相同：至少一个目标改善时接受为下一父代。
                if not all(old <= new for old, new in zip(current_objective, objective)):
                    current = candidate
                    current_objective = objective
            except Exception:
                errors += 1
        front = [objectives[index] for index in pareto_indices(objectives)] if objectives else []
        fronts.append(front)
        rows.append({
            "instance": instance,
            "event_kind": event.kind,
            "event_time": event.time,
            "event_target": event.target,
            "seed": seed,
            "operator_id": operator_id,
            "source": source,
            "decode_budget": decode_budget,
            "decode_count": decodes,
            "feasible_count": feasible,
            "feasible_rate": feasible / decode_budget,
            "changed_count": changed,
            "min_makespan": min((item[0] for item in objectives), default=""),
            "min_tec": min((item[1] for item in objectives), default=""),
            "hv": "",
            "runtime_seconds": time.perf_counter() - started,
            "error_count": errors,
        })
    if all(fronts):
        try:
            normalized = normalize_groups(fronts)
            for row, front in zip(rows, normalized):
                row["hv"] = hypervolume_2d(front)
        except ValueError:
            pass
    return rows
