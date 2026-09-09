"""第三轮真实计费局部运行器；冻结A0只读。"""

import argparse
import json
import math
import platform
import sys
import time
from itertools import product
from pathlib import Path

from adapter_v3 import discrete_state, old, region_context, second
from contract_v3 import validate_protocol
from mechanisms_v3 import (
    FactorizedQ,
    confidence_action,
    decay_state_credit,
    next_budget_tranche,
    state_change_score,
)


ROOT = Path(__file__).resolve().parents[1]
FALLBACK = (0.25, 3, 4)


def _actions(context, modules=(True, True, True)):
    use_k, use_n, use_b = modules
    actions = []
    k_values = context["k_candidates"] if use_k else (0.25,)
    for k in k_values:
        feasible = context["feasible25"] if k == 0.25 else context["feasible50"]
        n_values = feasible if use_n else (3,)
        b_values = (4, 8, 12) if use_b else (4,)
        actions.extend(product((k,), n_values, b_values))
    return sorted(actions)


def _rule_action(context, modules):
    use_k, use_n, use_b = modules
    k = 0.5 if use_k and 0.5 in context["k_candidates"] else 0.25
    opportunities = context["opportunities50"] if k == 0.5 else context["opportunities25"]
    feasible = context["feasible50"] if k == 0.5 else context["feasible25"]
    n = min(feasible, key=lambda item: (-opportunities[item], item)) if use_n and feasible else 3
    return k, n, 12 if use_b else 4


def choose_joint(table, state, actions, covered, minimum_gap):
    if not actions:
        return FALLBACK, "fallback"
    missing = {
        action: int(action[0] not in covered["K"])
        + int(action[1] not in covered["N"])
        + int(action[2] not in covered["b"])
        for action in actions
    }
    if max(missing.values()) > 0:
        return min(actions, key=lambda action: (-missing[action], action)), "cold_start"
    values = {action: table.value(state, action) for action in actions}
    selected = confidence_action(values, minimum_gap)
    return (selected, "adaptive") if selected is not None else (FALLBACK, "fallback")


def _state(context, recent_batches, remaining, total_budget):
    return discrete_state(
        len(context["k_candidates"]) > 1,
        context["opportunities25"],
        recent_batches,
        remaining,
        total_budget,
    )


def _fixed_context(data, parent, schedule, _minimum_gain):
    """固定对照只计算自身需要的A25区域，不承担创新特征开销。"""
    _, agv_scores, _ = old.scores(data, schedule)
    selected = old.region(
        [0.0] * parent.operation_count,
        agv_scores,
        max(1, math.ceil(parent.operation_count * 0.25)),
        "A",
        0,
    )
    opportunities = {n: float(n == 3) for n in range(1, 7)}
    return {
        "coverage25": 0.0,
        "coverage50": 0.0,
        "region25": selected,
        "region50": selected,
        "opportunities25": opportunities,
        "opportunities50": opportunities,
        "feasible25": (3,),
        "feasible50": (3,),
        "k_candidates": (0.25,),
    }


def _q_snapshot(table):
    return [(repr(key), value) for key, value in sorted(table.terms.items(), key=lambda row: repr(row[0]))]


def walk(data, parent, schedule, seed, budget, protocol, modules=(True, True, True), controller="q"):
    """运行一条独立局部轨迹；Q、覆盖记录和档案每次重置。"""
    base = schedule.makespan, schedule.machine_energy
    points = [(1.0, 1.0)]
    seen = {parent}
    table = FactorizedQ()
    covered = {"K": set(), "N": set(), "b": set()}
    recent_batches = []
    trace = []
    total = 0
    q_updates = 0
    context_builder = _fixed_context if controller == "fixed" and not any(modules) else region_context
    context_cache = {}

    def get_context(current_parent, current_schedule):
        if current_parent not in context_cache:
            context_cache[current_parent] = context_builder(
                data, current_parent, current_schedule, protocol["k_minimum_coverage_gain"]
            )
        return context_cache[current_parent]

    while total < budget and len(trace) < protocol["maximum_decisions"]:
        context = get_context(parent, schedule)
        state = _state(context, recent_batches, budget - total, budget)
        actions = _actions(context, modules)
        if controller == "fixed":
            action, mode = FALLBACK, "fixed"
        elif controller == "rule":
            action, mode = _rule_action(context, modules), "rule"
        elif controller == "q":
            action, mode = choose_joint(table, state, actions, covered, protocol["confidence_minimum_gap"])
        else:
            raise ValueError("controller必须为fixed、rule或q")
        k, n, maximum = action
        action_batches = []
        action_decodes = 0
        action_gain = 0.0
        old_region = context["region25"] if k == 0.25 else context["region50"]
        old_opportunities = context["opportunities25"] if k == 0.25 else context["opportunities50"]
        tranche_rows = []

        while total < budget:
            allowance = next_budget_tranche(
                action_decodes,
                action_batches,
                budget - total,
                protocol["continue_minimum_gain_per_decode"],
                maximum=maximum,
                tranche=protocol["budget_tranche"],
            )
            if allowance <= 0:
                break
            current = get_context(parent, schedule)
            selected = current["region25"] if k == 0.25 else current["region50"]
            parent, schedule, result = second.batch(
                data,
                parent,
                schedule,
                selected,
                n - 1,
                allowance,
                old.seed_of("knb3", seed, len(trace), len(tranche_rows), action),
                points,
                base,
                seen,
            )
            tranche_rows.append({key: value for key, value in result.items() if key != "candidates"})
            action_decodes += result["decodes"]
            total += result["decodes"]
            action_gain += result["gain"]
            action_batches.append((result["gain"], result["decodes"]))
            recent_batches[:] = action_batches[-1:]
            if result["decodes"] == 0:
                break

        new_context = get_context(parent, schedule)
        next_state = _state(new_context, recent_batches, budget - total, budget)
        change = state_change_score(
            old_region,
            new_context["region25"] if k == 0.25 else new_context["region50"],
            tuple(old_opportunities.values()),
            tuple((new_context["opportunities25"] if k == 0.25 else new_context["opportunities50"]).values()),
        )
        reward = action_gain / action_decodes if action_decodes else 0.0
        if action_decodes > 0 and controller == "q":
            if change >= protocol["state_change_threshold"]:
                decay_state_credit(table, state, protocol["credit_decay_factor"])
            table.update(
                state,
                action,
                reward,
                next_state,
                _actions(new_context, modules),
                protocol["alpha"],
                protocol["gamma"],
            )
            q_updates += 1
        covered["K"].add(k)
        covered["N"].add(n)
        covered["b"].add(maximum)
        trace.append(
            {
                "decision": len(trace),
                "state": state,
                "action": action,
                "mode": mode,
                "decodes": action_decodes,
                "actual_cap": action_decodes,
                "gain": action_gain,
                "reward": reward,
                "q_updated": action_decodes > 0,
                "state_change": change,
                "tranches": tranche_rows,
            }
        )
    return {
        "decodes": total,
        "budget": budget,
        "gain": max(0.0, old.hv(points) - old.hv([(1.0, 1.0)])),
        "trace": trace,
        "q_terms": _q_snapshot(table),
        "q_updates": q_updates,
    }


def _dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    validate_protocol(protocol)
    source = ROOT.parent / "KNb深挖" / "runs" / "budget-corrected"
    states_path = source / "states.json"
    states = json.loads(states_path.read_text(encoding="utf-8"))
    state = states[0]
    if state["instance"] not in protocol["instances"]:
        raise ValueError("冒烟输入不属于开发实例")
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    source_hashes = source_manifest["source_hashes"]
    if source_hashes != {name: old.sha(old.BASE / name) for name in source_hashes}:
        raise ValueError("冻结A0来源已经变化")

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "stage": args.stage,
        "performance_claims": False,
        "validation_data_opened": False,
        "protocol_sha256": old.sha(ROOT / "protocol.json"),
        "states_sha256": old.sha(states_path),
        "scripts": {path.name: old.sha(path) for path in Path(__file__).parent.glob("*.py")},
        "source_commit": source_manifest["source_commit"],
        "source_hashes": source_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "instance": state["instance"],
        "seed": 91,
        "acquisition_decodes": 0,
        "search_decodes": 0,
    }
    _dump(args.output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        data = old.load_case(state["instance"])
        parent = old.Chromosome(**{key: tuple(value) for key, value in state["chromosome"].items()})
        schedule = old.decode_static(data, parent)
        manifest["acquisition_decodes"] = 1
        result = walk(data, parent, schedule, manifest["seed"], protocol["smoke_budget"], protocol)
        manifest["search_decodes"] = result["decodes"]
        _dump(args.output / "result.json", result)
        if source_hashes != {name: old.sha(old.BASE / name) for name in source_hashes}:
            raise ValueError("运行过程中冻结A0来源发生变化")
        manifest.update(status="completed", source_unchanged=True)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["seconds"] = time.perf_counter() - started
        if (args.output / "result.json").exists():
            manifest["result_sha256"] = old.sha(args.output / "result.json")
        _dump(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
