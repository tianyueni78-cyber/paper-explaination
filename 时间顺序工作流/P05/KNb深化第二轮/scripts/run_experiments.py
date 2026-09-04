"""第二轮局部实验；A0只读，拒绝覆盖输出。"""
import argparse
import gzip
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'KNb深挖/scripts'))
import probe_knb as previous
from mechanisms import rank_region, state_id, choose_n, choose_b, remember

old = previous.old


def validate_protocol(protocol):
    # 此版本是冻结实验，不是任意参数配置器；改协议必须同步实现和测试。
    expected = dict(version='knb2-explore-v1', scope='local_mechanism_exploration_not_confirmation',
        instances=[f'Mk{i:02}' for i in range(1, 7)], heldout_forbidden=['Mk07', 'Mk08'],
        repeats=[0, 1, 2], fractions=[0.25, 0.5], budget=96, depths=[4, 12], window=3,
        state_thresholds=[0.5, 0.5], explore_every=4, max_batches=96, attempts_per_decode=40,
        timeout_seconds=1800, primary='final_local_archive_HV_gain',
        practical_effect_threshold=None, performance_claims_from_smoke=False)
    if protocol != expected:
        raise ValueError('协议与当前冻结实现不一致，禁止静默改变实验')


def batch(data, parent, schedule, selected, n, allowance, seed, points, base, seen, failure_record=None):
    rng = random.Random(seed)
    before = old.hv(points)
    best = None
    candidates = []
    counts = dict(attempts=0, outside=0, noop=0, duplicate=0, evaluated=0)
    initial_parent = parent
    with old.Meter(allowance) as meter:
        if failure_record is not None:
            failure_record.update(counts=counts, candidates=candidates, meter=meter)
        while meter.count < allowance and counts['attempts'] < 40 * allowance:
            if allowance - meter.count < (2 if n in (3, 5) else 1):
                break
            counts['attempts'] += 1
            child = old.nb.apply_neighborhood(data, initial_parent, n, rng)
            changed = old.touched(initial_parent, child, data.instance.operation_counts)
            if not changed:
                counts['noop'] += 1
                continue
            if not changed <= selected:
                counts['outside'] += 1
                continue
            if child in seen:
                counts['duplicate'] += 1
                continue
            seen.add(child)
            result = meter.decode(data, child)
            old.validate_schedule(data, child, result)
            assert child.empty_speed == initial_parent.empty_speed
            assert child.loaded_speed == initial_parent.loaded_speed
            objective = (result.makespan, result.machine_energy)
            point = (objective[0] / base[0], objective[1] / base[1])
            delta = old.hv(points + [point]) - before
            key = (-delta, objective, child.os, child.ms, child.agv)
            if delta > 0 and (best is None or key < best):
                best, parent, schedule = key, child, result
            counts['evaluated'] += 1
            candidates.append(dict(chromosome=asdict(child), objective=objective, decode_index=meter.count))
    points.extend((r['objective'][0] / base[0], r['objective'][1] / base[1]) for r in candidates)
    return parent, schedule, dict(**counts, decodes=meter.count, gain=max(0.0, old.hv(points) - before),
        candidates=candidates, accepted=asdict(parent) if best is not None else None)


def features(data, schedule, group, fraction, seed):
    _, l, details = previous.wait_features(data, schedule)
    alternatives = [len({option.machine_id for option in op.options}) > 1
                    for job in data.instance.jobs for op in job.operations]
    offsets = old.nb._offsets(data)
    transported = {offsets[b.job - 1] + b.opera - 1 for table in schedule.agv_tables for b in table
                   if b.job and b.opera > 0 and b.load_status == -2 and not b.charge}
    agv = [i in transported and data.agv.count > 1 for i in range(len(l))]
    k = math.ceil(len(l) * fraction)
    if group == 'K':
        selected = rank_region(l, alternatives, agv, k)
    else:
        selected = previous.regions(data, schedule, k, seed)[group]
    state = state_id([r['transport_wait'] for r in details], [r['machine_wait'] for r in details],
                     alternatives, selected)
    return selected, state


def walk(data, parent, schedule, config, fraction, seed, budget=96, deadline=None):
    if data.agv.count < 2:
        raise ValueError('本探索协议只支持当前多AGV配置，防止原N5单AGV死循环')
    group, selector, depth = config
    if (group not in ('A', 'L', 'K', 'R') or
        selector not in ('fixed', 'global', 'state', 'shuffled', *[f'N{i}' for i in range(1, 7)]) or
        depth not in ('4', '12', 'recent', 'cumulative')):
        raise ValueError('未定义的机制配置')
    points = [(1.0, 1.0)]
    base = (schedule.makespan, schedule.machine_energy)
    seen = {parent}
    histories = defaultdict(dict)
    visits = defaultdict(lambda: [0] * 6)
    cursors = defaultdict(int)
    decisions = defaultdict(int)
    depth_history = {}
    total = 0
    trace = []
    started = time.perf_counter()
    control_seconds = 0.0
    while total < budget and len(trace) < 96:
        step = len(trace)
        tick = time.perf_counter()
        selected, observed = features(data, schedule, group, fraction, old.seed_of(seed, 'region', step))
        state = observed if selector == 'state' else 0
        if selector == 'shuffled':
            state = random.Random(old.seed_of(seed, 'shuffle', step)).randrange(4)
        if selector.startswith('N'):
            n = int(selector[1:]) - 1
        elif selector == 'fixed':
            n = 2
        else:
            n, cursors[state] = choose_n(histories[state], visits[state], decisions[state], cursors[state])
        allowance = min(int(depth), budget - total) if depth.isdigit() else choose_b(depth_history, n, budget - total)
        if allowance < (2 if n in (3, 5) else 1):
            break
        stream = old.seed_of(seed, 'candidate', step, n)
        previous_parent = parent
        control_seconds += time.perf_counter() - tick
        failure_record = {}
        try:
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError('探索阶段30分钟上限，批次边界停止')
            parent, schedule, result = batch(data, parent, schedule, selected, n, allowance, stream,
                                             points, base, seen, failure_record)
        except BaseException as error:
            failed_cost = failure_record['meter'].count if failure_record else 0
            error.partial_result = dict(status='failed', error=repr(error), trace=trace,
                decodes=total + failed_cost, budget=budget, failed_batch=dict(step=step, n=n+1,
                allowance=allowance, rng_seed=stream, parent=asdict(previous_parent), region=sorted(selected),
                decodes=failed_cost, counts=failure_record.get('counts', {}),
                candidates=failure_record.get('candidates', [])))
            raise
        tick = time.perf_counter()
        visits[state][n] += 1
        decisions[state] += 1
        remember(histories[state], n, result['gain'], result['decodes'])
        remember(depth_history, n, result['gain'], result['decodes'], window=None if depth == 'cumulative' else 3)
        total += result['decodes']
        control_seconds += time.perf_counter() - tick
        trace.append(dict(step=step, state=state, observed_state=observed, n=n+1, allowance=allowance,
                          total_decodes=total, parent=asdict(previous_parent), region=sorted(selected),
                          rng_seed=stream, **result))
        if deadline is not None and time.perf_counter() > deadline:
            error = TimeoutError('探索阶段30分钟上限，批后边界停止')
            error.partial_result = dict(status='failed', error=repr(error), trace=trace,
                                        decodes=total, budget=budget, failed_batch=None)
            raise error
    return dict(decodes=total, budget=budget, gain=max(0.0, old.hv(points)-old.hv([(1, 1)])),
                trace=trace, control_seconds=control_seconds, wall_seconds=time.perf_counter()-started)


def configurations(stage):
    if stage == 'K':
        return [(g, f'N{n}', '4') for g in ('A', 'L', 'K', 'R') for n in range(1, 7)]
    if stage == 'N':
        return [(g, n, '4') for g in ('A', 'K') for n in ('fixed', 'global', 'state', 'shuffled')]
    if stage == 'b':
        return [('A', 'global', b) for b in ('4', '12', 'cumulative', 'recent')]
    if stage in ('interaction', 'smoke'):
        return [(g, n, b) for g in ('A', 'K') for n in ('fixed', 'state') for b in ('4', 'recent')]
    raise ValueError(stage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('K', 'N', 'b', 'interaction', 'smoke'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    validate_protocol(protocol)
    states_path = ROOT.parent / 'KNb深挖/runs/budget-corrected/states.json'
    states = json.loads(states_path.read_text(encoding='utf-8'))
    if any(s['instance'] not in protocol['instances'] for s in states):
        raise ValueError('拒绝非探索实例')
    smoke = args.stage == 'smoke'
    if smoke:
        states = states[:3]
    args.output.mkdir(parents=True, exist_ok=False)
    baseline_manifest = json.loads((states_path.parent/'manifest.json').read_text(encoding='utf-8'))
    before = {name: old.sha(old.BASE/name) for name in baseline_manifest['source_hashes']}
    if before != baseline_manifest['source_hashes']:
        raise ValueError('A0来源变更')
    manifest = dict(status='running', stage=args.stage, performance_claims=False,
        protocol_sha256=old.sha(ROOT/'protocol.json'), states_sha256=old.sha(states_path),
        scripts={f.name: old.sha(f) for f in Path(__file__).parent.glob('*.py')},
        inherited_scripts={str(f): old.sha(f) for f in (Path(previous.__file__), Path(old.__file__))},
        python=sys.version, source_commit=baseline_manifest['source_commit'], source_hashes=before,
        acquisition_decodes=0, search_decodes=0, trajectories=0)
    old.dump(args.output/'manifest.json', manifest)
    started = time.perf_counter()
    try:
        with gzip.open(args.output/'trajectories.jsonl.gz', 'wt', encoding='utf-8') as raw:
            for state in states:
                data = old.load_case(state['instance'])
                parent = old.Chromosome(**{key: tuple(value) for key, value in state['chromosome'].items()})
                manifest['acquisition_decodes'] += 1
                schedule = old.decode_static(data, parent)
                for fraction in ([0.5] if smoke else protocol['fractions']):
                    for rep in ([0] if smoke else protocol['repeats']):
                        seed = old.seed_of('knb2', state['state'], fraction, rep)
                        for config in configurations(args.stage):
                            if time.perf_counter()-started > protocol['timeout_seconds']:
                                raise TimeoutError('探索阶段30分钟上限')
                            try:
                                result = walk(data, parent, schedule, config, fraction, seed,
                                    24 if smoke else protocol['budget'], started + protocol['timeout_seconds'])
                            except BaseException as error:
                                partial = getattr(error, 'partial_result', dict(decodes=0, error=repr(error)))
                                raw.write(json.dumps(dict(state=state['state'], config=config,
                                    fraction=fraction, rep=rep, seed=seed, **partial))+'\n')
                                raw.flush()
                                manifest['search_decodes'] += partial['decodes']
                                raise
                            raw.write(json.dumps(dict(state=state['state'], instance=state['instance'], split='explore',
                                generation=state['generation'], initial_seed=state['seed'], config=config,
                                fraction=fraction, rep=rep, seed=seed, **result))+'\n')
                            manifest['search_decodes'] += result['decodes']
                            manifest['trajectories'] += 1
                print(state['state'], manifest['trajectories'], flush=True)
        assert before == {name: old.sha(old.BASE/name) for name in before}
        manifest.update(status='completed', source_unchanged=True)
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest['seconds'] = time.perf_counter()-started
        if (args.output/'trajectories.jsonl.gz').exists():
            manifest['raw_sha256'] = old.sha(args.output/'trajectories.jsonl.gz')
        old.dump(args.output/'manifest.json', manifest)


if __name__ == '__main__':
    main()
