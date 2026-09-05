"""固定提案池补样；不按区域拒绝评价，不更新父解，不训练控制器。"""
import argparse
import gzip
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from run_experiments import ROOT, old, previous, features, validate_protocol


def proposals(data, parent, schedule, state, deadline=None):
    if data.agv.count < 2:
        raise ValueError('仅支持冻结的多AGV输入')
    regions = {str(f): {g: features(data, schedule, g, f, old.seed_of('supplement-region', state, f))[0]
                       for g in ('A', 'L', 'K', 'R')} for f in (0.25, 0.5)}
    base = schedule.makespan, schedule.machine_energy
    seen = {parent}
    for n in range(6):
        for index in range(32):
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError('补样30分钟上限')
            seed = old.seed_of('supplement', state, n, index)
            row = dict(n=n+1, proposal=index, seed=seed, objective=None, changed=[], gain=0., duplicate=False)
            with old.Meter(2) as meter:
                try:
                    child = old.nb.apply_neighborhood(data, parent, n, random.Random(seed))
                    changed = old.touched(parent, child, data.instance.operation_counts)
                    row.update(changed=sorted(changed), chromosome=asdict(child), duplicate=child in seen)
                    seen.add(child)
                    row['passes'] = {f: {g: bool(changed) and changed <= region for g, region in groups.items()}
                                     for f, groups in regions.items()}
                    if changed:
                        result = meter.decode(data, child)
                        old.validate_schedule(data, child, result)
                        objective = result.makespan, result.machine_energy
                        row.update(objective=objective, gain=max(0., old.hv([(1, 1),
                            (objective[0]/base[0], objective[1]/base[1])])-old.hv([(1, 1)])))
                    row['decodes'] = meter.count
                except BaseException as error:
                    row.update(status='failed', error=repr(error), decodes=meter.count)
                    yield row
                    raise
            yield row
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError('补样30分钟上限，已保存末提案')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    validate_protocol(protocol)
    source = ROOT.parent/'KNb深挖/runs/budget-corrected'
    all_states = json.loads((source/'states.json').read_text(encoding='utf-8'))
    if any(s['instance'] not in protocol['instances'] for s in all_states):
        raise ValueError('禁止非探索实例')
    states = [s for s in all_states if s['seed'] == 101]
    if len(states) != 18 or len({s['state'] for s in states}) != 18:
        raise ValueError('补样父解清单不匹配')
    hashes = json.loads((source/'manifest.json').read_text(encoding='utf-8'))['source_hashes']
    if hashes != {name: old.sha(old.BASE/name) for name in hashes}:
        raise ValueError('A0来源变化')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', source_hashes=hashes, states_sha256=old.sha(source/'states.json'),
        protocol_sha256=old.sha(ROOT/'protocol.json'), scripts={f.name: old.sha(f) for f in Path(__file__).parent.glob('*.py')},
        inherited_scripts={str(f): old.sha(f) for f in (Path(previous.__file__), Path(old.__file__))},
        python=sys.version, proposals=0, acquisition_decodes=0, diagnostic_decodes=0, parents=18)
    old.dump(args.output/'manifest.json', manifest)
    start = time.perf_counter()
    try:
        with gzip.open(args.output/'proposals.jsonl.gz', 'wt', encoding='utf-8') as raw:
            for state in states:
                data = old.load_case(state['instance'])
                parent = old.Chromosome(**{k: tuple(v) for k, v in state['chromosome'].items()})
                manifest['acquisition_decodes'] += 1
                schedule = old.decode_static(data, parent)
                for row in proposals(data, parent, schedule, state['state'], start+1800):
                    raw.write(json.dumps(dict(state=state['state'], instance=state['instance'], **row))+'\n')
                    manifest['proposals'] += 1
                    manifest['diagnostic_decodes'] += row['decodes']
                raw.flush()
                print(state['state'], manifest['proposals'], flush=True)
        assert manifest['proposals'] == 3456
        assert manifest['diagnostic_decodes'] <= 4608
        assert hashes == {name: old.sha(old.BASE/name) for name in hashes}
        manifest.update(status='completed', source_unchanged=True)
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest['seconds'] = time.perf_counter()-start
        if (args.output/'proposals.jsonl.gz').exists():
            manifest['raw_sha256'] = old.sha(args.output/'proposals.jsonl.gz')
        old.dump(args.output/'manifest.json', manifest)


if __name__ == '__main__':
    main()
