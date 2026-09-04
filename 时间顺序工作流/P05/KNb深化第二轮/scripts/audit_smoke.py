"""重放已保存冒烟；独立按矩形条带重算HV，核验每个已评价候选。"""
import gzip
import json
import math
import sys
from pathlib import Path
from run_experiments import ROOT, old, walk


def area(points):
    xs = sorted({x for x, y in points if x < 1.1 and y < 1.1} | {1.1})
    return sum((right-left) * max(0, 1.1-min(y for x, y in points if x <= left))
               for left, right in zip(xs, xs[1:]))


def main():
    folder = Path(sys.argv[1])
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'completed'
    assert old.sha(folder/'trajectories.jsonl.gz') == manifest['raw_sha256']
    states = {s['state']: s for s in json.loads((ROOT.parent/'KNb深挖/runs/budget-corrected/states.json').read_text(encoding='utf-8'))}
    candidate_cost = acquisition = replay_cost = rows = 0
    with gzip.open(folder/'trajectories.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            data = old.load_case(row['instance'])
            parent = old.Chromosome(**{k: tuple(v) for k, v in states[row['state']]['chromosome'].items()})
            schedule = old.decode_static(data, parent)
            acquisition += 1
            base = schedule.makespan, schedule.machine_energy
            points = [(1, 1)]
            for batch in row['trace']:
                assert batch['decodes'] == batch['evaluated'] + (batch['attempts'] if batch['n'] in (4, 6) else 0)
                assert batch['attempts'] == sum(batch[k] for k in ('outside', 'noop', 'duplicate', 'evaluated'))
                before = area(points)
                batch_parent = old.Chromosome(**{k: tuple(v) for k, v in batch['parent'].items()})
                for candidate in batch['candidates']:
                    child = old.Chromosome(**{k: tuple(v) for k, v in candidate['chromosome'].items()})
                    assert old.touched(batch_parent, child, data.instance.operation_counts) <= set(batch['region'])
                    result = old.decode_static(data, child)
                    candidate_cost += 1
                    old.validate_schedule(data, child, result)
                    for observed, saved in zip((result.makespan, result.machine_energy), candidate['objective']):
                        assert math.isclose(observed, saved, rel_tol=1e-12, abs_tol=1e-10)
                    points.append((result.makespan/base[0], result.machine_energy/base[1]))
                assert math.isclose(area(points)-before, batch['gain'], abs_tol=1e-12)
            assert math.isclose(area(points)-area([(1, 1)]), row['gain'], abs_tol=1e-12)
            replay = walk(data, parent, schedule, tuple(row['config']), row['fraction'], row['seed'], row['budget'])
            assert json.loads(json.dumps(replay['trace'])) == row['trace']
            assert replay['decodes'] == row['decodes'] <= row['budget']
            replay_cost += replay['decodes']
            rows += 1
    assert rows == manifest['trajectories'] and replay_cost == manifest['search_decodes']
    print(json.dumps(dict(status='PASS', trajectories=rows, replay_decodes=replay_cost,
        candidate_verification_decodes=candidate_cost, audit_acquisition_decodes=acquisition,
        audit_total_decodes=replay_cost+candidate_cost+acquisition, performance_claims=False), indent=2))


if __name__ == '__main__':
    main()
