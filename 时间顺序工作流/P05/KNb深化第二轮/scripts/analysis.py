"""探索分析：配对轨迹差值，实例与实例内种子分层，不把候选当独立样本。"""
import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from run_experiments import old, configurations


def paired_summary(pairs):
    cells = defaultdict(lambda: defaultdict(list))
    for instance, seed, delta in pairs:
        cells[instance][seed].append(delta)
    groups = {i: [fmean(v) for _, v in sorted(seeds.items())] for i, seeds in sorted(cells.items())}
    per_instance = {i: fmean(v) for i, v in groups.items()}
    values = list(groups.values())
    rng = random.Random(20260905)
    samples = []
    for _ in range(10000):
        picked = rng.choices(values, k=len(values))
        samples.append(fmean(fmean(rng.choices(v, k=len(v))) for v in picked))
    samples.sort()
    return dict(mean_difference=fmean(per_instance.values()), per_instance=per_instance,
        interval95=[samples[249], samples[9749]], n_instances=len(values), n_pairs=len(pairs))


def supplement_summary(rows):
    groups = defaultdict(lambda: dict(proposals=0, evaluated_pool=0, positive_pool=0, passed=0, positive_passed=0))
    for row in rows:
        for fraction, regions in row['passes'].items():
            for region, passed in regions.items():
                cell = groups[f'N{row["n"]}/{fraction}/{region}']
                cell['proposals'] += 1
                cell['evaluated_pool'] += row['objective'] is not None
                cell['positive_pool'] += row['gain'] > 1e-12
                cell['passed'] += passed
                cell['positive_passed'] += passed and row['gain'] > 1e-12
    for cell in groups.values():
        cell['positive_recall'] = cell['positive_passed']/cell['positive_pool'] if cell['positive_pool'] else None
        cell['pass_rate'] = cell['passed']/cell['evaluated_pool'] if cell['evaluated_pool'] else None
    return dict(groups)


def three_way(v):
    return v[1,1,1]-v[1,1,0]-v[1,0,1]-v[0,1,1]+v[1,0,0]+v[0,1,0]+v[0,0,1]-v[0,0,0]


def factorial_effects(v):
    return {
        'K': fmean(v[1,n,b]-v[0,n,b] for n in (0,1) for b in (0,1)),
        'N': fmean(v[k,1,b]-v[k,0,b] for k in (0,1) for b in (0,1)),
        'b': fmean(v[k,n,1]-v[k,n,0] for k in (0,1) for n in (0,1)),
        'KxN': fmean(v[1,1,b]-v[1,0,b]-v[0,1,b]+v[0,0,b] for b in (0,1)),
        'Kxb': fmean(v[1,n,1]-v[1,n,0]-v[0,n,1]+v[0,n,0] for n in (0,1)),
        'Nxb': fmean(v[k,1,1]-v[k,1,0]-v[k,0,1]+v[k,0,0] for k in (0,1)),
        'KxNxb': three_way(v)}


def load_rows(folder, stage):
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed':
        raise ValueError('失败或未完成实验不能进入完整组比较')
    filename = 'proposals.jsonl.gz' if stage == 'supplement' else 'trajectories.jsonl.gz'
    if old.sha(folder/filename) != manifest['raw_sha256']:
        raise ValueError('原始数据哈希不一致')
    rows = []
    with gzip.open(folder/filename, 'rt', encoding='utf-8') as raw:
        for line in raw:
            row = json.loads(line)
            if stage != 'supplement':
                trace = row.pop('trace')
                if not sum(b['decodes'] for b in trace) == row['decodes'] <= row['budget']:
                    raise ValueError('轨迹真实解码计费不一致')
                if abs(sum(b['gain'] for b in trace)-row['gain']) >= 1e-10:
                    raise ValueError('轨迹收益合计不一致')
                for b in trace:
                    if b['decodes'] != b['evaluated'] + (b['attempts'] if b['n'] in (4, 6) else 0):
                        raise ValueError('批次内部解码计费不一致')
                    if b['attempts'] != sum(b[k] for k in ('outside', 'noop', 'duplicate', 'evaluated')):
                        raise ValueError('批次提案去向合计不一致')
                row['attempts'] = sum(b['attempts'] for b in trace)
                row['outside'] = sum(b['outside'] for b in trace)
                row['deep_batches'] = sum(b['allowance'] > 4 for b in trace)
                row['observed_states'] = sorted({b['observed_state'] for b in trace})
            rows.append(row)
    if stage == 'supplement':
        if len(rows) != manifest['proposals'] or len(rows) != 3456:
            raise ValueError('补样行数不完整')
        if sum(r['decodes'] for r in rows) != manifest['diagnostic_decodes']:
            raise ValueError('补样真实解码计费不一致')
    else:
        if len(rows) != manifest['trajectories'] or len(rows) != 54*6*len(configurations(stage)):
            raise ValueError('完整实验轨迹数不匹配')
        if sum(r['decodes'] for r in rows) != manifest['search_decodes']:
            raise ValueError('完整实验真实解码计费不一致')
    return rows, manifest


def compare(rows, left, right):
    selected = {}
    for config in (left, right):
        data = [r for r in rows if tuple(r['config']) == config]
        keyed = {(r['state'], r['fraction'], r['rep']): r for r in data}
        if len(keyed) != len(data) or len(keyed) != 324:
            raise ValueError('配对样本缺失或重复')
        selected[config] = keyed
    if selected[left].keys() != selected[right].keys():
        raise ValueError('配对键不一致')
    pairs = [(r['instance'], r['initial_seed'], r['gain']-selected[right][key]['gain'])
             for key, r in selected[left].items()]
    result = paired_summary(pairs)
    result.update(left=left, right=right)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('K', 'N', 'b', 'interaction', 'supplement'), required=True)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    rows, manifest = load_rows(args.folder, args.stage)
    if args.stage == 'supplement':
        result = dict(regions=supplement_summary(rows))
    else:
        contrasts = []
        if args.stage == 'K':
            contrasts = [(('K', f'N{n}', '4'), (g, f'N{n}', '4')) for n in range(1, 7) for g in ('A', 'L', 'R')]
        elif args.stage == 'N':
            contrasts = [((g, 'state', '4'), (g, n, '4')) for g in ('A', 'K') for n in ('fixed', 'global', 'shuffled')]
        elif args.stage == 'b':
            contrasts = [(('A', 'global', 'recent'), ('A', 'global', b)) for b in ('4', '12', 'cumulative')]
        grouped = defaultdict(list)
        for row in rows:
            grouped['/'.join(row['config'])].append(row)
        result = dict(comparisons=[compare(rows, a, b) for a, b in contrasts],
            configurations={key: dict(n=len(values), gain=fmean(r['gain'] for r in values),
                decodes=fmean(r['decodes'] for r in values), attempts=sum(r['attempts'] for r in values),
                outside=sum(r['outside'] for r in values), deep_batches=sum(r['deep_batches'] for r in values),
                control_seconds=fmean(r['control_seconds'] for r in values),
                wall_seconds=fmean(r['wall_seconds'] for r in values),
                observed_states=sorted({s for r in values for s in r['observed_states']})) for key, values in grouped.items()})
        if args.stage == 'interaction':
            cells = {(g == 'K', n == 'state', b == 'recent'): {(r['state'], r['fraction'], r['rep']): r
                    for r in rows if tuple(r['config']) == (g, n, b)}
                    for g in ('A', 'K') for n in ('fixed', 'state') for b in ('4', 'recent')}
            if any(len(v) != 324 for v in cells.values()):
                raise ValueError('交互单元不完整')
            keys = next(iter(cells.values())).keys()
            if any(v.keys() != keys for v in cells.values()):
                raise ValueError('交互配对键不一致')
            effects = defaultdict(list)
            for key in keys:
                base = cells[0,0,0][key]
                values = {cell: items[key]['gain'] for cell, items in cells.items()}
                for name, value in factorial_effects(values).items():
                    effects[name].append((base['instance'], base['initial_seed'], value))
            result['factorial_effects'] = {name: paired_summary(values) for name, values in effects.items()}
    result.update(stage=args.stage, raw_sha256=manifest['raw_sha256'], exploratory=True, rows=len(rows),
        analysis_sha256=old.sha(Path(__file__)), numerical_positive_tolerance=1e-12)
    with (args.folder/'analysis.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
