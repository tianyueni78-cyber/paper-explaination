"""只读历史实验；输出到标准输出，由调用者保存报告。无解码、无调参。"""
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


def funnel(rows):
    keys = ('attempts', 'outside', 'noop', 'duplicate', 'evaluated', 'decodes')
    result = {key: sum(int(row[key]) for row in rows) for key in keys}
    result['trials'] = len(rows)
    result['outside_rate'] = result['outside'] / result['attempts'] if result['attempts'] else None
    result['positive_trials'] = sum(float(row['gain']) > 0 for row in rows)
    result['mean_gain_per_allowed8'] = sum(float(row['gain']) / 8 for row in rows) / len(rows) if rows else None
    assert result['attempts'] == sum(result[k] for k in ('outside', 'noop', 'duplicate', 'evaluated'))
    return result


def tail(trace):
    last = max((i for i, batch in enumerate(trace) if batch['gain'] > 0), default=-1)
    return dict(tail_decodes=sum(batch['decodes'] for batch in trace[last + 1:]),
                tail_batches=len(trace) - last - 1, ever_improved=last >= 0)


def main():
    root = Path(__file__).resolve().parents[1]
    source = root.parent / 'KNb深挖/runs/budget-corrected'
    baseline = root.parents[3] / '_codex_worktrees/Python-NEW-paper-static-baseline-v1'
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    audit = json.loads((source / 'audit.json').read_text(encoding='utf-8'))
    source_checks = {path: sha(baseline / path) == value for path, value in manifest['source_hashes'].items()}
    artifact_checks = {path: sha(source / path) == value for path, value in audit['artifacts'].items()}
    assert all(source_checks.values()) and all(artifact_checks.values())
    with (source / 'D1.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        groups[f"N{row['n']}"].append(row)
        groups[f"{row['instance']}|{row['group']}|N{row['n']}|K{row['fraction']}"] .append(row)
    trajectories = []
    with gzip.open(source / 'D3_raw.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            trace = row.pop('trace')
            assert sum(batch['decodes'] for batch in trace) == row['decodes']
            assert abs(sum(batch['gain'] for batch in trace) - row['gain']) < 1e-10
            trajectories.append({**row, **tail(trace)})
    policies = {}
    for policy in sorted({row['policy'] for row in trajectories}):
        items = [row for row in trajectories if row['policy'] == policy]
        cost = sum(row['decodes'] for row in items)
        trailing = sum(row['tail_decodes'] for row in items)
        policies[policy] = dict(trajectories=len(items), decodes=cost, tail_decodes=trailing,
                               hindsight_tail_fraction=trailing / cost,
                               never_improved=sum(not row['ever_improved'] for row in items))
    result = dict(source=str(source), source_checks=source_checks, artifact_checks=artifact_checks,
                  historical_audit_status=audit['status'], fresh_decode_audit=False, new_decodes=0,
                  D1_records=len(rows), D3_trajectories=len(trajectories),
                  funnel={key: funnel(items) for key, items in sorted(groups.items())},
                  policies=policies, trajectories=trajectories)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
