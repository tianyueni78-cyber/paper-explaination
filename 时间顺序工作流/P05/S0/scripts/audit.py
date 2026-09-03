"""独立回放原始候选，检查区域、目标、预算守恒和实例隔离。"""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import probe as p

def check_counts(row):
    assert 0<=row['decodes']<=row['b']
    assert row['evaluated']==len(row['candidates'])
    assert row['decodes']==row['evaluated']+(row['attempts'] if row['n'] in (4,6) else 0)

def chromosome(d):return p.Chromosome(**{k:tuple(v) for k,v in d.items()})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
    target=args.run/'audit.json'
    if target.exists():raise FileExistsError(target)
    manifest=json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='completed'
    assert manifest['source_hashes']=={name:p.sha(p.BASE/name) for name in manifest['source_hashes']}
    states={s['state']:s for s in json.loads((args.run/'states.json').read_text(encoding='utf-8'))}
    inputs={name:p.load_case(name) for name in manifest['instances']}
    count=0;decoded=0;claimed=0;seen_keys=set();first_replay=set();replayed=0
    # 独立二维矩形并集：按x分段积分，而非复用实验的HV函数。
    def area(points):
        xs=sorted({x for x,y in points if x<1.1}|{1.1})
        return sum((r-l)*max(0,1.1-min([y for x,y in points if x<=l]+[1.1])) for l,r in zip(xs,xs[1:]))
    with gzip.open(args.run/'raw_trials.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);check_counts(row);count+=1;claimed+=row['decodes']
            key=(row['state'],row['group'],row['action'],row['rep'])
            assert key not in seen_keys;seen_keys.add(key)
            state=states[row['state']];data=inputs[row['instance']];parent=chromosome(state['chromosome'])
            points=[(1.,1.)];unique=set()
            assert len(row['region'])==row['k']
            assert row['attempts']==row['evaluated']+row['outside']+row['noop']+row['duplicate']
            for candidate in row['candidates']:
                child=chromosome(candidate['chromosome'])
                assert child not in unique;unique.add(child)
                assert p.touched(parent,child,data.instance.operation_counts)<=set(row['region'])
                assert child.empty_speed==parent.empty_speed and child.loaded_speed==parent.loaded_speed
                result=p.decode_static(data,child);decoded+=1
                p.validate_schedule(data,child,result)
                assert math.isclose(result.makespan,candidate['objective'][0],abs_tol=1e-9)
                assert math.isclose(result.machine_energy,candidate['objective'][1],abs_tol=1e-9)
                points.append((result.makespan/state['objective'][0],result.machine_energy/state['objective'][1]))
            assert math.isclose(row['gain'],max(0,area(points)-.01),abs_tol=1e-12)
            # 每组每邻域每实例抽一个完整试验重演，以检查随机流和计数复现。
            replay_key=(row['instance'],row['group'],row['n'])
            if replay_key not in first_replay:
                first_replay.add(replay_key)
                schedule=p.decode_static(data,parent);decoded+=1
                rr=p.trial(data,parent,schedule,set(row['region']),row['n']-1,row['b'],row['rng_seed'])
                replayed+=rr['decodes']
                for field in ('attempts','evaluated','outside','noop','duplicate','decodes','gain'):
                    assert rr[field]==row[field],(replay_key,field)
    assert count==manifest['trials'] and claimed==manifest['trial_decodes']
    summary=list(csv.DictReader((args.run/'trials.csv').open(encoding='utf-8-sig')))
    assert len(summary)==count
    assert sum(int(r['decodes']) for r in summary)==claimed
    folds=json.loads((args.run/'analysis/folds.json').read_text(encoding='utf-8'))
    for fold in folds:
        assert set(fold['train_states']).isdisjoint(fold['test_states'])
        assert all(states[s]['instance']!=fold['heldout'] for s in fold['train_states'])
        assert all(states[s]['instance']==fold['heldout'] for s in fold['test_states'])
    result={'status':'PASS','trial_records':count,'claimed_trial_decodes':claimed,'replayed_candidate_and_parent_decodes':decoded,'replayed_trial_decodes':replayed,'trial_replays':len(first_replay),'folds':len(folds),'source_unchanged':True,'artifacts':{str(f.relative_to(args.run)):p.sha(f) for f in args.run.rglob('*') if f.is_file()}}
    p.dump(target,result);print(json.dumps({k:v for k,v in result.items() if k!='artifacts'},indent=2))

if __name__=='__main__':main()
