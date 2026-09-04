"""独立目标/计费/连续接受/区域与训练隔离审计。"""
import argparse,csv,gzip,json,math
from pathlib import Path
import probe_knb as k
import analyze_knb as analysis
p=k.old

def area(points):
    xs=sorted({x for x,y in points if x<1.1}|{1.1})
    return sum((r-l)*max(0,1.1-min([y for x,y in points if x<=l]+[1.1])) for l,r in zip(xs,xs[1:]))

def check_counts(row,limit):
    assert 0<=row['decodes']<=limit
    assert row['evaluated']==len(row['candidates'])
    assert row['decodes']==row['evaluated']+(row['attempts'] if row['n'] in (4,6) else 0)
    assert row['attempts']==sum(row[f] for f in ('evaluated','outside','noop','duplicate'))

def chrom(d):return p.Chromosome(**{key:tuple(v) for key,v in d.items()})
def same_trace(a,b):return json.loads(json.dumps(a))==json.loads(json.dumps(b))
def near(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
    target=args.run/'audit.json'
    if target.exists():raise FileExistsError(target)
    m=json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))
    assert m['status']=='completed'
    import probe_knb_v1 as legacy
    runner=k if m.get('feedback_version',1)==2 else legacy
    assert m['source_hashes']=={name:p.sha(p.BASE/name) for name in m['source_hashes']}
    states={s['state']:s for s in json.loads((args.run/'states.json').read_text(encoding='utf-8'))}
    inputs={name:p.load_case(name) for name in m['instances']}
    schedules={}
    decoded=0
    for sid,s in states.items():
        schedule=p.decode_static(inputs[s['instance']],chrom(s['chromosome']));decoded+=1
        schedules[sid]=schedule
        near(schedule.makespan,s['objective'][0]);near(schedule.machine_energy,s['objective'][1])
        assert k.wait_features(inputs[s['instance']],schedule)[2]==s['waits']
    def check_candidates(row,parent,data,base,seen):
        nonlocal decoded
        result=[]
        for c in row['candidates']:
            child=chrom(c['chromosome']);assert child not in seen;seen.add(child)
            assert p.touched(parent,child,data.instance.operation_counts)<=set(row['region'])
            assert child.empty_speed==parent.empty_speed and child.loaded_speed==parent.loaded_speed
            schedule=p.decode_static(data,child);decoded+=1;p.validate_schedule(data,child,schedule)
            near(schedule.makespan,c['objective'][0]);near(schedule.machine_energy,c['objective'][1])
            result.append((child,schedule,(schedule.makespan/base[0],schedule.machine_energy/base[1])))
        return result
    d1_count=d1_cost=0;keys=set()
    with gzip.open(args.run/'D1_raw.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);check_counts(row,8);d1_count+=1;d1_cost+=row['decodes']
            key=(row['state'],row['group'],row['fraction'],row['n'],row['rep']);assert key not in keys;keys.add(key)
            state=states[row['state']];data=inputs[row['instance']];parent=chrom(state['chromosome'])
            sets=k.regions(data,schedules[row['state']],math.ceil(parent.operation_count*row['fraction']),row['rng_seed'])
            assert set(row['region'])==sets[row['group']]
            children=check_candidates(row,parent,data,state['objective'],{parent})
            near(row['gain'],max(0,area([(1.,1.)]+[c[2] for c in children])-.01))
    assert (d1_count,d1_cost)==(m['D1_trials'],m['D1_decodes'])
    print('D1 audit',d1_count,decoded,flush=True)
    d3_count=d3_cost=changed=0;replay_cost=0;replayed=set()
    fixed=json.loads((args.run/'selection.json').read_text(encoding='utf-8'))['fixed_action']
    with gzip.open(args.run/'D3_raw.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);state=states[row['state']];data=inputs[row['instance']]
            parent=chrom(state['chromosome']);schedule=schedules[row['state']]
            points=[(1.,1.)];seen={parent};total=0
            for batch in row['trace']:
                check_counts(batch,batch['allowance']);assert chrom(batch['parent'])==parent
                mm,aa,_=p.scores(data,schedule)
                expected=p.region(mm,aa,math.ceil(parent.operation_count*batch['fraction']),'A',batch['rng_seed'])
                assert set(batch['region'])==expected
                children=check_candidates(batch,parent,data,state['objective'],seen)
                before=area(points);choices=[]
                for child,cs,pt in children:
                    delta=area(points+[pt])-before
                    if delta>1e-12:choices.append(((-delta,(cs.makespan,cs.machine_energy),child.os,child.ms,child.agv),child,cs))
                if batch['accepted']:
                    accepted=chrom(batch['accepted'])
                    assert any(child==accepted for child,cs,pt in children)
                    # 浮点微差可能改变极近增量次序；接受须确实正增量且近似最大。
                    actual=next(x for x in children if x[0]==accepted)
                    delta=area(points+[actual[2]])-before
                    assert delta>=-1e-12
                    if choices:near(delta,-min(choices,key=lambda x:x[0])[0][0])
                    parent,schedule=actual[0],actual[1];changed+=1
                else:assert not choices
                points.extend(c[2] for c in children);total+=batch['decodes']
                assert total==batch['total_decodes'];near(batch['gain'],max(0,area(points)-before));near(batch['archive_gain'],max(0,area(points)-.01))
            assert total==row['decodes'] and total<=96;near(row['gain'],max(0,area(points)-.01))
            d3_count+=1;d3_cost+=total
            key=(row['instance'],row['policy'])
            if key not in replayed:
                replayed.add(key)
                rr=runner.walk(data,chrom(state['chromosome']),schedules[row['state']],row['policy'],fixed,96,p.seed_of(state['state'],row['rep']))
                replay_cost+=rr['decodes'];near(rr['gain'],row['gain']);assert same_trace(rr['trace'],row['trace'])
    assert (d3_count,d3_cost)==(m['D3_trajectories'],m['D3_decodes'])
    d1=analysis.read(args.run/'D1.csv')
    trained=k.fit(d1)
    assert trained==json.loads((args.run/'selection.json').read_text(encoding='utf-8'))
    for r in d1:
        if r['split']=='validation':r['gain_per_budget']=1e99
    assert k.fit(d1)==trained
    assert not {'Mk07','Mk08'}&set(m['instances'])
    result=dict(status='PASS',D1_records=d1_count,D1_decodes=d1_cost,D3_trajectories=d3_count,D3_decodes=d3_cost,accepted_parent_changes=changed,audit_candidate_and_parent_decodes=decoded,trajectory_replays=len(replayed),trajectory_replay_decodes=replay_cost,source_unchanged=True,validation_label_poison_test='PASS',final_holdout_unread=True,artifacts={str(f.relative_to(args.run)):p.sha(f) for f in args.run.rglob('*') if f.is_file()})
    p.dump(target,result);print(json.dumps({key:v for key,v in result.items() if key!='artifacts'},indent=2))
if __name__=='__main__':main()
