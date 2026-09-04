"""KNb机制探针，只读A0/S0，所有运行拒绝覆盖。"""
import argparse,gzip,json,math,platform,random,subprocess,sys,time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'S0/scripts'))
import probe as old
GROUPS=('R','S','A','C','W','L')
ACTIONS=[(f,n) for f in (.25,.5) for n in range(6)]

def wait_pair(p,d,a,s): return max(0.,d-p),max(0.,s-a)
def propagate(w,succ): return [v+sum(w[j] for j in set(succ[i])) for i,v in enumerate(w)]

def wait_features(data,schedule):
    offsets=[];total=0
    for c in data.instance.operation_counts: offsets.append(total);total+=c
    flat=lambda b:offsets[b.job-1]+b.opera-1
    machine={flat(b):b for t in schedule.machine_tables for b in t if b.job}
    loaded={flat(b):b for t in schedule.agv_tables for b in t if b.job and b.opera>0 and b.load_status==-2 and not b.charge}
    succ=[set() for _ in range(total)];w=[0.]*total;details=[]
    for j,count in enumerate(data.instance.operation_counts):
        for op in range(count):
            i=offsets[j]+op;p=machine[i-1].end if op else 0.
            d=loaded[i].start if i in loaded else p
            a=loaded[i].end if i in loaded else p
            tw,mw=wait_pair(p,d,a,machine[i].start);w[i]=tw+mw
            details.append(dict(operation=i,previous=p,departure=d,arrival=a,start=machine[i].start,transport_wait=tw,machine_wait=mw))
            if op<count-1:succ[i].add(i+1)
    for t in schedule.machine_tables:
        ops=[flat(b) for b in t if b.job]
        for i,j in zip(ops,ops[1:]):succ[i].add(j)
    for t in schedule.agv_tables:
        ops=[flat(b) for b in t if b.job and b.opera>0 and b.load_status==-2 and not b.charge]
        for i,j in zip(ops,ops[1:]):succ[i].add(j)
    return w,propagate(w,succ),details

def regions(data,schedule,k,seed):
    m,a,_=old.scores(data,schedule);w,l,_=wait_features(data,schedule)
    out={g:old.region(m,a,k,'R0' if g=='R' else g,seed) for g in ('R','S','A','C')}
    for g,v in [('W',w),('L',l)]:out[g]=set(sorted(range(len(v)),key=lambda i:(-v[i],i))[:k])
    return out

def capture(data,seed):
    count=0;snapshots=[];original=old.q.apply_neighborhood
    positions={g:random.Random(old.seed_of('capture',seed,g)).randrange(20) for g in (0,9,19)}
    def observe(d,c,n,rng):
        nonlocal count
        g,i=divmod(count,20)
        if positions.get(g)==i:snapshots.append((g,i,c))
        count+=1
        return original(d,c,n,rng)
    old.q.apply_neighborhood=observe
    try:
        with old.Meter() as meter:result=old.q.run_qnsga2(data,population_size=20,generations=20,seed=seed)
    finally:old.q.apply_neighborhood=original
    assert len(snapshots)==3 and count==400
    return snapshots,meter.count,result

def fit(rows):
    means=defaultdict(list);glob=defaultdict(list);fixed=defaultdict(list)
    for r in rows:
        if r['split']!='explore':continue
        g,f,n=r['group'],r['fraction'],r['n'];v=r['gain_per_budget']
        means[g,f,n].append(v);glob[n].append(v)
        if g=='A':fixed[f,n].append(v)
    cond={}
    for g,f in sorted({(g,f) for g,f,n in means}):
        ns=[n for gg,ff,n in means if (gg,ff)==(g,f)]
        cond[f'{g}|{f}']=min(ns,key=lambda n:(-fmean(means[g,f,n]),n))
    selected=min(fixed,key=lambda a:(-fmean(fixed[a]),a))
    return dict(conditional=cond,global_n=min(glob,key=lambda n:(-fmean(glob[n]),n)),fixed_action=ACTIONS.index((selected[0],selected[1]-1)))

def walk(data,parent,schedule,policy,fixed,budget,seed):
    if policy not in ('fixed','round','stop','feedback'):raise ValueError(policy)
    base=(schedule.makespan,schedule.machine_energy)
    points=[(1.,1.)];total=0;trace=[];seen={parent}
    visits=[0]*12;gains=[0.]*12;costs=[0]*12;channel=fixed;zeros=0
    started=time.perf_counter();feature_seconds=0.
    while total<budget and len(trace)<96:
        step=len(trace)
        if policy=='round':channel=(fixed+step)%12
        elif policy=='feedback':
            if step<12 or step%4==0:channel=(fixed+step)%12
            else:channel=min(range(12),key=lambda i:(-gains[i]/max(1,costs[i]),i))
        fraction,n=ACTIONS[channel];allowance=min(4,budget-total)
        if allowance<(2 if n in (3,5) else 1):break
        stream=old.seed_of(seed,channel,visits[channel]);visits[channel]+=1;rng=random.Random(stream)
        tick=time.perf_counter();m,a,_=old.scores(data,schedule)
        selected=old.region(m,a,math.ceil(parent.operation_count*fraction),'A',stream)
        feature_seconds+=time.perf_counter()-tick
        previous_parent=parent;initial_hv=old.hv(points);choice=None;choice_key=None
        counts=dict(attempts=0,evaluated=0,outside=0,noop=0,duplicate=0);candidates=[]
        with old.Meter(allowance) as meter:
            while meter.count<allowance and counts['attempts']<40*allowance:
                if allowance-meter.count<(2 if n in (3,5) else 1):break
                counts['attempts']+=1
                child=old.nb.apply_neighborhood(data,previous_parent,n,rng)
                changed=old.touched(previous_parent,child,data.instance.operation_counts)
                if not changed:counts['noop']+=1;continue
                if not changed<=selected:counts['outside']+=1;continue
                if child in seen:counts['duplicate']+=1;continue
                seen.add(child);result=meter.decode(data,child);old.validate_schedule(data,child,result)
                assert child.empty_speed==parent.empty_speed and child.loaded_speed==parent.loaded_speed
                obj=(result.makespan,result.machine_energy);pt=(obj[0]/base[0],obj[1]/base[1])
                delta=old.hv(points+[pt])-initial_hv
                counts['evaluated']+=1
                candidates.append(dict(chromosome=asdict(child),objective=obj,decode_index=meter.count))
                key=(-delta,obj,child.os,child.ms,child.agv)
                if delta>0 and (choice_key is None or key<choice_key):choice=(child,result);choice_key=key
        points.extend((c['objective'][0]/base[0],c['objective'][1]/base[1]) for c in candidates)
        delta=max(0.,old.hv(points)-initial_hv)
        if choice:parent,schedule=choice
        total+=meter.count;gains[channel]+=delta;costs[channel]+=meter.count
        trace.append(dict(step=step,action=channel,n=n+1,fraction=fraction,rng_seed=stream,region=sorted(selected),parent=asdict(previous_parent),accepted=asdict(parent) if choice else None,allowance=allowance,decodes=meter.count,total_decodes=total,gain=delta,archive_gain=max(0.,old.hv(points)-old.hv([(1.,1.)])),candidates=candidates,**counts))
        if policy=='stop':
            zeros=0 if delta>0 else zeros+1
            if zeros>=2:channel=(channel+1)%12;zeros=0
    return dict(policy=policy,decodes=total,budget=budget,gain=max(0.,old.hv(points)-old.hv([(1.,1.)])),batches=len(trace),evaluated=sum(t['evaluated'] for t in trace),feature_seconds=feature_seconds,wall_seconds=time.perf_counter()-started,trace=trace)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    names=['Mk01'] if args.smoke else ['Mk01','Mk02','Mk04','Mk05','Mk03','Mk06']
    seeds=[101] if args.smoke else [101,202,303]
    sources=list((old.BASE/'paper_static_baseline/dfjspt').glob('*.py'))+[old.BASE/'paper_static_baseline/data/resources/static_algorithm_comparison.json']+[old.BASE/'paper_static_baseline/data/brandimarte'/f'{n}.fjs' for n in names]
    before={str(f.relative_to(old.BASE)):old.sha(f) for f in sources}
    manifest=dict(status='running',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=old.BASE,text=True).strip(),source_hashes=before,script_hash=old.sha(Path(__file__)),protocol_hash=old.sha(Path(__file__).parents[1]/'KNb深挖实验协议.md'),s0_script_hash=old.sha(Path(old.__file__)),python=sys.version,platform=platform.platform(),instances=names,seeds=seeds,smoke=args.smoke)
    old.dump(args.output/'manifest.json',manifest)
    states=[];rows=[];trajectories=[];acquisition=0;started=time.perf_counter()
    try:
        with gzip.open(args.output/'D1_raw.jsonl.gz','wt',encoding='utf-8') as raw:
            for name in names:
                data=old.load_case(name)
                for seed in seeds:
                    captured,cost,_=capture(data,seed);acquisition+=cost
                    for g,index,parent in captured:
                        if time.perf_counter()-started>1800:raise TimeoutError('30分钟上限')
                        schedule=old.decode_static(data,parent);acquisition+=1
                        identity=dict(state=f'{name}-{seed}-{g}',instance=name,split='validation' if name in ('Mk03','Mk06') else 'explore',seed=seed,generation=g,population_index=index)
                        _,_,waits=wait_features(data,schedule)
                        states.append(dict(**identity,chromosome=asdict(parent),objective=[schedule.makespan,schedule.machine_energy],waits=waits))
                        for fraction in (.25,.5):
                            for n in range(6):
                                for rep in range(3):
                                    stream=old.seed_of(identity['state'],fraction,n,rep)
                                    tick=time.perf_counter();sets=regions(data,schedule,math.ceil(parent.operation_count*fraction),stream);feature_time=time.perf_counter()-tick
                                    for group in GROUPS:
                                        result=old.trial(data,parent,schedule,sets[group],n,8,stream)
                                        item=dict(**identity,fraction=fraction,n=n+1,rep=rep,group=group,rng_seed=stream,feature_seconds_all_groups=feature_time,**result)
                                        raw.write(json.dumps(dict(**item,region=sorted(sets[group])))+'\n')
                                        rows.append({k:v for k,v in item.items() if k!='candidates'})
                        print('D1',identity['state'],len(rows),round(time.perf_counter()-started,1),flush=True)
        old.dump(args.output/'states.json',states);old.csv_write(args.output/'D1.csv',rows)
        fitted=fit(rows);old.dump(args.output/'selection.json',fitted)
        with gzip.open(args.output/'D3_raw.jsonl.gz','wt',encoding='utf-8') as raw:
            for state in states:
                data=old.load_case(state['instance']);parent=old.Chromosome(**{k:tuple(v) for k,v in state['chromosome'].items()})
                schedule=old.decode_static(data,parent);acquisition+=1
                for rep in range(3):
                    for policy in ('fixed','round','stop','feedback'):
                        if time.perf_counter()-started>1800:raise TimeoutError('30分钟上限')
                        result=walk(data,parent,schedule,policy,fitted['fixed_action'],96,old.seed_of(state['state'],rep))
                        identity={k:state[k] for k in ('state','instance','split','generation','seed')}
                        item=dict(**identity,rep=rep,**result);raw.write(json.dumps(item)+'\n')
                        trajectories.append({k:v for k,v in item.items() if k!='trace'})
                print('D3',state['state'],len(trajectories),round(time.perf_counter()-started,1),flush=True)
        old.csv_write(args.output/'D3.csv',trajectories)
        assert before=={str(f.relative_to(old.BASE)):old.sha(f) for f in sources}
        manifest.update(status='completed',states=len(states),D1_trials=len(rows),D1_decodes=sum(r['decodes'] for r in rows),D3_trajectories=len(trajectories),D3_decodes=sum(r['decodes'] for r in trajectories),acquisition_decodes=acquisition,source_unchanged=True,seconds=time.perf_counter()-started)
    except BaseException as error:
        manifest.update(status='failed',error=repr(error),seconds=time.perf_counter()-started);raise
    finally:old.dump(args.output/'manifest.json',manifest)
if __name__=='__main__':main()
