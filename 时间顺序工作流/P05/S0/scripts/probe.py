"""静态S0隔离诊断；原A0文件只读，所有输出拒绝覆盖。"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

sys.dont_write_bytecode = True
BASE = Path(os.environ.get('P05_BASELINE', str(Path(__file__).resolve().parents[5] / '_codex_worktrees/Python-NEW-paper-static-baseline-v1')))
sys.path.insert(0,str(BASE))
from paper_static_baseline.dfjspt import neighborhoods as nb, qnsga2 as q
from paper_static_baseline.dfjspt.chromosome import Chromosome
from paper_static_baseline.dfjspt.data import load_experiment_input
from paper_static_baseline.dfjspt.decoder import decode_static, validate_schedule
from paper_static_baseline.dfjspt.initialization import hybrid_population

class BudgetExhausted(Exception): pass

class Meter:
    def __init__(self, limit=None):
        self.limit=limit; self.count=0
    def decode(self,*args,**kwargs):
        if self.limit is not None and self.count>=self.limit: raise BudgetExhausted()
        self.count+=1
        return decode_static(*args,**kwargs)
    def __enter__(self):
        self.old=(nb.decode_static,q.decode_static)
        nb.decode_static=q.decode_static=self.decode
        return self
    def __exit__(self,*args): nb.decode_static,q.decode_static=self.old

def load_case(name):
    root=BASE/'paper_static_baseline/data'
    return load_experiment_input(root/'brandimarte'/f'{name}.fjs',root/'resources/static_algorithm_comparison.json')

def seed_of(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big')

def flat_tokens(chromosome,counts):
    offsets=[]; total=0
    for c in counts: offsets.append(total); total+=c
    seen=[0]*len(counts); result=[]
    for job in chromosome.os:
        result.append(offsets[job]+seen[job]); seen[job]+=1
    return result

def touched(a,b,counts):
    changed={i for i,(x,y) in enumerate(zip(a.ms,b.ms)) if x!=y}
    changed.update(i for i,(x,y) in enumerate(zip(a.agv,b.agv)) if x!=y)
    for x,y in zip(flat_tokens(a,counts),flat_tokens(b,counts)):
        if x!=y: changed.update((x,y))
    return changed

def hv(points):
    area=0.; best_y=1.1
    for x,y in sorted(set(tuple(p) for p in points)):
        if x<1.1 and y<best_y:
            area+=(1.1-x)*(best_y-y); best_y=y
    return area

def scores(data,schedule):
    offsets=[]; total=0
    for c in data.instance.operation_counts: offsets.append(total); total+=c
    m=[0.]*total; a=[0.]*total
    ml=[sum(b.end-b.start for b in t if b.job) for t in schedule.machine_tables]
    al=[sum(b.end-b.start for b in t if b.job and b.charge==0 and b.load_status<0) for t in schedule.agv_tables]
    for j,table in enumerate(schedule.machine_tables):
        for b in table:
            if b.job: m[offsets[b.job-1]+b.opera-1]=(b.end-b.start)*ml[j]
    for j,table in enumerate(schedule.agv_tables):
        for b in table:
            if b.job and b.opera>0 and not b.charge and b.load_status<0:
                a[offsets[b.job-1]+b.opera-1]+=(b.end-b.start)*al[j]
    m=[x/(max(m) or 1.) for x in m]
    a=[x/(max(a) or 1.) for x in a]
    features=[max(ml)/(fmean(ml) or 1),sum(ml)/(len(ml)*schedule.makespan),max(al)/(fmean(al) or 1),sum(al)/(len(al)*schedule.makespan)]
    return m,a,features

def region(m,a,k,group,seed):
    c=[x+y for x,y in zip(m,a)]
    order=lambda values: sorted(range(len(values)),key=lambda i:(-values[i],i))
    if group=='S': return set(order(m)[:k])
    if group=='A': return set(order(a)[:k])
    if group=='C': return set(order(c)[:k])
    pool=list(range(len(m))) if group=='R0' else order(c)[:min(2*k,len(m))]
    return set(random.Random(seed).sample(pool,k))

def trial(data,parent,schedule,selected,n,b,seed):
    start=time.perf_counter(); rng=random.Random(seed)
    points=[(1.,1.)]; seen={parent}; candidates=[]
    counts=dict(attempts=0,evaluated=0,outside=0,noop=0,duplicate=0)
    base=(schedule.makespan,schedule.machine_energy)
    with Meter(b) as meter:
        while meter.count<b and counts['attempts']<40*b:
            # 内部一次解码的N4/N6须同时预留评价额度。
            if b-meter.count < (2 if n in (3,5) else 1): break
            counts['attempts']+=1
            child=nb.apply_neighborhood(data,parent,n,rng)
            changed=touched(parent,child,data.instance.operation_counts)
            if not changed: counts['noop']+=1; continue
            if not changed<=selected: counts['outside']+=1; continue
            if child in seen: counts['duplicate']+=1; continue
            seen.add(child)
            result=meter.decode(data,child)
            validate_schedule(data,child,result)
            assert child.empty_speed==parent.empty_speed and child.loaded_speed==parent.loaded_speed
            counts['evaluated']+=1
            obj=(result.makespan,result.machine_energy)
            points.append((obj[0]/base[0],obj[1]/base[1]))
            candidates.append({'chromosome':asdict(child),'objective':obj,'decode_index':meter.count})
    gain=max(0.,hv(points)-hv([(1.,1.)]))
    return {**counts,'decodes':meter.count,'gain':gain,'gain_per_decode':gain/meter.count if meter.count else None,'gain_per_budget':gain/b,'wall_seconds':time.perf_counter()-start,'candidates':candidates}

def training_states(states,heldout):
    return [s for s in states if s['instance']!=heldout]

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def capture(data,seed):
    snapshots=[]; calls=0; original=q.apply_neighborhood
    def observe(d,c,n,rng):
        nonlocal calls
        if calls%10==0: snapshots.append(c)
        calls+=1
        return original(d,c,n,rng)
    q.apply_neighborhood=observe
    try:
        with Meter() as meter: q.run_qnsga2(data,population_size=10,generations=3,seed=seed)
    finally: q.apply_neighborhood=original
    assert len(snapshots)==3
    return snapshots,meter.count

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    names=['Mk01'] if args.smoke else ['Mk01','Mk02','Mk04','Mk05']
    seeds=[11] if args.smoke else [11,22,33]
    sources=list((BASE/'paper_static_baseline/dfjspt').glob('*.py'))
    sources += [BASE/'paper_static_baseline/config/paper_static_v1.json',BASE/'paper_static_baseline/data/resources/static_algorithm_comparison.json']
    sources += [BASE/'paper_static_baseline/data/brandimarte'/f'{name}.fjs' for name in names]
    before={str(p.relative_to(BASE)):sha(p) for p in sources}
    manifest={'status':'running','python':sys.version,'platform':platform.platform(),'source_repo':str(BASE),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),'source_hashes':before,'protocol_hash':sha(Path(__file__).parents[1]/'实验协议.md'),'script_hash':sha(Path(__file__)),'instances':names,'seeds':seeds,'smoke':args.smoke}
    dump(args.output/'manifest.json',manifest)
    states=[]; rows=[]; acquisition=0; started=time.perf_counter()
    try:
        with gzip.open(args.output/'raw_trials.jsonl.gz','wt',encoding='utf-8') as raw:
            for name in names:
                data=load_case(name)
                for seed in seeds:
                    snapshots,cost=capture(data,seed); acquisition+=cost
                    for generation,parent in enumerate(snapshots):
                        state_id=f'{name}-{seed}-{generation}'
                        schedule=decode_static(data,parent); acquisition+=1
                        tick=time.perf_counter(); m,a,f=scores(data,schedule); feature_time=time.perf_counter()-tick
                        states.append({'state':state_id,'instance':name,'seed':seed,'generation':generation,'chromosome':asdict(parent),'features':[generation/2,*f],'objective':[schedule.makespan,schedule.machine_energy],'feature_seconds':feature_time})
                        for fraction in (.25,.5):
                            k=math.ceil(parent.operation_count*fraction)
                            for n in range(6):
                                for b in (4,8):
                                    for rep in range(4):
                                        stream=seed_of(state_id,fraction,n,rep)
                                        for group in ('R0','R1','S','A','C'):
                                            tick=time.perf_counter(); selected=region(m,a,k,group,seed_of('region',stream)); region_time=time.perf_counter()-tick
                                            result=trial(data,parent,schedule,selected,n,b,stream)
                                            identity={'state':state_id,'instance':name,'seed':seed,'generation':generation,'fraction':fraction,'k':k,'n':n+1,'b':b,'rep':rep,'group':group,'action':f'{fraction}-N{n+1}-b{b}','rng_seed':stream,'region_seconds':region_time}
                                            raw.write(json.dumps({**identity,'region':sorted(selected),**result})+'\n')
                                            rows.append({**identity,**{key:value for key,value in result.items() if key!='candidates'}})
                        print(state_id,'trials',len(rows),'seconds',round(time.perf_counter()-started,1),flush=True)
        csv_write(args.output/'trials.csv',rows); dump(args.output/'states.json',states)
        assert before=={str(p.relative_to(BASE)):sha(p) for p in sources}, 'A0源文件改变'
        manifest.update(status='completed',states=len(states),trials=len(rows),acquisition_decodes=acquisition,trial_decodes=sum(r['decodes'] for r in rows),evaluated_candidates=sum(r['evaluated'] for r in rows),source_unchanged=True,seconds=time.perf_counter()-started)
    except BaseException as error:
        manifest.update(status='failed',error=repr(error),seconds=time.perf_counter()-started)
        raise
    finally: dump(args.output/'manifest.json',manifest)

if __name__=='__main__': main()
