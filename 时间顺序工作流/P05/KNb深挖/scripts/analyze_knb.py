"""预定D1/D2/D3统计；选择只用探索集，实例等权。"""
import argparse,csv,json,random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
import probe_knb as k

def instance_mean(rows,metric):
    groups=defaultdict(list)
    for r in rows:groups[r['instance']].append(r[metric])
    return fmean(fmean(v) for v in groups.values())

def shuffle_labels(rows):
    out=[dict(r) for r in rows]
    indices=[i for i,r in enumerate(out) if r['split']=='explore']
    labels=[out[i]['group'] for i in indices];random.Random(719).shuffle(labels)
    for i,label in zip(indices,labels):out[i]['group']=label
    return out

def read(path):
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    for r in rows:
        for key in ('n','rep','generation','seed','decodes','evaluated','outside','noop','duplicate','attempts','batches','budget'):
            if key in r:r[key]=int(r[key])
        for key in ('fraction','gain','gain_per_budget','wall_seconds'):
            if key in r:r[key]=float(r[key])
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    out=args.run/'analysis';out.mkdir(exist_ok=False)
    rows=read(args.run/'D1.csv');walks=read(args.run/'D3.csv')
    fitted=k.fit(rows);shuffled=k.fit(shuffle_labels(rows))
    k.old.dump(out/'selection_audit.json',dict(original=fitted,shuffled=shuffled,training_instances=sorted({r['instance'] for r in rows if r['split']=='explore'})))
    d1=[]
    for name in sorted({r['instance'] for r in rows}):
        for group in k.GROUPS:
            sub=[r for r in rows if r['instance']==name and r['group']==group]
            d1.append(dict(instance=name,split=sub[0]['split'],group=group,gain_per_budget=fmean(r['gain_per_budget'] for r in sub),mean_gain=fmean(r['gain'] for r in sub),decodes=sum(r['decodes'] for r in sub),attempts=sum(r['attempts'] for r in sub),evaluated=sum(r['evaluated'] for r in sub),zero_evaluation=sum(r['evaluated']==0 for r in sub)))
    d2=[]
    for state in sorted({r['state'] for r in rows}):
        for group in k.GROUPS:
            for fraction in (.25,.5):
                sub=[r for r in rows if r['state']==state and r['group']==group and r['fraction']==fraction]
                choose={'conditional':fitted['conditional'][f'{group}|{fraction}'],'fixed':fitted['global_n'],'shuffled':shuffled['conditional'][f'{group}|{fraction}']}
                for policy,n in choose.items():
                    selected=[r for r in sub if r['n']==n]
                    d2.append(dict(state=state,instance=sub[0]['instance'],split=sub[0]['split'],generation=sub[0]['generation'],group=group,fraction=fraction,policy=policy,n=n,gain_per_budget=fmean(r['gain_per_budget'] for r in selected)))
    d3=[]
    for name in sorted({r['instance'] for r in walks}):
        for policy in ('fixed','round','stop','feedback'):
            sub=[r for r in walks if r['instance']==name and r['policy']==policy]
            d3.append(dict(instance=name,split=sub[0]['split'],policy=policy,gain=fmean(r['gain'] for r in sub),mean_decodes=fmean(r['decodes'] for r in sub),mean_wall_seconds=fmean(r['wall_seconds'] for r in sub),unspent_runs=sum(r['decodes']<r['budget'] for r in sub)))
    stages=[]
    for gen in (0,9,19):
        for policy in ('conditional','fixed','shuffled'):
            sub=[r for r in d2 if r['split']=='validation' and r['generation']==gen and r['policy']==policy]
            if sub:stages.append(dict(experiment='D2',generation=gen,policy=policy,value=instance_mean(sub,'gain_per_budget')))
        for group in k.GROUPS:
            sub=[r for r in rows if r['split']=='validation' and r['generation']==gen and r['group']==group]
            if sub:stages.append(dict(experiment='D1',generation=gen,policy=group,value=instance_mean(sub,'gain_per_budget')))
        for policy in ('fixed','round','stop','feedback'):
            sub=[r for r in walks if r['split']=='validation' and r['generation']==gen and r['policy']==policy]
            if sub:stages.append(dict(experiment='D3',generation=gen,policy=policy,value=instance_mean(sub,'gain')))
    summary={}
    for split in ('explore','validation'):
        if not any(r['split']==split for r in rows):continue
        summary[split]={
          'D1':{g:instance_mean([r for r in d1 if r['split']==split and r['group']==g],'gain_per_budget') for g in k.GROUPS},
          'D2':{p:instance_mean([r for r in d2 if r['split']==split and r['policy']==p],'gain_per_budget') for p in ('conditional','fixed','shuffled')},
          'D3':{p:instance_mean([r for r in d3 if r['split']==split and r['policy']==p],'gain') for p in ('fixed','round','stop','feedback')}}
    for name,table in [('D1_instances.csv',d1),('D2_states.csv',d2),('D3_instances.csv',d3),('validation_stages.csv',stages)]:
        if table:k.old.csv_write(out/name,table)
    k.old.dump(out/'summary.json',summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
