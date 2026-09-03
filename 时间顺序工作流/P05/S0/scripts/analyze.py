"""分析固定E0/E1/E2，不调参、不增加试验。"""
import argparse
import csv
import itertools
import json
import math
import time
from collections import defaultdict,Counter
from pathlib import Path
from statistics import fmean,pstdev

def best(values): return min(values,key=lambda a:(-values[a],a))
def action_means(rows,repeats):
    values=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row['rep'] in repeats: values[row['state']][row['action']].append(row['utility'])
    return {s:{a:fmean(v) for a,v in acts.items()} for s,acts in values.items()}

def predict(train,test,means,dims,k=3):
    scales=[pstdev([s['features'][j] for s in train]) or 1 for j in range(dims)]
    nearest=sorted(train,key=lambda s:(sum(((s['features'][j]-test['features'][j])/scales[j])**2 for j in range(dims)),s['state']))[:k]
    return best({a:fmean(means[s['state']][a] for s in nearest) for a in means[nearest[0]['state']]})

def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    out=args.run/'analysis';out.mkdir(exist_ok=False)
    rows=list(csv.DictReader((args.run/'trials.csv').open(encoding='utf-8-sig')))
    for r in rows:
        for k in ('rep','n','b','decodes','attempts','evaluated','outside','noop','duplicate'): r[k]=int(r[k])
        for k in ('gain','gain_per_budget','wall_seconds','region_seconds'): r[k]=float(r[k])
        # 决策效用：零解码/零收益动作为零，原始比率null不覆盖。
        r['utility']=float(r['gain_per_decode']) if r['gain_per_decode'] else 0.
    states=json.loads((args.run/'states.json').read_text(encoding='utf-8'))
    names=sorted({s['instance'] for s in states})
    e0=[]; stratified=[]
    for name,group in itertools.product(names,('R0','R1','S','A','C')):
        rs=[r for r in rows if r['instance']==name and r['group']==group]
        e0.append({'instance':name,'group':group,'utility':fmean(r['utility'] for r in rs),'gain_per_budget':fmean(r['gain_per_budget'] for r in rs),'mean_gain':fmean(r['gain'] for r in rs),'decodes':sum(r['decodes'] for r in rs),'evaluated':sum(r['evaluated'] for r in rs),'zero_decode_trials':sum(r['decodes']==0 for r in rs),'zero_gain_trials':sum(r['gain']==0 for r in rs),'wall_seconds':sum(r['wall_seconds']+r['region_seconds'] for r in rs)})
    for group,n,b in itertools.product(('R0','R1','S','A','C'),range(1,7),(4,8)):
        rs=[r for r in rows if r['group']==group and r['n']==n and r['b']==b]
        stratified.append({'group':group,'n':n,'b':b,'utility':fmean(r['utility'] for r in rs),'gain_per_budget':fmean(r['gain_per_budget'] for r in rs),'mean_gain':fmean(r['gain'] for r in rs),'attempts':sum(r['attempts'] for r in rs),'outside':sum(r['outside'] for r in rs),'noop':sum(r['noop'] for r in rs),'duplicate':sum(r['duplicate'] for r in rs),'evaluated':sum(r['evaluated'] for r in rs),'zero_evaluation_trials':sum(r['evaluated']==0 for r in rs)})
    write(out/'E0_instances.csv',e0);write(out/'E0_neighborhoods.csv',stratified)
    pairs=[]
    for control in ('R0','S','R1','A'):
        differences=[next(r['utility'] for r in e0 if r['instance']==name and r['group']=='C')-next(r['utility'] for r in e0 if r['instance']==name and r['group']==control) for name in names]
        boot=sorted(fmean(sample) for sample in itertools.product(differences,repeat=len(names)))
        pairs.append({'comparison':f'C-{control}','instance_differences':differences,'mean_difference':fmean(differences),'descriptive_cluster_interval':[boot[int(.025*(len(boot)-1))],boot[int(.975*(len(boot)-1))]]})
    e1=[];e2=[];folds=[]
    for group in ('S','C'):
        rs=[r for r in rows if r['group']==group]
        train=action_means(rs,{0,1}); test=action_means(rs,{2,3})
        actions=sorted(next(iter(train.values())))
        fixed=best({a:fmean(v[a] for v in train.values()) for a in actions})
        for s in states:
            chosen=best(train[s['state']]); v=test[s['state']]
            e1.append({'group':group,'instance':s['instance'],'state':s['state'],'selected_action':chosen,'fixed_action':fixed,'selected_test_utility':v[chosen],'fixed_test_utility':v[fixed],'random_test_utility':fmean(v.values()),'hindsight_oracle_utility':max(v.values())})
        for heldout in names:
            ts=[s for s in states if s['instance']!=heldout]
            assert ts and all(s['instance']!=heldout for s in ts)
            fixed=best({a:fmean(train[s['state']][a] for s in ts) for a in actions})
            folds.append({'group':group,'heldout':heldout,'train_states':[s['state'] for s in ts],'test_states':[s['state'] for s in states if s['instance']==heldout],'train_repeats':[0,1],'test_repeats':[2,3],'fixed_action':fixed})
            for s in states:
                if s['instance']!=heldout: continue
                v=test[s['state']]
                for label,dims in (('simple',3),('cross',5)):
                    tick=time.perf_counter(); chosen=predict(ts,s,train,dims); elapsed=time.perf_counter()-tick
                    e2.append({'group':group,'features':label,'instance':heldout,'state':s['state'],'selected_action':chosen,'fixed_action':fixed,'selected_utility':v[chosen],'fixed_utility':v[fixed],'random_utility':fmean(v.values()),'hindsight_oracle_utility':max(v.values()),'prediction_seconds':elapsed})
    write(out/'E1_states.csv',e1);write(out/'E2_states.csv',e2)
    summary={'E0':pairs,'E1':[],'E2':[]}
    for group in ('S','C'):
        rs=[r for r in e1 if r['group']==group]
        summary['E1'].append({'group':group,'selected':fmean(r['selected_test_utility'] for r in rs),'fixed':fmean(r['fixed_test_utility'] for r in rs),'random':fmean(r['random_test_utility'] for r in rs),'selected_actions':dict(Counter(r['selected_action'] for r in rs))})
        for label in ('simple','cross'):
            rs=[r for r in e2 if r['group']==group and r['features']==label]
            summary['E2'].append({'group':group,'features':label,'selected':fmean(r['selected_utility'] for r in rs),'fixed':fmean(r['fixed_utility'] for r in rs),'random':fmean(r['random_utility'] for r in rs),'per_instance_difference':{name:fmean(r['selected_utility']-r['fixed_utility'] for r in rs if r['instance']==name) for name in names}})
    (out/'folds.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
