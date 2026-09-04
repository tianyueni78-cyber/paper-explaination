"""仅修正受影响feedback；其余记录字节/数值复用并记录来源。"""
import argparse,gzip,json,shutil,time
from pathlib import Path
import probe_knb as k

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    original=json.loads((args.source/'manifest.json').read_text(encoding='utf-8'))
    assert original['status']=='completed'
    for name in ('D1_raw.jsonl.gz','D1.csv','states.json','selection.json'):
        shutil.copyfile(args.source/name,args.output/name)
    manifest=dict(original,status='running',feedback_version=2,source_run=str(args.source),source_manifest_hash=k.old.sha(args.source/'manifest.json'),D1_reused=True,script_hash=k.old.sha(Path(k.__file__)),correction_script_hash=k.old.sha(Path(__file__)),correction_note_hash=k.old.sha(Path(__file__).parents[1]/'协议偏差与修正.md'))
    k.old.dump(args.output/'manifest.json',manifest)
    states={s['state']:s for s in json.loads((args.output/'states.json').read_text(encoding='utf-8'))}
    fixed=json.loads((args.output/'selection.json').read_text(encoding='utf-8'))['fixed_action']
    rows=[];new_cost=0;reused_cost=0;count=0;cached={};started=time.perf_counter()
    try:
        with gzip.open(args.source/'D3_raw.jsonl.gz','rt',encoding='utf-8') as src,gzip.open(args.output/'D3_raw.jsonl.gz','wt',encoding='utf-8') as dst:
            for line in src:
                r=json.loads(line)
                if r['policy']=='feedback':
                    if time.perf_counter()-started>1800:raise TimeoutError('30分钟上限')
                    s=states[r['state']]
                    if r['state'] not in cached:
                        data=k.old.load_case(s['instance']);parent=k.old.Chromosome(**{key:tuple(v) for key,v in s['chromosome'].items()})
                        cached[r['state']]=(data,parent,k.old.decode_static(data,parent))
                    data,parent,schedule=cached[r['state']]
                    result=k.walk(data,parent,schedule,'feedback',fixed,96,k.old.seed_of(s['state'],r['rep']))
                    r={**{key:r[key] for key in ('state','instance','split','generation','seed','rep')},**result}
                    new_cost+=r['decodes'];count+=1
                    if count%9==0:print('corrected feedback',count,round(time.perf_counter()-started,1),flush=True)
                else:reused_cost+=r['decodes']
                dst.write(json.dumps(r)+'\n');rows.append({key:v for key,v in r.items() if key!='trace'})
        k.old.csv_write(args.output/'D3.csv',rows)
        manifest.update(status='completed',D3_decodes=sum(r['decodes'] for r in rows),new_feedback_decodes=new_cost,reused_D3_decodes=reused_cost,new_acquisition_decodes=len(cached),rerun_trajectories=count,seconds=time.perf_counter()-started)
    except BaseException as e:
        manifest.update(status='failed',error=repr(e));raise
    finally:k.old.dump(args.output/'manifest.json',manifest)

if __name__=='__main__':main()
