"""Stratified pilot of the repaired importer before the full corpus build."""
import sys
sys.dont_write_bytecode=True
import argparse,collections,hashlib,heapq,json,time
from pathlib import Path
ROOT=Path(r'D:\git\GenUI-LM')
WORK=ROOT/'working_dir/dataset_v11_20260930'
REPO=Path(r'D:\git\anup-code\A2UI')
sys.path.insert(0,str(ROOT/'training/scripts'))
import prepare_space_v11_repaired as p

def select():
    heaps=collections.defaultdict(list)
    count=0
    for run,row in p.joined_rows(WORK/'projected',{}):
        count+=1
        key=f"{run}:{row.get('ui_id')}:{row['archive_record']['line']}"
        rank=-int(p.sha('repair-pilot-20260930:'+key),16)
        accept=row.get('training_acceptance') or {}
        buckets=['uniform','eligible:'+str(accept.get('eligible'))]+['reason:'+str(x) for x in accept.get('review_reasons',[])+accept.get('blocking_reasons',[])]
        item=(rank,key,run,row)
        for bucket in buckets:
            limit=256 if bucket=='uniform' else 5
            h=heaps[bucket]
            if len(h)<limit:heapq.heappush(h,item)
            elif rank>h[0][0]:heapq.heapreplace(h,item)
    selected={item[1]:item for heap in heaps.values() for item in heap}
    with (WORK/'repaired_pilot_inputs.jsonl').open('w',encoding='utf8') as f:
        for key,(_,_,run,row) in sorted(selected.items()):f.write(p.dumps({'run':run,'row':row})+'\n')
    p.dump(WORK/'repaired_pilot_selection.json',{'population':count,'selected':len(selected),'buckets':{k:len(v) for k,v in heaps.items()},'method':'lowest deterministic SHA-256 ranks per flag bucket plus256uniform'})
    print('Selected',len(selected),'of',count,flush=True)

def run():
    counts=collections.Counter();reasons=collections.Counter();repairs=collections.Counter();started=time.monotonic()
    scaffold=json.loads((WORK/'base_review/scaffold.json').read_text())
    p.init(REPO,{'scaffold':scaffold})
    items=((x['run'],x['row']) for _,_,x in p.lines(WORK/'repaired_pilot_inputs.jsonl'))
    with (WORK/'repaired_pilot_outcomes.jsonl').open('w',encoding='utf8') as events,(WORK/'repaired_pilot_accepted.jsonl').open('w',encoding='utf8') as accepted:
        for row,event,profile in p.bounded_map(p.new_item,items,REPO,8,{'scaffold':scaffold}):
            outcome='QUARANTINE' if row is None else ('REPAIR' if row['repair']['applied'] else 'KEEP')
            counts[outcome]+=1;reasons.update(event['reasons']);repairs.update((event.get('repair_changes') or {}).keys())
            events.write(p.dumps({'outcome':outcome,**event})+'\n')
            if row:accepted.write(p.dumps(row)+'\n')
            if sum(counts.values())%25==0:print(sum(counts.values()),dict(counts),'elapsed',round(time.monotonic()-started),flush=True)
    result={'counts':dict(counts),'reasons':dict(reasons),'repair_kinds_overlap':dict(repairs),'seconds':time.monotonic()-started}
    p.dump(WORK/'repaired_pilot_summary.json',result);print(p.dumps(result),flush=True)

if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('stage',choices=['select','run']);a=args.parse_args()
    (select if a.stage=='select' else run)()
