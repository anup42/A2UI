#!/usr/bin/env python3
"""Recover source-proven v10 audit holds into a separate screened base copy."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
from prepare_space_v11_repaired import init,lines,file_sha,sha,dump,dumps,describe,gates


def build(args):
    init(args.policy_repo)
    from space_v11_known_hold_repairs import repair_known_hold
    from ir_training.data.express_preparation import serialize_checked
    prior=json.loads((args.prior_work_dir/'base_report.json').read_text())
    original=Path(prior['base_dir'])
    for name,digest in prior['input_split_sha256'].items():
        if file_sha(original/name)!=digest:raise ValueError('Original v10 changed')
    holds=json.loads((args.prior_work_dir/'known_v10_holds.json').read_text())
    candidates=defaultdict(list)
    for split in ('train','val'):
        for number,raw,row in lines(original/f'{split}.jsonl'):
            if row['source_id'] not in holds:continue
            finding=holds[row['source_id']]
            fixed,proofs,unresolved=repair_known_hold(row,finding['case'])
            if not unresolved:
                checked=serialize_checked(fixed['completion'],'root-first')
                unresolved=gates(fixed['response_text'],fixed['completion'],checked,(fixed['metadata'].get('archive_recovery') or {}).get('original_source_sha256',''),(fixed['metadata'].get('url_preprocessing') or {}).get('url_map',{}))
            candidates[row['source_id']].append({'split':split,'line':number,'original':row,'fixed':fixed,'proofs':proofs,'unresolved':unresolved,'original_row_sha256':__import__('hashlib').sha256(raw).hexdigest()})
    recovered={family:members for family,members in candidates.items() if all(not x['unresolved'] and x['proofs'] for x in members)}
    repairs={member['original']['id']:member for members in recovered.values() for member in members}
    unresolved_holds={family:value for family,value in holds.items() if family not in recovered}
    if args.work_dir.exists():raise FileExistsError(args.work_dir)
    args.work_dir.mkdir(parents=True)
    for name in ('scaffold.json','base_index.jsonl'):
        shutil.copyfile(args.prior_work_dir/name,args.work_dir/name)
    dump(args.work_dir/'known_v10_holds.json',unresolved_holds)
    dump(args.work_dir/'known_v10_holds_original.json',holds)
    counts=Counter();repair_counts=Counter();cases=Counter();audit=[]
    with (args.work_dir/'decisions.jsonl').open('w',encoding='utf8') as decisions,(args.work_dir/'base_index.jsonl').open('a',encoding='utf8') as index:
        for split in ('train','val'):
            with (args.work_dir/f'{split}.jsonl').open('wb') as out:
                for number,raw,row in lines(original/f'{split}.jsonl'):
                    event={'id':row['id'],'split':split,'line':number,'family':row['source_id'],'outcome':'KEEP','reasons':[]}
                    if row['source_id'] in unresolved_holds:
                        event.update(outcome='QUARANTINE',reasons=['known_v10_manual_review_family'])
                    elif row['id'] in repairs:
                        repair=repairs[row['id']];fixed=repair['fixed']
                        encoded=(dumps(fixed)+'\n').encode('utf8')
                        out.write(encoded);counts[split]+=1;repair_counts[split]+=1;cases[str(holds[row['source_id']]['case'])]+=1
                        event.update(outcome='REPAIR',proofs=repair['proofs'])
                        checked=serialize_checked(fixed['completion'],'root-first')
                        profile=describe(fixed,checked,'v10')
                        entry={key:profile[key] for key in ('signature','normalized_sha','source_sha','family','semantic_sha','pair_sha')}
                        index.write(dumps({**entry,'id':row['id']+':v11-repaired','split':split,'line':number,'excluded':False})+'\n')
                        audit.append({'id':row['id'],'split':split,'line':number,'case':holds[row['source_id']]['case'],'original_row_sha256':repair['original_row_sha256'],'output_row_sha256':__import__('hashlib').sha256(encoded).hexdigest(),'proofs':repair['proofs']})
                    else:
                        out.write(raw);counts[split]+=1
                    decisions.write(dumps(event)+'\n')
    for name,digest in prior['input_split_sha256'].items():
        if file_sha(original/name)!=digest:raise ValueError('Original v10 changed during repair')
    result=deepcopy(prior)
    result.update(policy='space-v11-reviewed-base-repairs-20260930',retained_rows=dict(counts),quarantined_rows={s:prior['input_rows'][s]-counts[s] for s in counts},reasons_overlap={'known_v10_manual_review_family':sum(prior['input_rows'].values())-sum(counts.values())},known_manual_hold_families=len(unresolved_holds))
    result['base_repairs']={'original_held_rows':sum(prior['quarantined_rows'].values()),'recovered_rows':sum(repair_counts.values()),'recovered_by_split':dict(repair_counts),'recovered_cases':dict(cases),'remaining_held_rows':sum(result['quarantined_rows'].values()),'remaining_held_families':len(unresolved_holds),'source_factual_scope':'Only audited target fidelity and exact internally proven arithmetic; other source facts remain unverified.'}
    result['base_repair_script_sha256']=file_sha(Path(__file__))
    result['base_repair_implementation_sha256']=file_sha(Path(__file__).parent/'space_v11_known_hold_repairs.py')
    result['retained_split_sha256']={s+'.jsonl':file_sha(args.work_dir/f'{s}.jsonl') for s in counts}
    dump(args.work_dir/'known_v10_repairs.json',audit)
    dump(args.work_dir/'base_report.json',result)
    print(dumps({'retained':dict(counts),'repairs':result['base_repairs']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('policy-repo','prior-work-dir','work-dir'):parser.add_argument('--'+name,type=Path,required=True)
    build(parser.parse_args())
