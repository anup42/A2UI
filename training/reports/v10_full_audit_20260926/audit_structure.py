"""Read-only complete v10 structure/length audit. Outputs only beside this script.

Uses the repository's existing schema parity harness, exact deployed SentencePiece
vocabulary, production parser, reachability check and semantic round-trip emitter.
Token counts describe this local tokenizer, not an unobserved remote training run.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'training/outputs/datasets/full_data_archive_recovered_v10'
TOKENIZER = ROOT / 'tmp/e2b_mobile_20260921/tokenizer.model'
sys.path[:0] = [str(ROOT/'training/src'), str(ROOT/'dataset/src'), str(ROOT/'training/scripts/audits')]
SP = None
SHARED = None

def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf8')

def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()

def init():
    global SP, SHARED
    import sentencepiece as spm
    from full_data_ir_20260913 import wire_setup
    wire_setup(False)
    SP = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    SHARED = json.loads((ROOT/'GenUICraft/genuicraft/src/main/assets/genuicraft/prompts/e2b_v10_shared_prompt.json').read_text())['scaffold']

def prompt_text(messages):
    # Match the deployed non-thinking template. BOS is counted separately.
    return ''.join('<|turn>'+('model' if m['role']=='assistant' else m['role'])+'\n'+m['content']+'<turn|>\n' for m in messages)+'<|turn>model\n'

def strings(v):
    if isinstance(v, str): yield v
    elif isinstance(v, dict):
        for child in v.values(): yield from strings(child)
    elif isinstance(v, list):
        for child in v: yield from strings(child)

def one(item):
    split, line, raw = item
    r = json.loads(raw)
    src, target, msgs = r.get('response_text',''), r.get('completion',''), r.get('messages',[])
    result = {'split':split,'line':line,'id':r.get('id'), 'source_chars':len(src),'target_chars':len(target)}
    flags=[]
    if [m.get('role') for m in msgs] != ['system','user','assistant','user','assistant']:flags.append('unexpected_roles')
    if not msgs or msgs[-1].get('content') != target:flags.append('completion_message_mismatch')
    if len(msgs)<2 or msgs[-2].get('content') != SHARED['task_prefix']+src:flags.append('source_message_mismatch')
    if not src.strip():flags.append('empty_source')
    if not target.strip():flags.append('empty_target')
    if not target.strip().startswith('<a2ui>') or not target.strip().endswith('</a2ui>'):flags.append('envelope')
    if '\ufffd' in src+target:flags.append('replacement_character')
    if re.search(r'<\|(?:turn|im_start|im_end)|<turn\|>|<bos>|<eos>',src+target):flags.append('special_token_in_content')
    sharedmsgs=SHARED['messages']+[{'role':'user','content':SHARED['task_prefix']+src}]
    old_prompt=prompt_text(msgs[:-1]);new_prompt=prompt_text(sharedmsgs)
    nt=len(SP.encode(target,out_type=int))
    result.update(source_tokens=len(SP.encode(src,out_type=int)), target_tokens=nt,
                  archived_prompt_tokens=1+len(SP.encode(old_prompt,out_type=int)),
                  deployed_prompt_tokens=1+len(SP.encode(new_prompt,out_type=int)),
                  deployed_sequence_tokens=1+len(SP.encode(new_prompt+target+'<turn|>\n',out_type=int)),
                  archived_sequence_tokens=1+len(SP.encode(old_prompt+target+'<turn|>\n',out_type=int)),
                  system_sha=sha(msgs[0]['content']) if msgs else '',
                  scaffold_sha=sha(json.dumps(msgs[:3],sort_keys=True,ensure_ascii=False)),
                  source_has_markdown_heading=bool(re.search(r'^#{1,6}\s',src,re.M)),
                  source_has_pipe_table=bool(re.search(r'^\|.*\|',src,re.M)),
                  source_has_media=bool(re.search(r'(?im)^\s*(?:Media:|Images?:|Icons?:)',src)),
                  source_has_action=bool(re.search(r'(?im)Action:|Quick Actions',src)),
                  source_has_url_placeholder=bool(re.search(r'\[(?:ACTION|SOURCE|IMAGE|ICON|VIDEO|URL)[A-Z_]*_\d+\]',src)),
                  source_has_numeric_citation=bool(re.search(r'\[\d+\]',src)),
                  source_has_live_url=bool(re.search(r'https?://',src)),
                  source_non_ascii=bool(re.search(r'[^\x00-\x7f]',src)),
                  target_has_source_placeholder='@source' in target)
    try:
        from ir_training.data.express_preparation import serialize_checked
        from pipeline.renderer_semantics import iter_renderer_references
        s=serialize_checked(target,'root-first');g=s.graph;els=g['elements']
        result['valid']=True
        result['semantic_hash']=s.semantic_sha256
        result['canonical_target_tokens']=len(SP.encode(s.text,out_type=int))
        result['component_counts']=dict(Counter(e['type'] for e in els.values()))
        result['component_count']=len(els)
        depths={}
        def depth(k):
            if k not in depths:depths[k]=1+max([depth(e.target_id) for e in iter_renderer_references(els[k])] or [0])
            return depths[k]
        result['depth']=depth(g['root'])
        result['table_domains']=dict(Counter(e['props'].get('domain','<unset>') for e in els.values() if e['type']=='Table'))
        result['table_presentations']=dict(Counter(e['props'].get('preferredPresentation','<unset>') for e in els.values() if e['type']=='Table'))
        result['state_chars']=len(json.dumps(g['state'],ensure_ascii=False))
        result['longest_text_chars']=max([len(e['props'].get('text','')) for e in els.values() if e['type']=='Text' and isinstance(e['props'].get('text'),str)] or [0])
        visible=list(strings(g['state']))+[v for e in els.values() for v in strings(e['props'])]
        repeated=[(t,n) for t,n in Counter(visible).items() if n>=5 and len(t)>=40]
        result['repeated_long_literals']=repeated[:5]
        result['alpha_id_components']=sum(bool(re.fullmatch('[a-z]{1,3}',k)) for k in els if k!='root')
    except Exception as e:
        result.update(valid=False,error=f'{type(e).__name__}:{str(e)[:700]}')
    result['flags']=flags
    return result

def chunks(limit=None, skip=None):
    for split in ['train','val']:
        with (DATA/(split+'.jsonl')).open('rb') as f:
            batch=[]
            for i,line in enumerate(f,1):
                if limit and i>limit:break
                if skip and i<=skip.get(split,0):continue
                batch.append((split,i,line))
                if len(batch)==64:yield batch;batch=[]
            if batch:yield batch

def batch(items):return [one(i) for i in items]

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true');args=p.parse_args()
    from full_data_ir_20260913 import wire_setup
    t=time.time();parity=wire_setup(True);dump('schema_parity.json',parity)
    counters=defaultdict(Counter);values=defaultdict(lambda:defaultdict(list));examples=defaultdict(list);hashes={}
    for split in ['train','val']:
        h=hashlib.sha256()
        with (DATA/(split+'.jsonl')).open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        hashes[split]=h.hexdigest()
    total=0;skip={};old=[]
    if args.resume and (OUT/'structure_rows.jsonl').exists():
        path=OUT/'structure_rows.jsonl';last_good=0
        with path.open('rb') as f:
            for raw in f:
                try:r=json.loads(raw)
                except json.JSONDecodeError:break
                old.append(r);skip[r['split']]=r['line'];last_good=f.tell()
        with path.open('r+b') as f:f.truncate(last_good)
    numkeys=['source_chars','target_chars','source_tokens','target_tokens','canonical_target_tokens','archived_prompt_tokens','deployed_prompt_tokens','archived_sequence_tokens','deployed_sequence_tokens','component_count','depth','longest_text_chars','alpha_id_components']
    boolkeys=['source_has_markdown_heading','source_has_pipe_table','source_has_media','source_has_action','source_has_url_placeholder','source_has_numeric_citation','source_has_live_url','source_non_ascii','target_has_source_placeholder']
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init) as pool, (OUT/'structure_rows.jsonl').open('a' if args.resume else 'w',encoding='utf8') as out:
        # Bound submitted batches to keep corpus bytes off the executor queue.
        it=iter(chunks(args.limit,skip));pending=[]
        for _ in range(args.workers*2):
            b=next(it,None)
            if b:pending.append(pool.submit(batch,b))
        replay=bool(old)
        while pending or replay:
            results=old if replay else pending.pop(0).result()
            for r in results:
                if not replay:out.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
                total+=1
                cohorts=[r['split'],'all']
                for cohort in cohorts:
                    c=counters[cohort];c['rows']+=1;c['valid' if r['valid'] else 'invalid']+=1
                    for k in boolkeys:c[k]+=int(r[k])
                    for f in r['flags']:c['flag:'+f]+=1
                    for k in ['system_sha','scaffold_sha']:counters[cohort+':'+k][r[k]]+=1
                    for k in numkeys:
                        if k in r:values[cohort][k].append(r[k])
                    for k in ['component_counts','table_domains','table_presentations']:counters[cohort+':'+k].update(r.get(k,{}))
                    for k in r.get('component_counts',{}):counters[cohort+':rows_with_component'][k]+=1
                    for budget in [1024,1536,2048,3072,4096]:c[f'target_over_{budget}']+=r['target_tokens']>budget
                    for budget in [4096,6144,8192]:
                        c[f'deployed_sequence_over_{budget}']+=r['deployed_sequence_tokens']>budget
                        c[f'archived_sequence_over_{budget}']+=r['archived_sequence_tokens']>budget
                    c['rows_repeated_long_literals']+=bool(r.get('repeated_long_literals'))
                issues=r['flags']+([] if r['valid'] else ['invalid'])+(['target_over_2048'] if r['target_tokens']>2048 else [])+(['repeated_long_literals'] if r.get('repeated_long_literals') else [])
                for issue in issues:
                    if len(examples[issue])<12:examples[issue].append(r)
            if replay:old=[];replay=False
            else:
                b=next(it,None)
                if b:pending.append(pool.submit(batch,b))
            if total%1024==0:print(f'{total} rows; {time.time()-t:.1f}s',flush=True)
    import numpy as np
    stats={cohort:{k:{'min':min(v),'p50':float(np.percentile(v,50)),'p90':float(np.percentile(v,90)),'p95':float(np.percentile(v,95)),'p99':float(np.percentile(v,99)),'max':max(v),'mean':float(np.mean(v))} for k,v in vv.items()} for cohort,vv in values.items()}
    dump('structure_summary.json',{'dataset':str(DATA),'rows':total,'input_sha256':hashes,'tokenizer_sha256':hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),'tokenizer_path':str(TOKENIZER),'token_method':'Exact local SentencePiece; BOS added once, Gemma non-thinking text template, <turn|> newline EOS. Remote HF tokenizer parity not asserted.','seconds':time.time()-t,'counts':dict(counters),'stats':stats,'examples':dict(examples)})
    print(f'DONE {total} rows in {time.time()-t:.1f}s',flush=True)

if __name__=='__main__':main()
