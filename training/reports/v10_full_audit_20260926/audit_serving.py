"""Read-only text-distribution and serving-stop compatibility census."""
import collections, hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'training/outputs/datasets/full_data_archive_recovered_v10'
START=re.compile(r'(?i)(?:\bchildren\s*=\s*|\b(?:column|row|stack|card|grid|section)\s*\(\s*)\[')

def alpha(v):
    if not re.fullmatch('[a-z]{1,6}',v.lower()):return None
    n=0
    for c in v.lower():n=n*26+ord(c)-96
    return n

def candidates(text):
    # The guard ignores a complete block, so replay each list before </a2ui>.
    found=[]
    for m in START.finditer(text):
        after=text[m.end():];end=after.find(']')
        if end<0:continue
        raw=after[:end]
        if not re.fullmatch(r'\s*@?[\w]+(?:\s*,\s*@?[\w]+)*\s*',raw):continue
        ids=[x.strip().lstrip('@') for x in raw.split(',')]
        if any(not (x[0].isalpha() or x[0]=='_') for x in ids):continue
        repeated=max(collections.Counter(ids).values(),default=0)
        run=maxrun=1
        for prev,cur in zip(ids,ids[1:]):
            a,b=alpha(prev),alpha(cur)
            run=run+1 if a is not None and b==a+1 else 1
            maxrun=max(maxrun,run)
        if repeated>=20 or maxrun>=20:found.append({'start':m.start(),'repeated':repeated,'max_sequential':maxrun,'ids':ids})
    return found

def main():
    counts=collections.defaultdict(collections.Counter);examples=[];scaffolds={};domain_counts=collections.Counter();whitespace=0
    for split in ['train','val']:
        for line,raw in enumerate((DATA/(split+'.jsonl')).open(encoding='utf8'),1):
            r=json.loads(raw);s=r['response_text'];t=r['completion'];c=counts[split];c['rows']+=1
            if any(m['content']!=m['content'].strip() for m in r['messages']):c['message_outer_whitespace']+=1
            matches=candidates(t)
            if matches:
                c['stop_candidates']+=1
                if len(examples)<30:examples.append({'split':split,'line':line,'id':r['id'],'matches':matches,'source':s,'completion':t})
            fields=r['metadata'].get('archive_recovery',{})
            c['original_query_missing']+=not bool(r['metadata'].get('query_id'))
            c['intent_unknown']+=r.get('intent_bucket')=='unknown_archive'
            c['contains_sourceText']=c['contains_sourceText']+int('sourceText=' in t)
            c['number_runs_10plus_source']+=bool(re.search(r'\d{10,}',s))
            c['number_runs_10plus_target']+=bool(re.search(r'\d{10,}',t))
            c['repeated_digit_8plus_source']+=bool(re.search(r'(\d)\1{7,}',s))
            c['repeated_digit_8plus_target']+=bool(re.search(r'(\d)\1{7,}',t))
            c['literal_aa_ab_ac_sequence']+=bool(re.search(r'\baa,\s*ab,\s*ac\b',t))
            c['raw_url_in_target']+=bool(re.search(r'https?://',t))
            c['numeric_citation_in_target']+=bool(re.search(r'\[\d+\]',t))
            scaffold=r['messages'][:3];h=hashlib.sha256(json.dumps(scaffold,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
            if h not in scaffolds:scaffolds[h]={'count':0,'system_chars':len(scaffold[0]['content']),'messages':scaffold,'first_id':r['id']}
            scaffolds[h]['count']+=1
    result={'dataset':str(DATA),'counts':dict(counts),'scaffolds':scaffolds,'repetition_guard_candidates':examples,'method':'Replay complete bare component lists before closing sentinel; candidate implies a streaming prefix would meet current Kotlin guard condition. No device generation performed.'}
    (OUT/'serving_distribution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'counts':dict(counts),'scaffolds':len(scaffolds),'candidate_examples':len(examples)},indent=2))

if __name__=='__main__':main()
