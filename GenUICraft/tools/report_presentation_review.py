"""Compare byte-identical saved E2B outputs through two SDK versions, with an offline gallery."""
from collections import Counter
from pathlib import Path
import hashlib
import html
import json
import os

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'GenUICraft/validation/20260929_bixby50_presentation'
SUITES = ['r64_fp32_off','r64_fp32_on','historical50']

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def pointer(state, path):
    try:
        for key in str(path).strip('/').split('/'):
            state = state[int(key)] if isinstance(state,list) else state[key.replace('~1','/').replace('~0','~')]
        return state
    except (KeyError,IndexError,ValueError,TypeError): return None

def graph(path):
    if not path.exists(): return None
    messages=read(path)
    components={}
    state={}
    for msg in messages:
        for node in msg.get('updateComponents',{}).get('components',[]): components[node['id']]=node
        update=msg.get('updateDataModel',{})
        if update.get('path','/') in ('','/') and isinstance(update.get('value'),dict): state.update(update['value'])
    seen=set()
    ordered=[]
    def visit(node_id):
        if node_id in seen or node_id not in components: return
        seen.add(node_id)
        node=components[node_id]
        ordered.append(node)
        for child in node.get('children',[]):
            if isinstance(child,str): visit(child)
    visit('root')
    texts=[n['text'] for n in ordered if n.get('component')=='Text' and isinstance(n.get('text'),str)]
    tables=[]
    for n in ordered:
        if n.get('component') != 'Table': continue
        rows=n.get('rows')
        if rows is None: rows=pointer(state,n.get('statePath'))
        columns=n.get('columns',[])
        labels=[c.get('label',c.get('key','')) if isinstance(c,dict) else str(c) for c in columns]
        tables.append({'title':n.get('title'),'labels':labels,'rows':rows,'rowCount':len(rows) if isinstance(rows,list) else None})
    return {'componentCount':len(components),'reachableCount':len(seen),
            'textCount':len(texts),'texts':texts,'tables':tables,
            'firstText':texts[0] if texts else None}

def url(path): return html.escape(os.path.relpath(path,OUT).replace(os.sep,'/'),quote=True)
def esc(value): return html.escape(str(value))
def dump(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    manifest={m['suite']:m for m in read(OUT/'input_manifest.json')}
    corpus={r['id']:r for r in map(json.loads,(ROOT/'android/app/src/main/assets/genuicraft_bixby50.jsonl').read_text(encoding='utf-8').splitlines())}
    summaries=[]
    cases=[]
    for suite in SUITES:
        before=OUT/'device/before'/suite
        after=OUT/'device/after'/suite
        bs=read(before/'replay_summary.json'); ads=read(after/'replay_summary.json')
        assert bs['total']==ads['total']==manifest[suite]['count']
        assert bs['modelCalls']==ads['modelCalls']==0
        expected={c['id']:c for c in manifest[suite]['cases']}
        br={r['id']:r for r in read(before/'replay_results.json')}
        ar={r['id']:r for r in read(after/'replay_results.json')}
        assert br.keys()==ar.keys()==expected.keys()
        for case_id in sorted(expected):
            bp=before/case_id; ap=after/case_id
            raw_b=(bp/'source.output.express').read_bytes(); raw_a=(ap/'source.output.express').read_bytes()
            assert raw_b==raw_a and hashlib.sha256(raw_a).hexdigest()==expected[case_id]['sha256']
            bg=graph(bp/'recovered.output.a2ui.json'); ag=graph(ap/'recovered.output.a2ui.json')
            gained=list((Counter(ag['texts'] if ag else [])-Counter(bg['texts'] if bg else [])).elements())
            missing=list((Counter(bg['texts'] if bg else [])-Counter(ag['texts'] if ag else [])).elements())
            row={'suite':suite,'id':case_id,'query':corpus[case_id].get('query',''), 'sourceText':corpus[case_id].get('text',''),
                 'rawSha256':expected[case_id]['sha256'],'beforeStatus':br[case_id]['status'],
                 'afterStatus':ar[case_id]['status'],'beforeRepair':br[case_id].get('repairKind'),
                 'afterRepair':ar[case_id].get('repairKind'),'beforeGraph':bg,'afterGraph':ag,
                 'newlyReachableText':gained,'noLongerTextComponents':missing,
                 'beforeCaptureIssues':br[case_id].get('issues',[]),'afterCaptureIssues':ar[case_id].get('issues',[]),
                 'beforeVerticalLimitReached':br[case_id].get('verticalLimitReached'),
                 'afterVerticalLimitReached':ar[case_id].get('verticalLimitReached'),
                 'beforeDir':bp.relative_to(OUT).as_posix(),'afterDir':ap.relative_to(OUT).as_posix()}
            cases.append(row)
        own=[c for c in cases if c['suite']==suite]
        summaries.append({'suite':suite,'before':bs,'after':ads,
                          'casesWithNewReachableText':sum(bool(c['newlyReachableText']) for c in own),
                          'casesWithChangedTextRepresentation':sum(bool(c['noLongerTextComponents']) for c in own)})
    result={'newInference':False,'device':'Fold7 SM-F966B R3CY30QFWLP','sdkBefore':'0.5.5','sdkAfter':'0.5.6',
            'method':'Same raw model bytes; generated-output recovery only, no source fallback. Native screenshots and accessibility traversal.',
            'boundary':'The 50-case historical model and 27 rank64 captures are separate populations. Graph reachability is not source truth or pixel-level fidelity. Missing source facts cannot be invented by repair.',
            'summaries':summaries,'cases':cases}
    dump(OUT/'comparison.json',result)
    sections=[]
    for c in cases:
        columns=[]
        for phase in ('before','after'):
            directory=OUT/c[phase+'Dir']
            initial=directory/'initial.png'
            if not initial.exists(): initial=directory/'failure.png'
            visual=f'<a href="{url(initial)}"><img loading="lazy" src="{url(initial)}" alt="{phase} {c["id"]}"></a>' if initial.exists() else '<p>No renderable screenshot.</p>'
            pages=[p for p in directory.glob('*.png') if p!=initial]
            page_links=' · '.join(f'<a href="{url(p)}">{esc(p.stem)}</a>' for p in pages)
            repaired=directory/'recovered.output.express'
            dsl=esc(repaired.read_text(encoding='utf-8')) if repaired.exists() else 'No repaired document.'
            columns.append(f'<div><h3>{phase.title()} · SDK {"0.5.5" if phase=="before" else "0.5.6"}</h3><p>{esc(c[phase+"Status"])} · {esc(c[phase+"Repair"])}</p>{visual}<p class="pages">{page_links}</p><details><summary>Compiled / repaired Express</summary><pre>{dsl}</pre></details></div>')
        gain=''.join(f'<li>{esc(t)}</li>' for t in c['newlyReachableText'])
        raw=(OUT/c['beforeDir']/'source.output.express').read_text(encoding='utf-8')
        sections.append(f'<section data-suite="{c["suite"]}" data-search="{esc(c["id"]+" "+c["query"])}"><h2>{c["id"]} · {esc(c["suite"])}</h2><p>{esc(c["query"])}</p><div class="pair">{"".join(columns)}</div><details><summary>Newly reachable generated Text ({len(c["newlyReachableText"])})</summary><ul>{gain or "<li>No change in reachable Text nodes; table and styling changes may still apply.</li>"}</ul></details><details><summary>Original model output — unchanged</summary><pre>{esc(raw)}</pre></details><details><summary>Frozen source Markdown</summary><pre>{esc(c["sourceText"])}</pre></details></section>')
    summary_rows=''.join(f'<tr><td>{esc(r["suite"])}</td><td>{r["before"]["rendered"]}/{r["before"]["total"]}</td><td>{r["after"]["rendered"]}/{r["after"]["total"]}</td><td>{r["casesWithNewReachableText"]}</td></tr>' for r in summaries)
    page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bixby50 renderer and recovery review</title><style>
body{font:16px system-ui,sans-serif;margin:0;background:#f6f8fc;color:#18283d}main{max-width:1160px;margin:auto;padding:28px}h1{font-size:32px}p{line-height:1.5}section{margin:25px 0;background:white;border:1px solid #dae2ed;border-radius:14px;padding:22px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}.pair>div{min-width:0}img{width:100%;border:1px solid #dce3ef}details{margin-top:16px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f8;padding:16px;font-size:12px}a{color:#175cc4}.pages{font-size:13px}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid #dce3ef}.controls{position:sticky;top:0;background:#f6f8fcef;padding:14px 0;display:flex;gap:16px;z-index:2}input,select{font:inherit;padding:10px;border:1px solid #b7c5d8;border-radius:6px}input{flex:1}section[hidden]{display:none}@media(max-width:700px){.pair{grid-template-columns:1fr}main{padding:14px}}
</style></head><body><main><h1>Bixby50: renderer and Express recovery</h1><p>Same captured E2B outputs replayed on Fold7 through SDK 0.5.5 and 0.5.6. No new inference or source-text fallback. Raw, repaired and rendered results remain separate.</p><p>The historical 50-case model and newer rank-64 FP32 subsets are different populations. A successful render does not certify source accuracy; missing facts and changed numbers remain model defects.</p><table><tr><th>Captured population</th><th>Before renders</th><th>After renders</th><th>Cases with recovered Text</th></tr>'''+summary_rows+'''</table><p><a href="comparison.json">Machine-readable comparison</a> · <a href="audit/REPORT.md">Source fidelity audit</a></p><div class="controls"><select id="suite"><option value="">All captured outputs</option><option>r64_fp32_off</option><option>r64_fp32_on</option><option>historical50</option></select><input id="search" placeholder="Filter by case ID or query"></div>'''+''.join(sections)+'''</main><script>const filter=()=>{let s=document.getElementById('suite').value,q=document.getElementById('search').value.toLowerCase();document.querySelectorAll('section').forEach(e=>e.hidden=(s&&e.dataset.suite!==s)||!e.dataset.search.toLowerCase().includes(q))};document.getElementById('suite').addEventListener('change',filter);document.getElementById('search').addEventListener('input',filter);</script></body></html>'''
    (OUT/'before_after.html').write_text(page,encoding='utf-8')
    print(json.dumps([{k:v for k,v in x.items() if k not in ('before','after')} | {'beforeRendered':x['before']['rendered'],'afterRendered':x['after']['rendered']} for x in summaries]))

if __name__=='__main__': main()
