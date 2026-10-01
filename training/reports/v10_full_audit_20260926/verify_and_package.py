"""Verify complete audit coverage and immutable inputs, then compress row evidence."""
import csv, gzip, hashlib, json, re, subprocess
from datetime import datetime, timezone
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    summary=json.loads((OUT/'structure_summary.json').read_text())
    assert summary['rows']==92977
    raw=OUT/'structure_rows.jsonl';zipped=OUT/'structure_rows.jsonl.gz'
    last={'train':0,'val':0};n=0;ids=set();valid=0
    opener=lambda:raw.open('rt',encoding='utf8') if raw.exists() else gzip.open(zipped,'rt',encoding='utf8')
    with opener() as f:
        for line in f:
            r=json.loads(line);split=r['split'];assert r['line']==last[split]+1
            assert r['id'] not in ids;ids.add(r['id']);last[split]=r['line'];n+=1;valid+=r['valid']
    assert last=={'train':91115,'val':1862} and n==92977 and valid==92977
    with (OUT/'row_metrics.csv').open(encoding='utf8') as f:assert sum(1 for _ in csv.DictReader(f))==n
    after={}
    for split in last:
        p=ROOT/'training/outputs/datasets/full_data_archive_recovered_v10'/f'{split}.jsonl'
        after[split]=digest(p);assert after[split]==summary['input_sha256'][split]
    missing=[]
    for p in OUT.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf8')):
            if '://' in link or link.startswith('#'):continue
            if not (p.parent/link.split('#')[0]).exists():missing.append([p.name,link])
    assert not missing,missing
    original_hash=None
    if not raw.exists() and (OUT/'audit_manifest.json').exists():
        original_hash=json.loads((OUT/'audit_manifest.json').read_text()).get('raw_structure_rows_sha256')
    if raw.exists():
        original_hash=digest(raw)
        with raw.open('rb') as inp,zipped.open('wb') as out,gzip.GzipFile(filename='',fileobj=out,mode='wb',compresslevel=6,mtime=0) as gz:
            for b in iter(lambda:inp.read(1024*1024),b''):gz.write(b)
        h=hashlib.sha256()
        with gzip.open(zipped,'rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        assert h.hexdigest()==original_hash
        # Only this audit's own redundant uncompressed output is removed.
        assert raw.resolve().parent==OUT.resolve() and raw.name=='structure_rows.jsonl'
        raw.unlink()
    code_paths=['training/src/ir_training/data/express_preparation.py','training/src/ir_training/data/shared_prompt.py','training/src/ir_training/train/sft.py','training/src/ir_training/pipeline/golden_training.py','dataset/src/pipeline/ir_formats/express.py','dataset/src/pipeline/ir_formats/canonical.py','dataset/schema/genuicraft_a2ui_v1_wire.schema.json','dataset/prompts/genui_gen_mobile_a2ui_express_v1.md','GenUICraft/genuicraft/src/main/java/com/samsung/genuicraft/sdk/GenUiTrainedConverter.kt','GenUICraft/genuicraft/src/main/java/com/samsung/genuicraft/sdk/provider/GenUiOutputRepetitionGuard.kt']
    manifest={'completed_utc':datetime.now(timezone.utc).isoformat(),'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'scope':'base v10 audit only; no training, generation, export or Android run; unrelated concurrent workspace edits preserved','rows_checked':last,'strict_valid':valid,'inputs_sha256_after':after,'inputs_unchanged':True,'all_local_markdown_links_resolve':True,'raw_structure_rows_sha256':original_hash,'structure_scan_note':'Full scan resumed once from verified complete output rows; summary seconds describe final invocation, not total wall time. Every coordinate checked exactly once in final evidence.','code_sha256_at_completion':{p:digest(ROOT/p) for p in code_paths},'audit_artifacts_sha256':{p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.suffix in ['.md','.py','.json','.jsonl','.gz','.csv'] and p.name!='audit_manifest.json'}}
    (OUT/'audit_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'rows':last,'valid':valid,'inputs_unchanged':True,'links_valid':True,'compressed_evidence_bytes':zipped.stat().st_size}))

if __name__=='__main__':main()
