"""Replace one projected live source with its verified complete-record snapshot."""
import json
import hashlib
import pathlib
import shutil

root = pathlib.Path(__file__).resolve().parent
main = root / 'projected'
overlay = root / 'projected_snapshot'
manifest_path = main / 'archive_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf8'))
snapshot = json.loads((overlay/'archive_manifest.json').read_text(encoding='utf8'))
expected = 'dataset_muse_glimmer_100k_r8_2/genui.jsonl'
if manifest.get('source_overlays') or set(snapshot['selected']) != {expected}:
    raise ValueError('Unexpected or already-applied overlay')
if snapshot['status'] != 'verified' or not snapshot['gzip_crc_verified']:
    raise ValueError('Snapshot is not verified')
old, new = manifest['selected'][expected], snapshot['selected'][expected]
original = main / old['projected_path']
replacement = overlay / new['projected_path']
def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
if digest(original) != old['projected_sha256'] or digest(replacement) != new['projected_sha256']:
    raise ValueError('Projected input changed')
backup = original.with_name('genui.training.before_live_snapshot.jsonl')
if backup.exists(): raise FileExistsError(backup)
original.rename(backup)
shutil.copyfile(replacement,original)
manifest['superseded_selected'] = {expected:{**old,'projected_path':backup.relative_to(main).as_posix()}}
manifest['selected'][expected] = {**new,'source_archive_sha256':snapshot['archive_sha256']}
manifest['runs']['dataset_muse_glimmer_100k_r8_2']['genui.jsonl'] = snapshot['runs']['dataset_muse_glimmer_100k_r8_2']['genui.jsonl']
manifest['source_overlays'] = [{k:v for k,v in snapshot.items() if k not in {'selected','runs'}}]
manifest['snapshot_policy'] = 'The live r8_2 root genui.jsonl is replaced by a fixed prefix ending at its last complete JSONL record. Both original archives and the superseded projection are preserved.'
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
print(json.dumps({'old_rows':old['rows'],'new_rows':new['rows'],'old_invalid_json_rows':old['invalid_json_rows'],'new_invalid_json_rows':new['invalid_json_rows'],'snapshot_sha256':snapshot['archive_sha256']}))
