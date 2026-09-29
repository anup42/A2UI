"""Stage byte-identical captured E2B output for before/after SDK replay (no inference)."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'GenUICraft/validation/20260929_bixby50_presentation'

def main():
    suites = {
        'historical50': sorted((ROOT / 'GenUICraft/validation/20260921_e2b_mobile_full50/native').glob('BXP-*/output.express')),
        'r64_fp32_off': sorted((ROOT / 'GenUICraft/validation/20260928_fp32_bixby50_mtp/batches').glob('group_*_mtp_off/BXP-*/output.express')),
        'r64_fp32_on': sorted((ROOT / 'GenUICraft/validation/20260928_fp32_bixby50_mtp/batches').glob('group_*_mtp_on/BXP-*/output.express')),
    }
    expected = {'historical50': 50, 'r64_fp32_off': 15, 'r64_fp32_on': 12}
    manifests = []
    for name, files in suites.items():
        if name != 'historical50':
            files = [f for f in files if (f.parent / 'result.json').is_file()]
        assert len(files) == expected[name], (name, len(files))
        assert len({f.parent.name for f in files}) == len(files)
        run_id = 'presentation_input_' + name
        directory = OUT / 'inputs' / run_id
        directory.mkdir(parents=True, exist_ok=True)
        entries = []
        for raw in files:
            target = directory / raw.parent.name
            target.mkdir(exist_ok=True)
            dest = target / raw.name
            data = raw.read_bytes()
            if dest.exists():
                assert dest.read_bytes() == data, 'Never overwrite a changed replay input'
            else:
                dest.write_bytes(data)
            source = raw.parent / 'source.json'
            if source.exists():
                shutil.copyfile(source, target / 'source.json')
            entries.append({'id': raw.parent.name, 'source': raw.relative_to(ROOT).as_posix(),
                            'sha256': hashlib.sha256(data).hexdigest()})
        manifest = {'suite': name, 'runId': run_id, 'count': len(entries), 'cases': entries,
                    'newInference': False, 'kind': 'captured_model_output_replay',
                    'profile': 'original_v10_gpu_mtp' if name == 'historical50' else 'corrected_rank64_gpu_fp32',
                    'provenanceNote': 'Historical captures. Keep separate; not a fresh full50 evaluation of the rank64 model.'}
        (directory / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        manifests.append(manifest)
    (OUT / 'input_manifest.json').write_text(json.dumps(manifests, indent=2) + '\n', encoding='utf-8')
    print(json.dumps([{'suite': x['suite'], 'count': x['count'], 'runId': x['runId']} for x in manifests]))

if __name__ == '__main__':
    main()
