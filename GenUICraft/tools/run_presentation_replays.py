"""Replay preserved E2B outputs through the installed SDK; no model/provider calls."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'GenUICraft/validation/20260929_bixby50_presentation'
PKG = 'com.samsung.genuicraft'

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--serial', required=True)
    ap.add_argument('--phase', choices=['before', 'after'], required=True)
    ap.add_argument('--sdk-version', required=True)
    ap.add_argument('--attempt', default='r2')
    ap.add_argument('--suites', default='r64_fp32_off,r64_fp32_on,historical50')
    args = ap.parse_args()
    adb = [shutil.which('adb') or 'adb', '-s', args.serial]
    def run(*parts, timeout=60):
        r = subprocess.run(adb + list(parts), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
        if r.returncode: raise RuntimeError(r.stderr[-2000:])
        return r.stdout
    manifests = {x['suite']: x for x in json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))}
    apk = run('shell','pm','path',PKG).strip().removeprefix('package:')
    provenance = {'sdkVersion':args.sdk_version, 'serial':args.serial,
                  'installedMainApkSha256':run('shell','sha256sum',apk).split()[0],
                  'display':run('shell','wm','size'), 'newInference':False}
    phase = OUT/'device'/args.phase
    phase.mkdir(parents=True, exist_ok=True)
    for suite in args.suites.split(','):
        manifest = manifests[suite]
        case_ids = [x['id'] for x in manifest['cases']]
        run_id = f'presentation_{args.phase}_{suite}_{args.attempt}'
        destination = phase/suite
        assert not destination.exists(), f'Preserve existing run: {destination}'
        remote = f'/sdcard/Android/data/{PKG}/files/sdk_benchmark/{run_id}'
        command = ['shell','am','instrument','-w','-r','-e','class',PKG+'.GenUiSdkBixby50Test#replaySavedBixbyCorpus',
            '-e','replayMode','express_repair_only','-e','sourceRunId',manifest['runId'],'-e','runId',run_id,
            '-e','cases',','.join(case_ids),'-e','maxVerticalSwipes','7','-e','maxHorizontalSwipes','4',
            '-e','renderFontScale','1.0','-e','renderDark','false',PKG+'.test/androidx.test.runner.AndroidJUnitRunner']
        log = phase/(suite+'.instrumentation.txt')
        print(f'START {run_id}: {len(case_ids)} preserved model outputs', flush=True)
        start=time.monotonic()
        with log.open('wb') as f:
            process=subprocess.Popen(adb+command, stdout=f, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(10)
                if time.monotonic()-start > 1200:
                    raise TimeoutError('Replay exceeded watchdog; inspect device before restarting')
        run('pull',remote,str(destination),timeout=90)
        summary=json.loads((destination/'replay_summary.json').read_text(encoding='utf-8'))
        reports=json.loads((destination/'replay_results.json').read_text(encoding='utf-8'))
        assert len(reports)==len(case_ids) and {x['id'] for x in reports}==set(case_ids)
        assert summary['modelCalls']==0 and summary['sourceTextFallbacks']==0
        receipt={**provenance,'sourceManifest':manifest,'command':adb+command,
                 'hostElapsedSeconds':round(time.monotonic()-start,3),'adbExitCode':process.returncode,
                 'summary':summary}
        (destination/'host_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        print(f'DONE {suite}: {summary["rendered"]}/{summary["total"]} render; '
              f'{summary["repairRejected"]} repair rejected; {summary["renderFailures"]} render failures',flush=True)

if __name__=='__main__': main()
