"""Verify complete native collection, then apply the published AAR to all outputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODEL = "de60d19c8e1ef06ed1032212d0d4ae5d9b0ec105db72db34a988453004e63e62"
EXPECTED_AAR = "c59eeae4debf4e8428a806ac7b7d7cb1b13f2d3559a12a655a56ef2f0be723b1"


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def write_rows(path, rows):
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+"\n")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir",type=Path,required=True)
    args=ap.parse_args();out=args.run_dir.resolve()
    assert not (out/"model_inputs.jsonl").exists(), "Preserve existing input manifest"
    corpus={r["id"]:r for r in [json.loads(l) for l in (ROOT/"android/app/src/main/assets/genuicraft_bixby50.jsonl").read_text(encoding="utf-8").splitlines()]}
    checkpoints={r["id"]:r for r in [json.loads(l) for l in (ROOT/"GenUICraft/validation/20260929_bixby50_v54_model_scores/scored/checkpoint_r64.jsonl").read_text(encoding="utf-8").splitlines()]}
    reused=load(out/"reused_generations.json");records=[];source_files={};batches={};proofs=[]

    def record(batch, case, serial, origin, expected_mtp):
        config=load(batch/"run_config.json");source=load(case/"source.json");result=load(case/"result.json")
        rt=config["runtime"];case_id=result["id"]
        assert source["id"]==case_id and source["text"]==corpus[case_id]["text"] and source["query"]==corpus[case_id]["query"]
        assert rt["accelerator"]=="GPU" and rt["gpuPrecision"]=="FP32" and rt["maxContextTokens"]==8192 and rt["maxOutputTokens"]==2048
        assert rt["mtpEnabled"]==expected_mtp and rt["temperature"]==0.0 and rt["thinkingEnabled"] is False
        assert rt["sourceFallbackEnabled"] is False and rt["generatedDslRepairEnabled"] is True and rt["requireSourceIntegrity"] is False
        assert config["model"]["basename"]=="gemma4_e2b_a2ui_mobile_r64_qat_compatible.litertlm"
        assert result.get("usedFallback",False) is False and result.get("providerCalls",0)<=1
        raw_file=case/"output.express";raw=raw_file.read_text(encoding="utf-8") if raw_file.is_file() else None
        metrics_path=case/"metrics.json";metrics=load(metrics_path) if metrics_path.exists() else result.get("metrics") or {}
        if raw is not None:
            assert result.get("providerCalls")==1
            assert metrics["runtime"]=="LiteRT-LM/Gemma4/GPU+FP32"+("+MTP" if expected_mtp else "")
            assert result["renderedPromptSha256"]==checkpoints[case_id]["runtime"]["prompt_sha256"]
        else:
            assert result["status"] in ("timeout","runtime_error","case_error","cancelled"), f"Unexplained missing output {case}"
        key=str(batch)
        if key not in batches:
            session_path=batch/"session_metrics.json"
            session=load(session_path) if session_path.exists() else {}
            if session and not session.get("error"):
                assert session.get("speculativeDecodingEnabled")==expected_mtp
            batches[key]={"serial":serial,"mtp":expected_mtp,"cases":config["cases"],"generation_origin":origin,"session_metrics":session,"run_config":config}
        for path in (batch/"run_config.json",case/"source.json",case/"result.json",metrics_path,raw_file):
            if path.is_file():source_files[str(path)]=sha(path)
        records.append({"model":"litert_mtp_on" if expected_mtp else "litert_mtp_off","id":case_id,
            "query":source["query"],"response_text":source["text"],"raw_output":raw,
            "raw_output_sha256":hashlib.sha256(raw.encode("utf-8")).hexdigest() if raw is not None else None,
            "serial":serial,"device_model":config["device"]["model"],"batch":batch.name,"source_run":str(batch),
            "generation_origin":origin,"runtime":metrics,"provider_result":result,
            "finish_reason":metrics.get("finishReason") or result["status"],"raw_input_file":str(raw_file) if raw is not None else None})

    for item in reused:
        path=Path(item["raw_input_file"]);assert sha(path)==item["sha256"]
        record(path.parent.parent,path.parent,item["serial"],"prior_verified",item["model"]=="litert_mtp_on")
    for alias in ("flip8","fold7"):
        directory=out/"devices"/alias;plan=load(directory/"plan.json")
        assert (directory/"COMPLETE.json").exists(), f"Shard still running: {alias}"
        provenance=load(directory/"provenance.json");assert provenance["modelSha256"]==EXPECTED_MODEL
        for entry in plan["batches"]:
            batch=directory/"batches"/entry["name"];summary=load(batch/"summary.json")
            assert summary["runComplete"] and summary["completed"]==len(entry["cases"])
            for case_id in entry["cases"]:record(batch,batch/case_id,plan["serial"],"new_run",entry["mtp"])
    by_key={(r["model"],r["id"]):r for r in records};expected={(m,f"BXP-{i:03}") for m in ("litert_mtp_off","litert_mtp_on") for i in range(1,51)}
    assert len(records)==len(by_key)==100 and set(by_key)==expected
    records=sorted(records,key=lambda r:(r["model"],r["id"]))
    for i in range(1,51):assert by_key[("litert_mtp_off",f"BXP-{i:03}")]["serial"]==by_key[("litert_mtp_on",f"BXP-{i:03}")]["serial"]
    write_rows(out/"model_inputs.jsonl",records)
    save(out/"generation_source_hashes.json",source_files);save(out/"batch_runtime_evidence.json",batches)
    aar=ROOT/"GenUICraft/build/repo/com/samsung/genuicraft/genuicraft/0.5.6/genuicraft-0.5.6.aar";assert sha(aar)==EXPECTED_AAR
    runtime=out/"runtime";runtime.mkdir(exist_ok=True)
    with zipfile.ZipFile(aar) as archive:(runtime/"genuicraft-0.5.6-classes.jar").write_bytes(archive.read("classes.jar"))
    cache=Path(os.environ["USERPROFILE"])/".gradle/caches/modules-2/files-2.1"
    dependencies=[next((cache/"com.google.code.gson/gson/2.11.0").glob("*/*.jar")),next((cache/"org.jetbrains.kotlin/kotlin-stdlib/2.2.21").glob("*/*.jar"))]
    # Preserve the exact dependency bytes alongside the run so replay does not
    # depend on later Gradle cache cleanup or changes.
    local_dependencies=[]
    for dependency in dependencies:
        local=runtime/dependency.name;shutil.copyfile(dependency,local)
        assert sha(local)==sha(dependency)
        local_dependencies.append(local)
    jars=[runtime/"genuicraft-0.5.6-classes.jar",*local_dependencies]
    cp=os.pathsep.join(map(str,jars));classes=runtime/"driver_classes";classes.mkdir(exist_ok=True)
    driver=ROOT/"GenUICraft/tools/GenUiRepairReplay.java"
    subprocess.run([shutil.which("javac"),"-encoding","UTF-8","-cp",cp,"-d",str(classes),str(driver)],check=True)
    repair_input=runtime/"repair_input.jsonl";available=[{k:r[k] for k in ("model","id","raw_output")} for r in records if r["raw_output"] is not None]
    write_rows(repair_input,available)
    captured=runtime/"repair_captured_outcomes.jsonl"
    cmd=[shutil.which("java"),"-Xmx1g","-cp",str(classes)+os.pathsep+cp,"GenUiRepairReplay",str(repair_input),str(captured)]
    result=subprocess.run(cmd,capture_output=True,text=True,encoding="utf-8",errors="replace")
    (runtime/"repair.log").write_text(result.stdout+result.stderr,encoding="utf-8");print(result.stdout,flush=True);result.check_returncode()
    repairs=[json.loads(l) for l in captured.read_text(encoding="utf-8").splitlines()]
    for r in records:
        if r["raw_output"] is None:
            repairs.append({"model":r["model"],"id":r["id"],"repair_success":False,"repair_kind":"RUNTIME_NO_OUTPUT","repair_error":r["provider_result"].get("error"),"model_calls":0,"source_text_supplied_to_repair":False,"source_fallback_enabled":False})
    repairs.sort(key=lambda r:(r["model"],r["id"]))
    assert len(repairs)==100
    for repair in repairs:
        r=by_key[(repair["model"],repair["id"])];case=Path(r["source_run"])/r["id"]
        if r["generation_origin"]=="new_run" and (case/"a2ui.json").exists() and repair["repair_success"]:
            equivalent=load(case/"a2ui.json")==json.loads(repair["repaired_a2ui_json"])
            proofs.append({"model":r["model"],"id":r["id"],"check":"new_device_A2UI_matches_host_AAR","matches":equivalent})
            assert equivalent, f"Host/device SDK repair mismatch: {r['model']}/{r['id']}"
    write_rows(out/"repair_outcomes.jsonl",repairs)
    save(out/"repair_device_parity.json",proofs)
    save(out/"repair_provenance.json",{"aar":str(aar),"aar_sha256":sha(aar),"driver_source":str(driver),"driver_sha256":sha(driver),
        "classpath":[{"path":str(p),"sha256":sha(p)} for p in jars],"command":cmd,"new_model_inference":False,
        "source_text_supplied_to_repair":False,"fallback_enabled":False,"attempted_outputs":len(available),"runtime_no_output":100-len(available),
        "new_device_a2ui_parity_checks":len(proofs),"new_device_a2ui_parity_passed":all(p["matches"] for p in proofs)})
    print(f"Ready for scoring: {len(records)} total generations, {len(available)} raw outputs, {sum(r['repair_success'] for r in repairs)} SDK documents",flush=True)


if __name__=="__main__":main()
