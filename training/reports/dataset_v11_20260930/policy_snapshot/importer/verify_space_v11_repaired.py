#!/usr/bin/env python3
"""Independently verify the emitted merge, compute statistics, and seal it."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import html
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from prepare_space_v11_repaired import Stats, bounded_map, describe, dump, dumps, file_sha, gates, goldens, init, lines, policy_files, reassessment_policy_files, sha


def json_value(value):
    """Compare replay evidence in its persisted JSON representation.

    Repair helpers may return Python tuples, which JSON stores as arrays.
    This normalizes that representation only; all values and array order remain.
    """
    return json.loads(dumps(value))


def inspect(item):
    from ir_training.data.express_preparation import _api, SerializedTarget, TASK_PREFIX
    from ir_training.data.archive_recovery import placeholder_tokens
    split,n,raw = item
    row = json.loads(raw)
    source,target = row["response_text"],row["completion"]
    active,_,semantic_hash,references = _api()
    graph = active.decode_express_completion(target)
    semantic = semantic_hash(graph)
    if row["target_format"] != "a2ui_express_v1" or row["source_format"] != "a2ui_express_v1":
        raise ValueError(f"Wrong format at {split}:{n}")
    if row["messages"][-2] != {"role":"user","content":TASK_PREFIX+source} or row["messages"][-1] != {"role":"assistant","content":target}:
        raise ValueError(f"Message binding mismatch at {split}:{n}")
    if not placeholder_tokens(target) <= placeholder_tokens(source):
        raise ValueError(f"Unbound target reference at {split}:{n}")
    meta = row["metadata"]
    if row["source_id"] != meta["source_id"] or meta["assigned_split"] != split:
        raise ValueError(f"Identity or split mismatch at {split}:{n}")
    evidence = meta.get("space_import") or meta.get("archive_recovery")
    if not evidence: raise ValueError("Missing provenance")
    for key,value in (("effective_source_sha256",sha(source)),("effective_target_sha256",sha(target)),("effective_semantic_sha256",semantic)):
        if evidence[key] != value: raise ValueError(f"Provenance mismatch: {split}:{n}:{key}")
    origin = (meta.get("space_import") or {}).get("run","v10")
    if origin != "v10":
        repairs=evidence["repair_pipeline"]
        if bool(row.get("repair",{}).get("applied")) != bool(repairs["changes"]):
            raise ValueError("Repair flag mismatch")
        if (evidence["target_semantics_changed"] or evidence["source_content_changed"]) and not repairs["proofs"]:
            raise ValueError("Unproven semantic modification")
        if evidence["reassessment"]["eligible"] is not True or evidence["reassessment"]["reasons"]:
            raise ValueError("Unresolved reassessment admitted")
    checked = SerializedTarget(target,semantic,tuple(e["type"] for e in graph["elements"].values()),tuple(edge.reference_kind for e in graph["elements"].values() for edge in references(e)),graph)
    profile=describe(row,checked,origin)
    from ir_training.data.archive_refinement import source_signature
    profile["query_signature"] = source_signature(meta["query_text"]) if meta.get("query_text") else None
    profile["scenario"] = meta.get("scenario_family_id")
    if origin != "v10":
        profile["source_evidence"] = {k:evidence[k] for k in ("archive_record","original_source_sha256","original_target_sha256","repair_pipeline","reassessment","source_content_changed","target_semantics_changed")}
        profile["source_evidence"]["target_sha"]=sha(target)
        profile["source_evidence"]["url_map"]=(meta.get("url_preprocessing") or {}).get("url_map",{})
    elif meta.get("v11_known_hold_repair"):
        profile["base_repair_evidence"]={"case":meta["v11_known_hold_repair"]["case"],"row_sha256":hashlib.sha256(raw).hexdigest()}
    return split,n,raw,row["id"],profile


def replay_new(item):
    """Recompute the repair and admission from immutable Stage 3 evidence."""
    from ir_training.data.express_preparation import _api,serialize_checked
    from ir_training.data.reference_binding import _reference_map
    from space_v11_repairs import normalize_for_repair,repair_chain
    from space_v11_reassessment import reassess
    record,expected=item
    coordinate=(record["archive_record"]["member"],record["archive_record"]["line"])
    if record["archive_record"] != expected["archive_record"] or sha(record["response_text"]) != expected["original_source_sha256"] or sha(record["completion"]) != expected["original_target_sha256"]:
        raise ValueError(f"Stage 3 lineage mismatch: {coordinate}")
    active,express,semantic_hash,_=_api()
    original=active.decode_express_completion(record["completion"])
    normalized,mixed=normalize_for_repair(record["response_text"],original,_reference_map(record))
    repaired=repair_chain(normalized.response_text,normalized.canonical_graph,normalized.url_map,original_source_sha256=sha(record["response_text"]))
    if repaired["issues"]:
        raise ValueError(f"Replayed repair has issues: {coordinate}: {repaired['issues']}")
    checked=serialize_checked(express.encode(repaired["graph"],shorten_ids=False),"root-first")
    expected_repair=expected["repair_pipeline"]
    if json_value(repaired["proofs"]) != expected_repair["proofs"] or repaired["changes"] != expected_repair["changes"]:
        raise ValueError(f"Repair proof replay mismatch: {coordinate}")
    if repaired["url_map"] != expected["url_map"]:
        raise ValueError(f"Emitted reference identity map differs from replay: {coordinate}")
    if sha(normalized.response_text) != expected_repair["normalized_input_source_sha256"] or semantic_hash(normalized.canonical_graph) != expected_repair["normalized_input_semantic_sha256"]:
        raise ValueError(f"Normalization replay mismatch: {coordinate}")
    if sha(repaired["source"]) != expected["source_sha"] or checked.semantic_sha256 != expected["semantic_sha"] or sha(checked.text) != expected["target_sha"]:
        raise ValueError(f"Output differs from source-proven repair: {coordinate}")
    if expected["source_content_changed"] != (repaired["source"]!=normalized.response_text) or expected["target_semantics_changed"] != (checked.graph!=normalized.canonical_graph):
        raise ValueError(f"Semantic change flag mismatch: {coordinate}")
    reassessed=reassess(repaired["source"],checked.text,checked,record["training_acceptance"],repaired["url_map"])
    if not reassessed["eligible"] or reassessed["reasons"]:
        raise ValueError(f"Independent reassessment failed: {coordinate}: {reassessed}")
    # Wall-time telemetry is allowed to differ; admission itself must agree.
    def without_timing(value):
        if isinstance(value,dict):return {key:without_timing(child) for key,child in value.items() if key not in {"elapsed_ms","rescore_elapsed_ms"}}
        if isinstance(value,list):return [without_timing(child) for child in value]
        return value
    if json_value(without_timing(reassessed["evidence"])) != without_timing(expected["reassessment"]["evidence"]):
        raise ValueError(f"Reassessment evidence mismatch: {coordinate}")
    return bool(repaired["changes"]),not record["training_acceptance"]["eligible"]


def table(headers,rows):
    esc=lambda x:html.escape(str(x))
    return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def report_html(report,cleaning,archive):
    datasets=report["statistics"]
    out=['<!doctype html><html lang="en"><meta charset="utf-8"><title>v10, v11 and v11s dataset report</title>',
         '<style>body{font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 24px;color:#172331;background:#fafbfd}h1,h2{line-height:1.2}h2{margin-top:36px}table{border-collapse:collapse;width:100%;background:white;margin:16px 0}td,th{padding:8px 12px;text-align:left;border-bottom:1px solid #dde3ea}th{background:#eaf0f7}tr:hover{background:#f0f5fa}code{word-break:break-all}p{max-width:1050px}.note{border-left:4px solid #4177ab;padding:10px 16px;background:#eef5fc}</style>',
         '<h1>Training data v10, v11 and v11s</h1><p>Build date: 30 September 2026 (Asia/Calcutta). All counts below are computed from actual files.</p>',
         '<p class="note">v10 is the byte-preserved baseline. v11 contains screened/repaired v10 rows plus cleaned/repaired Space data. v11s contains only the new Space data, with detected v10 source overlaps removed. Use v11s/val for held-out evaluation: those rows are also held out in v11. v11s/train is part of v11/train. This is the expanded source-proven repair release requested after the initial conservative build.</p>',
         '<h2>Dataset comparison</h2>']
    rows=[]
    for metric,fn in [("Training rows",lambda d:d["train"]["rows"]),("Validation rows",lambda d:d["val"]["rows"]),("Total rows",lambda d:d["combined"]["rows"]),("Unique normalized sources",lambda d:d["combined"]["unique_normalized_sources"]),("Source families",lambda d:d["combined"]["source_families"]),("Unique exact pairs",lambda d:d["combined"]["unique_exact_pairs"]),("Original queries available",lambda d:d["combined"]["distributions"].get("flags",{}).get("with_original_query",0))]:
        rows.append([metric,*[f'{fn(datasets[k]):,}' for k in ("v10","v11","v11s")]])
    out.append(table(["Metric","v10","v11","v11s"],rows))
    history=cleaning["v10_creation_history"]
    out.append('<h2>How v10 was created</h2><p>v10 descends from the recovered messages archive: transport/encoding recovery, strict Express validation, source-bound URL handling, duplicate removal, paragraph/letter completeness checks, benchmark exclusion, and a source-family split with a 2% validation target. The final v9-to-v10 pass applied only source-proven repairs and quarantined unresolved semantic issues. Its later 100-case manual audit was not applied to the published v10 bundle. This build reconsiders those 31 flagged families using hash-bound source evidence; unresolved cases remain excluded.</p>')
    out.append(table(["Historical stage","Records"],[["Original archive",history["original_archive_records"]],["v9 input to final semantic pass",sum(history["v9_input_rows"].values())],["v10 final",sum(history["v10_output_rows"].values())]]))
    out.append(table(["v9-to-v10 outcome","Train","Validation"],[[outcome,history["v9_to_v10_outcomes"]["train"].get(outcome,0),history["v9_to_v10_outcomes"]["val"].get(outcome,0)] for outcome in ("KEEP","REPAIR","QUARANTINE")]))
    out.append('<h2>Archive and raw generation inventory</h2><p>Downloaded archive: '+f'{archive["archive_bytes"]:,} bytes; {archive["archive_members"]:,} members; {archive["archive_uncompressed_member_bytes"]:,} uncompressed bytes. Gzip CRC checked. Historical Stage 3 backups were excluded from admission; they remain in the original archive.</p>')
    raw_rows=[]
    for run,counts in cleaning["new_input_inventory"].items():
        raw_rows.append([run,counts.get("queries.jsonl",0),counts.get("responses.jsonl",0),counts.get("genui_records",0),counts.get("stage2_responses_without_stage3_record",0),counts.get("eligible:True",0),datasets["v11s"]["combined"]["distributions"]["origin"].get(run,0)])
    out.append(table(["Run","Queries","Responses","Stage 3 records","Responses missing Stage 3","Generator eligible","Final new rows"],raw_rows))
    if archive.get("source_overlays"):
        out.append('<p>The live r8_2 Stage 3 file changed during the initial tar read. A separately downloaded, verified snapshot ending at a complete JSONL record replaces that one source for training. Both archives and the superseded projection are preserved; caches and live logs are not training inputs.</p>')
    out.append('<h2>Cleaning and merge</h2><p>Existing v10 train/validation memberships are preserved. Unmodified retained rows keep their original bytes; repaired held rows carry exact before/after evidence. New data uses v10 reference, text, table/state-path and encoding repair rules, followed by strict schema, content, reference closure, idempotence and Stage 1/2/3 lineage checks. The original generator flags remain in metadata. Review-only flags are reassessed against renderer-visible lexical/numeric evidence; specific structural, action, media, table, role or dynamic findings require fresh checks. Unresolved cases stay quarantined. Repaired outputs are generated mechanically by the recorded policy, with no invented facts or ad hoc edits to sample JSON.</p>')
    out.append('<p>The initial conservative build retained 13,372 new rows by requiring the original generator eligibility flag. Its files remain in v11_conservative/v11s_conservative. The generator gate is explicitly uncalibrated: its strict content-block matching can reject faithful regrouping, so its original rejection count is not a measured defect rate.</p>')
    out.append(table(["New-record outcome","Rows"],sorted(cleaning["new_cleaning_counts"].items())))
    out.append('<h3>Source-proven repairs retained (row counts overlap by repair kind)</h3>')
    out.append(table(["Repair kind","New rows"],sorted(cleaning.get("new_retained_repair_kinds_overlap",{}).items(),key=lambda x:-x[1])))
    out.append('<h3>Known v10 findings reprocessed</h3>')
    out.append('<pre>'+html.escape(json.dumps(cleaning.get("base_repairs",{}),indent=2))+'</pre>')
    out.append('<h3>v10 exclusions from v11 (reason counts can overlap)</h3>')
    out.append(table(["Reason","Rows"],sorted(cleaning["base_filtering"]["reasons_overlap"].items(),key=lambda x:-x[1])))
    out.append('<h3>New-record exclusions (reason counts can overlap)</h3>')
    out.append(table(["Reason","Rows"],sorted(cleaning["new_quarantine_reasons_overlap"].items(),key=lambda x:-x[1])))
    out.append('<h2>Lengths and structural complexity</h2><p>Character/word/element counts are exact. Generation token counts, when present in the JSON report, are teacher telemetry and may include reasoning; they are not model-specific training token counts. No records were truncated.</p>')
    for metric in ("source_chars","source_words","target_chars","target_lines","component_count"):
        out.append('<h3>'+html.escape(metric)+'</h3>')
        out.append(table(["Dataset","Min","Mean","Median","P90","P95","P99","Max"],[[name,*[datasets[name]["combined"]["lengths"][metric][k] for k in ("min","mean","p50","p90","p95","p99","max")]] for name in ("v10","v11","v11s")]))
    out.append('<h2>Components present in rows</h2>')
    keys=sorted(set().union(*(set(datasets[n]["combined"]["distributions"]["component_presence"]) for n in datasets)))
    out.append(table(["Component","v10","v11","v11s"],[[k,*[datasets[n]["combined"]["distributions"]["component_presence"].get(k,0) for n in ("v10","v11","v11s")]] for k in keys]))
    for dimension in ("intent","difficulty","modality","table_domains","teacher","source_review_status","flags"):
        out.append('<h2>'+html.escape(dimension.replace('_',' ').title())+'</h2>')
        keys=sorted(set().union(*(set(datasets[n]["combined"]["distributions"].get(dimension,{})) for n in datasets)))
        out.append(table([dimension,"v10","v11","v11s"],[[k,*[datasets[n]["combined"]["distributions"].get(dimension,{}).get(k,0) for n in ("v10","v11","v11s")]] for k in keys]))
    out.append('<h2>Verification</h2>')
    out.append(table(["Check","Result"],[(k,json.dumps(v)) for k,v in report["checks"].items()]))
    out.append('<h2>Quality limits and training use</h2><p>This is an offline-screened training candidate. Heuristic content checks do not certify all source facts or rendering quality. The complete v10 manual audit was a 100-row sample, not a population accuracy estimate. New source factual-review status is preserved in metadata and shown above. Model-specific tokenizer limits, training, Golden performance and Android rendering were not run. Use the training launcher with a fresh output run and --input-dir pointing to v11 or v11s; allow its normal tokenizer preparation to run. The full data is A2UI Express v1, matching v10, with a validated flat canonical graph underneath.</p>')
    out.append('<p>Companion artifacts: statistics.json, verification.json, cleaning_summary.json, new_decisions.jsonl, v10_decisions.jsonl, new_family_assignments.jsonl, and each dataset manifest. Rejected records remain recoverable by archive member, line and SHA-256.</p></html>')
    return '\n'.join(out)


def verify(args):
    init(args.policy_repo)
    from ir_training.data.archive_refinement import NearSourceIndex
    from ir_training.data.express_preparation import serialize_checked,_wire_validator
    from pipeline.ir_formats import a2ui_wire
    start=time.monotonic()
    archive=json.loads((args.project_dir/"archive_manifest.json").read_text())
    for member,entry in archive["selected"].items():
        if file_sha(args.project_dir/entry["projected_path"]) != entry["projected_sha256"]:
            raise ValueError(f"Changed projected source: {member}")
    cleaning=json.loads((args.audit_dir/"cleaning_summary.json").read_text())
    if file_sha(args.project_dir/"archive_manifest.json") != cleaning["archive_manifest_sha256"]:
        raise ValueError("Original archive projection manifest changed")
    if cleaning["implementation_sha256"] != policy_files(args.policy_repo): raise ValueError("Policy drift")
    if cleaning["source_quality_recheck_implementation_sha256"] != file_sha(args.policy_repo/"dataset/src/pipeline/source_quality.py"):
        raise ValueError("Source quality policy drift")
    for name,digest in cleaning["repair_implementation_sha256"].items():
        if file_sha(Path(__file__).parent/name) != digest:
            raise ValueError(f"Repair implementation drift: {name}")
    if cleaning["reassessment_reference_implementation_sha256"] != reassessment_policy_files(args.policy_repo):
        raise ValueError("Reference reassessment implementation drift")
    base=json.loads((args.work_dir/"base_report.json").read_text())
    for name,digest in base["retained_split_sha256"].items():
        if file_sha(args.work_dir/name)!=digest:raise ValueError("Screened/repaired v10 base changed")
    if file_sha(Path(__file__).parent/"repair_v11_base.py") != base["base_repair_script_sha256"] or file_sha(Path(__file__).parent/"space_v11_known_hold_repairs.py") != base["base_repair_implementation_sha256"]:
        raise ValueError("Base repair implementation drift")
    for name,digest in base["input_split_sha256"].items():
        if file_sha(Path(base["base_dir"])/name) != digest: raise ValueError("v10 changed")
    manifests={name:json.loads((args.output_root/name/"manifest.json").read_text()) for name in ("v11","v11s")}
    for name,manifest in manifests.items():
        for filename,entry in manifest["outputs"].items():
            path=args.output_root/name/filename
            if file_sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                raise ValueError(f"Output hash mismatch: {path}")
    stats={name:{s:Stats() for s in ("train","val","combined")} for name in ("v11","v11s")}
    normalized={s:{} for s in ("train","val")}
    family_sets={s:set() for s in ("train","val")}
    exact_sets={s:set() for s in ("train","val")}
    query_sets={s:set() for s in ("train","val")}
    scenario_sets={s:set() for s in ("train","val")}
    new_normalized=set()
    source_evidence={}
    base_repair_evidence={}
    identifiers=set()
    seen_pairs=set()
    old_hash={s:hashlib.sha256() for s in ("train","val")}
    new_hash={s:hashlib.sha256() for s in ("train","val")}
    counts=defaultdict(Counter)
    samples={s:[] for s in ("train","val")}
    holds=set(json.loads((args.work_dir/"known_v10_holds.json").read_text()))
    def items():
        for split in ("train","val"):
            with (args.output_root/"v11"/f"{split}.jsonl").open("rb") as f:
                for n,raw in enumerate(f,1): yield split,n,raw
    for split,n,raw,rid,profile in bounded_map(inspect,items(),args.policy_repo,args.workers,{}):
        if rid in identifiers: raise ValueError("Duplicate row identity")
        identifiers.add(rid)
        if profile["pair_sha"] in seen_pairs: raise ValueError("Duplicate exact source/target pair")
        seen_pairs.add(profile["pair_sha"])
        if profile["family"] in holds: raise ValueError("Known manual hold admitted")
        stats["v11"][split].add(profile);stats["v11"]["combined"].add(profile)
        normalized[split][profile["normalized_sha"]]=profile["signature"]
        family_sets[split].add(profile["family"]);exact_sets[split].add(profile["source_sha"])
        if profile["query_signature"]: query_sets[split].add(profile["query_signature"])
        if profile["scenario"]: scenario_sets[split].add(profile["scenario"])
        counts["v11"][split]+=1
        if profile["origin"] == "v10":
            if n > base["retained_rows"][split]: raise ValueError("Unexpected old record order")
            old_hash[split].update(raw)
            if profile.get("base_repair_evidence"):
                base_repair_evidence[rid]=profile["base_repair_evidence"]
        else:
            if n <= base["retained_rows"][split]: raise ValueError("Unexpected new record order")
            new_hash[split].update(raw)
            counts["v11s"][split]+=1
            stats["v11s"][split].add(profile);stats["v11s"]["combined"].add(profile)
            new_normalized.add(profile["normalized_sha"])
            evidence=profile["source_evidence"]
            coordinate=(evidence["archive_record"]["member"],evidence["archive_record"]["line"])
            if coordinate in source_evidence: raise ValueError("Stage 3 record used twice")
            source_evidence[coordinate]={**evidence,"source_sha":profile["source_sha"],"semantic_sha":profile["semantic_sha"]}
        rank=-int(sha(f"20260930:{rid}"),16)
        sample=(rank,raw)
        if len(samples[split]) < 256: heapq.heappush(samples[split],sample)
        elif rank > samples[split][0][0]: heapq.heapreplace(samples[split],sample)
        if sum(counts["v11"].values())%5000==0: print(f"Independent output checks {sum(counts['v11'].values()):,}; elapsed {time.monotonic()-start:.0f}s",flush=True)
    for split in ("train","val"):
        if old_hash[split].hexdigest() != file_sha(args.work_dir/f"{split}.jsonl"): raise ValueError("Retained v10 prefix changed")
        if new_hash[split].hexdigest() != manifests["v11s"]["outputs"][f"{split}.jsonl"]["sha256"]: raise ValueError("New-only split differs from merge suffix")
    for name in ("v11","v11s"):
        if dict(counts[name]) != manifests[name]["rows"]: raise ValueError("Output counts differ")
    from space_v11_known_hold_repairs import repair_known_hold
    base_repairs={row["id"]:row for row in json.loads((args.work_dir/"known_v10_repairs.json").read_text())}
    if set(base_repairs)!=set(base_repair_evidence):raise ValueError("Known-hold repair inventory mismatch")
    base_replays=0
    for split in ("train","val"):
        for number,raw,row in lines(Path(base["base_dir"])/f"{split}.jsonl"):
            event=base_repairs.get(row["id"])
            if event is None:continue
            if event["split"]!=split or event["line"]!=number or event["original_row_sha256"]!=hashlib.sha256(raw).hexdigest():
                raise ValueError("Original known-hold evidence mismatch")
            fixed,proofs,unresolved=repair_known_hold(row,event["case"])
            digest=hashlib.sha256((dumps(fixed)+"\n").encode("utf8")).hexdigest()
            if unresolved or json_value(proofs)!=event["proofs"] or digest!=event["output_row_sha256"] or digest!=base_repair_evidence[row["id"]]["row_sha256"]:
                raise ValueError(f"Known-hold source-proven repair replay mismatch: case {event['case']}, {row['id']}")
            base_replays+=1
    if base_replays!=len(base_repairs):raise ValueError("Incomplete known-hold repair replay")
    overlaps={"cross_split_families":len(family_sets["train"]&family_sets["val"]),"cross_split_exact_sources":len(exact_sets["train"]&exact_sets["val"]),"cross_split_normalized_sources":len(set(normalized["train"])&set(normalized["val"]))}
    overlaps.update({"cross_split_available_queries":len(query_sets["train"]&query_sets["val"]),"cross_split_available_scenarios":len(scenario_sets["train"]&scenario_sets["val"])})
    if any(overlaps.values()): raise ValueError(dumps(overlaps))
    base_norm={r["normalized_sha"] for _,_,r in lines(args.work_dir/"base_index.jsonl")}
    if base_norm & new_normalized: raise ValueError("New-only data overlaps v10")
    evidence_checked=0
    replay_repaired,replay_recovered=0,0
    def replay_items():
        for path in sorted(args.project_dir.glob("*/genui.training.jsonl")):
            for _,_,record in lines(path):
                coordinate=(record["archive_record"]["member"],record["archive_record"]["line"])
                expected=source_evidence.get(coordinate)
                if expected is not None:
                    yield record,expected
    for repaired,recovered in bounded_map(replay_new,replay_items(),args.policy_repo,args.workers,{}):
        evidence_checked+=1;replay_repaired+=repaired;replay_recovered+=recovered
        if evidence_checked%2000==0:
            print(f"Original Stage 3 repairs/admission independently replayed: {evidence_checked:,}; elapsed {time.monotonic()-start:.0f}s",flush=True)
    if evidence_checked != sum(counts["v11s"].values()): raise ValueError("Incomplete Stage 3 provenance replay")
    golden_sources,golden_hashes=goldens(args.policy_repo)
    if golden_hashes != cleaning["benchmark_sha256"]: raise ValueError("Benchmarks changed")
    golden_index=NearSourceIndex(golden_sources)
    golden_exact=set(golden_sources.values())
    validation_index=NearSourceIndex(normalized["val"])
    overlap_details=[]
    for split in ("train","val"):
        for i,(key,signature) in enumerate(normalized[split].items(),1):
            if signature in golden_exact or list(golden_index.matches(signature,containment=True)):
                overlap_details.append({"type":"benchmark_overlap","split":split,"normalized_sha256":key})
            if split=="train":
                for other,jaccard,containment in validation_index.matches(signature):
                    overlap_details.append({"type":"train_val_near_overlap","train":key,"val":other,"jaccard":jaccard})
            if i%10000==0: print(f"Independent source leakage checks {split}: {i:,}",flush=True)
    if overlap_details:
        dump(args.audit_dir/"detected_overlap_failures.json",overlap_details)
        raise ValueError(f"Found {len(overlap_details)} source overlaps; outputs are not sealed")
    sample_count=0
    for split,selected in samples.items():
        for _,raw in selected:
            row=json.loads(raw)
            checked=serialize_checked(row["completion"],"root-first")
            _wire_validator().validate(a2ui_wire.encode(checked.graph,shorten_ids=False))
            evidence=row["metadata"].get("space_import") or row["metadata"].get("archive_recovery")
            if row["metadata"].get("space_import"):
                check_content=gates
            else:
                from prepare_space_v11 import gates as check_content
            residual=check_content(row["response_text"],row["completion"],checked,evidence.get("original_source_sha256",""),(row["metadata"].get("url_preprocessing") or {}).get("url_map"))
            if residual: raise ValueError(f"Residual sampled content issues: {row['id']}:{residual}")
            sample_count+=1
    checks={"original_v10_hashes_unchanged":True,"all_output_hashes_and_counts":True,"all_row_graph_message_reference_and_identity_checks":sum(counts["v11"].values()),"all_new_changes_match_source_proven_repair_replay":True,"screened_base_bytes_preserved_in_merge":True,"v11s_splits_equal_v11_new_suffixes":True,"unresolved_v10_hold_families_admitted":0,"duplicate_exact_pairs":0,**overlaps,"new_vs_v10_normalized_sources":0,"benchmark_exact_normalized_or_detected_lexical_overlaps":0,"detected_train_val_lexical_near_overlaps":0,"strict_production_validator_and_content_sample_rows":sample_count,"sample_residual_issues":0,"lexical_search_scope":"same bounded rare-trigram retrieval and thresholds as v10; not exhaustive semantic equivalence","tokenizer_model_training_device_checks":"not_run","new_repaired_rows_independently_replayed":replay_repaired,"generator_review_rows_independently_recovered":replay_recovered}
    checks["original_stage3_semantic_provenance_rows_replayed"]=evidence_checked
    checks["known_v10_repairs_independently_replayed"]=base_replays
    allstats={"v10":base["statistics"],**{name:{s:p.result() for s,p in ss.items()} for name,ss in stats.items()}}
    report={"status":"verified","checks":checks,"statistics":allstats,"elapsed_seconds":round(time.monotonic()-start,2),"verification_script_sha256":file_sha(Path(__file__))}
    dump(args.audit_dir/"statistics.json",allstats)
    dump(args.audit_dir/"verification.json",{k:v for k,v in report.items() if k!="statistics"})
    (args.audit_dir/"REPORT.html").write_text(report_html(report,cleaning,archive),encoding="utf-8")
    for name in ("v11","v11s"):
        manifest=manifests[name]
        manifest["status"]="offline_verified_candidate"
        manifest["verification_sha256"]=file_sha(args.audit_dir/"verification.json")
        manifest["statistics_sha256"]=file_sha(args.audit_dir/"statistics.json")
        dump(args.output_root/name/"manifest.json",manifest)
    print(dumps({"status":"verified","rows":{n:{s:x["rows"] for s,x in splits.items()} for n,splits in allstats.items()},"checks":checks}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("policy-repo","work-dir","project-dir","output-root","audit-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=8)
    verify(parser.parse_args())
