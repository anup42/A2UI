#!/usr/bin/env python3
"""Build v11/v11s from completed Space records using the v10 review policy.

Run `base` while the archive downloads, then `new` after projection. Semantic
repairs are quarantined, never applied. Source/target URL masking is allowed
only with an exact inverse. All destinations are fresh; original v10 is frozen.
The policy repository is explicit because its newer archive gates are not
present in every GenUI-LM checkout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
POLICY = "space-v11-v10-gates-no-semantic-rewrite-20260930"
CONFIG = {}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def lines(path):
    with Path(path).open("rb") as stream:
        for n, raw in enumerate(stream, 1):
            if not raw.strip():
                raise ValueError(f"Blank JSONL record: {path}:{n}")
            yield n, raw, json.loads(raw)


def init(policy_repo, config=None):
    global CONFIG
    CONFIG = config or {}
    root = Path(policy_repo)
    for rel in ("training/src", "dataset/src", "training/scripts/audits"):
        sys.path.insert(0, str(root / rel))
    from full_data_ir_20260913 import wire_setup
    wire_setup(False)


def policy_files(repo):
    root = Path(repo)
    folders = [root / "training/src/ir_training/data", root / "dataset/src/pipeline/ir_formats"]
    paths = [p for folder in folders for p in folder.rglob("*.py")]
    paths += list((root / "dataset/schema").glob("*.json"))
    paths += [root / "training/scripts/audits/full_data_ir_20260913.py", root / "training/data/quality/v9_manual100_findings.json"]
    return {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(paths)}


def percentile(values, p):
    return values[min(len(values)-1, max(0, math.ceil(p * len(values))-1))] if values else None


class Stats:
    def __init__(self):
        self.rows = 0
        self.counts = defaultdict(Counter)
        self.lengths = defaultdict(list)
        self.sources, self.normalized, self.families, self.pairs, self.semantic_pairs = set(), set(), set(), set(), set()

    def add(self, item):
        self.rows += 1
        for key in ("components", "component_presence", "table_domains", "reference_kinds"):
            self.counts[key].update(item[key])
        for key in ("origin", "intent", "difficulty", "target_format", "source_review_status", "teacher", "modality"):
            self.counts[key][str(item.get(key, "unknown"))] += 1
        for key, value in item["lengths"].items():
            if value is not None:
                self.lengths[key].append(value)
        for key, value in item["flags"].items():
            if value:
                self.counts["flags"][key] += 1
        self.sources.add(item["source_sha"])
        self.normalized.add(item["normalized_sha"])
        self.families.add(item["family"])
        self.pairs.add(item["pair_sha"])
        self.semantic_pairs.add((item["normalized_sha"], item["semantic_sha"]))

    def result(self):
        sizes = {}
        for name, values in self.lengths.items():
            values = sorted(values)
            sizes[name] = {"count":len(values), "total":sum(values), "min":values[0], "mean":round(sum(values)/len(values), 2), "p50":percentile(values,.5), "p90":percentile(values,.9), "p95":percentile(values,.95), "p99":percentile(values,.99), "max":values[-1]}
        return {"rows":self.rows, "unique_source_texts":len(self.sources), "unique_normalized_sources":len(self.normalized), "source_families":len(self.families), "unique_exact_pairs":len(self.pairs), "unique_normalized_source_semantic_pairs":len(self.semantic_pairs), "distributions":{k:dict(sorted(v.items())) for k,v in self.counts.items()}, "lengths":sizes}


def describe(row, checked, origin):
    from ir_training.data.archive_refinement import source_signature
    from ir_training.data.archive_recovery import placeholder_tokens
    graph = checked.graph
    source, target = row["response_text"], row["completion"]
    components = Counter(e["type"] for e in graph["elements"].values())
    meta = row.get("metadata") or {}
    space = meta.get("space_import") or {}
    gen = space.get("gen") or {}
    query = meta.get("query_quality") or {}
    quality = meta.get("source_quality") or {}
    signature = source_signature(source)
    return {"source_sha":sha(source), "normalized_sha":sha(signature), "signature":signature,
            "family":row["source_id"], "semantic_sha":checked.semantic_sha256, "pair_sha":sha(source+"\0"+target),
            "components":dict(components), "component_presence":{k:1 for k in components},
            "table_domains":dict(Counter(str(e.get("props",{}).get("domain","generic")) for e in graph["elements"].values() if e["type"] == "Table")),
            "reference_kinds":dict(Counter(checked.reference_kinds)), "origin":origin,
            "intent":meta.get("intent") or row.get("intent_bucket") or "unknown_archive",
            "difficulty":meta.get("difficulty") or "unknown", "target_format":row.get("target_format", "a2ui_express_v1"),
            "source_review_status":quality.get("status", "unknown_archive"), "teacher":gen.get("model","unknown_archive"),
            "modality":query.get("modality","unknown_archive"),
            "lengths":{"source_chars":len(source), "source_words":len(source.split()), "target_chars":len(target), "target_lines":len(target.splitlines()), "component_count":len(graph["elements"]), "generation_input_tokens":gen.get("input_tokens"), "generation_output_tokens":gen.get("output_tokens"), "generation_quality_0_100":space.get("quality_metrics",{}).get("genui_quality_v5_4")},
            "flags":{"with_original_query":bool(meta.get("query_text")), "with_source_placeholders":bool(placeholder_tokens(source)), "with_url_map":bool((meta.get("url_preprocessing") or {}).get("url_map")), "historically_repaired":bool(row.get("repair",{}).get("applied")), "mixed_reference_normalization":bool(space.get("mixed_reference_normalization")), "with_replacement_character": "\ufffd" in source or "\ufffd" in target}}


def gates(source, target, checked, original_sha="", url_map=None):
    from ir_training.data.archive_semantic_review import process_graph
    from ir_training.data.archive_refinement import review_warnings
    from ir_training.data.archive_final_review import paragraph_gaps
    from ir_training.data.archive_letter_review import letter_gaps
    from ir_training.data.archive_recovery import placeholder_tokens
    result = process_graph(source, checked.graph, original_source_sha256=original_sha, url_map=url_map)
    reasons = {x["code"] for x in result["issues"]}
    # Do not apply v10's historical content rewrites to newly generated IR.
    reasons.update("requires_regeneration:"+kind for kind in result["changes"])
    warnings, _, _ = review_warnings(source, target, checked.graph)
    reasons.update(warnings)
    if paragraph_gaps(source, checked.graph):
        reasons.add("legacy_paragraph_gap")
    if letter_gaps(source, checked.graph):
        reasons.add("legacy_letter_gap")
    if not placeholder_tokens(target) <= placeholder_tokens(source):
        reasons.add("target_reference_absent_from_source")
    return sorted(reasons)


def normalize_references(source, graph, mapping):
    """Resolve a mixed raw-source/masked-target representation, never content.

    In some Stage 3 records response_text has raw URLs while the completion
    uses reference_map tokens. Restore these exact saved identities before
    applying v10's source-first masking; target-only URLs still fail closure.
    Already closed placeholder pairs retain their original token identities.
    """
    from ir_training.data.reference_binding import reference_tokens
    from ir_training.data.url_preprocess import preprocess_training_urls,restore_url_placeholders,SOURCE_IDENTITY_BINDING
    extras = reference_tokens(graph)-reference_tokens(source)
    input_source,input_graph = source,graph
    if extras:
        if not extras <= set(mapping):
            raise ValueError("target_reference_absent_from_source")
        input_source = restore_url_placeholders(source,mapping)
        input_graph = restore_url_placeholders(graph,mapping)
        for token in extras:
            value=mapping[token]
            destination=value.get("url") if isinstance(value,dict) else value
            if destination not in input_source:
                raise ValueError("target_reference_absent_from_source")
    processed = preprocess_training_urls(input_source,input_graph,binding_policy=SOURCE_IDENTITY_BINDING)
    if restore_url_placeholders(processed.response_text,processed.url_map) != input_source or restore_url_placeholders(processed.canonical_graph,processed.url_map) != input_graph:
        raise ValueError("url_mask_not_lossless")
    if not reference_tokens(processed.canonical_graph) <= reference_tokens(processed.response_text):
        raise ValueError("target_reference_absent_from_source")
    return processed, bool(extras)


def base_item(item):
    from ir_training.data.express_preparation import serialize_checked, TASK_PREFIX
    split, number, raw = item
    row = json.loads(raw)
    checked = serialize_checked(row["completion"], "root-first")
    if row["messages"][-2] != {"role":"user", "content":TASK_PREFIX+row["response_text"]} or row["messages"][-1] != {"role":"assistant", "content":row["completion"]}:
        raise ValueError(f"v10 message mismatch: {split}:{number}")
    meta = row.get("metadata") or {}
    recovery = meta.get("archive_recovery") or {}
    for field, value in (("effective_source_sha256",sha(row["response_text"])),("effective_target_sha256",sha(row["completion"])),("effective_semantic_sha256",checked.semantic_sha256)):
        if recovery.get(field) != value:
            raise ValueError(f"v10 identity mismatch: {split}:{number}:{field}")
    stats = describe(row, checked, "v10")
    reasons = gates(row["response_text"], row["completion"], checked, recovery.get("original_source_sha256",""), (meta.get("url_preprocessing") or {}).get("url_map"))
    if row["source_id"] in CONFIG["manual_hold_families"]:
        reasons.append("known_v10_manual_review_family")
    return split, number, raw, row["id"], stats, sorted(set(reasons))


def bounded_map(fn, items, repo, workers, config):
    from ir_training.common.parallel import ordered_bounded_map
    return ordered_bounded_map(fn, items, workers=workers, initializer=init, initargs=(str(repo),config), batch_size=32)


def base(args):
    init(args.policy_repo)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = args.policy_repo / "training/data/train/v10/bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    for entry in bundle["files"]:
        if file_sha(args.base_dir / entry["name"]) != entry["sha256"]:
            raise ValueError(f"v10 split hash mismatch: {entry['name']}")
    review = args.policy_repo / "training/reports/v10_manual100_20260916"
    verdicts = {x[0]:x for x in json.loads((review/"manual_verdicts.json").read_text(encoding="utf-8"))}
    holds = {}
    for _,_,sample in lines(review/"samples.jsonl"):
        verdict = verdicts[sample["sample"]]
        if verdict[1] != "KEEP":
            holds[sample["source_id"]] = {"case":sample["sample"], "verdict":verdict[1], "reason":verdict[3], "source_sha256":sha(sample["source"])}
    dump(args.work_dir/"known_v10_holds.json", holds)
    first = next(lines(args.base_dir/"train.jsonl"))[2]
    dump(args.work_dir/"scaffold.json", {"messages":first["messages"][:-2], "shared_prompt":first["metadata"]["shared_prompt"]})
    stats = {s:Stats() for s in ("train","val","combined")}
    retained = Counter()
    reasons = Counter()
    input_rows = Counter()
    streams = {s:(args.work_dir/f"{s}.jsonl").open("wb") for s in ("train","val")}
    start = time.monotonic()
    def items():
        for split in ("train","val"):
            with (args.base_dir/f"{split}.jsonl").open("rb") as f:
                for n,raw in enumerate(f,1):
                    yield split,n,raw
    with (args.work_dir/"base_index.jsonl").open("w",encoding="utf-8") as index, (args.work_dir/"decisions.jsonl").open("w",encoding="utf-8") as decisions:
        for split,n,raw,rid,profile,flags in bounded_map(base_item,items(),args.policy_repo,args.workers,{"manual_hold_families":holds}):
            input_rows[split] += 1
            stats[split].add(profile)
            stats["combined"].add(profile)
            info = {k:profile[k] for k in ("signature","normalized_sha","source_sha","family","semantic_sha","pair_sha")}
            index.write(dumps({**info,"id":rid,"split":split,"line":n,"excluded":bool(flags)})+"\n")
            decisions.write(dumps({"id":rid,"split":split,"line":n,"family":profile["family"],"outcome":"QUARANTINE" if flags else "KEEP","reasons":flags})+"\n")
            if flags:
                reasons.update(flags)
            else:
                streams[split].write(raw)
                retained[split] += 1
            if sum(input_rows.values()) % 2000 == 0:
                print(f"v10 reviewed {sum(input_rows.values()):,}; retained {sum(retained.values()):,}; elapsed {time.monotonic()-start:.0f}s",flush=True)
    for stream in streams.values(): stream.close()
    for entry in bundle["files"]:
        if input_rows[Path(entry["name"]).stem] != entry["rows"] or file_sha(args.base_dir/entry["name"]) != entry["sha256"]:
            raise ValueError("Original v10 changed or row counts mismatch")
    report = {"status":"complete","policy":POLICY,"base_dir":str(args.base_dir.resolve()),"bundle_sha256":file_sha(bundle_path),"input_split_sha256":{x["name"]:x["sha256"] for x in bundle["files"]},"input_rows":dict(input_rows),"retained_rows":dict(retained),"quarantined_rows":{s:input_rows[s]-retained[s] for s in input_rows},"reasons_overlap":dict(reasons),"known_manual_hold_families":len(holds),"statistics":{s:p.result() for s,p in stats.items()},"implementation_sha256":policy_files(args.policy_repo),"elapsed_seconds":round(time.monotonic()-start,2)}
    dump(args.work_dir/"base_report.json",report)
    print(dumps({k:v for k,v in report.items() if k not in {"statistics","implementation_sha256"}}),flush=True)


def new_item(item):
    from ir_training.data.express_preparation import serialize_checked, PreparationError, TASK_PREFIX, _api
    from ir_training.data.reference_binding import reference_tokens, _reference_map
    from ir_training.data.url_preprocess import preprocess_training_urls, restore_url_placeholders, SOURCE_IDENTITY_BINDING
    run,row = item
    event = {"run":run,"ui_id":row.get("ui_id"),"response_id":row.get("response_id"),"archive_record":row["archive_record"],"reasons":[]}
    reasons = set()
    if row.get("import_error"): reasons.add("invalid_json")
    if row.get("record_status") != "accepted": reasons.add("generation_status:"+str(row.get("record_status","missing")))
    accept = row.get("training_acceptance") or {}
    if accept.get("eligible") is not True:
        reasons.add("generator_training_ineligible")
        reasons.update("generator:"+str(x) for x in accept.get("blocking_reasons",[])+accept.get("review_reasons",[]))
    gen = row.get("gen") or {}
    if gen.get("error") or gen.get("finish_reason") in {"length","max_tokens"} or gen.get("completion_complete") is False:
        reasons.add("incomplete_generation")
    quality = row.get("source_quality") or {}
    rejected = {"reject","rejected","exclude","excluded","ineligible","blocked","failed","invalid"}
    if quality.get("training_eligibility") in rejected or quality.get("status") in rejected:
        reasons.add("source_quality_blocked")
    if (row.get("query_quality") or {}).get("training_eligibility") in rejected:
        reasons.add("query_quality_blocked")
    for finding in quality.get("findings",[]):
        if finding.get("severity") in {"error","block","blocker","blocking","reject"}:
            reasons.add("source_quality:"+str(finding.get("code")))
    if row.get("join_errors"): reasons.update(row["join_errors"])
    if reasons:
        event["reasons"] = sorted(reasons)
        return None,event,None
    try:
        source, target = row.get("response_text"), row.get("completion")
        if not isinstance(source,str) or not source.strip() or not isinstance(target,str) or not target.strip():
            raise ValueError("empty_source_or_target")
        if row.get("source_format") != "a2ui_express_v1" or row.get("target_format") != "a2ui_express_v1":
            raise ValueError("wrong_target_format")
        if row.get("a2ui_express",target) != target:
            raise ValueError("target_alias_mismatch")
        if row.get("reference_source_sha256") != sha(source):
            raise ValueError("reference_source_hash_mismatch")
        checked = serialize_checked(target,"root-first")
        active, express, semantic_hash, _ = _api()
        mapping = _reference_map(row)
        if reference_tokens(source) - set(mapping):
            raise ValueError("incomplete_source_reference_map")
        if (reference_tokens(checked.graph)-reference_tokens(source))-set(mapping):
            raise ValueError("target_reference_absent_from_source")
        original_graph = checked.graph
        stored_graph = row.get("canonical_graph")
        restored_graph = restore_url_placeholders(original_graph,mapping)
        if stored_graph is not None and stored_graph not in (original_graph,restored_graph):
            # Canonicalization handles representational differences, not content.
            from pipeline.ir_formats.canonical import canonical_graph
            if canonical_graph(stored_graph) not in (original_graph,restored_graph):
                raise ValueError("canonical_graph_mismatch")
        allowed_hashes = {semantic_hash(original_graph),semantic_hash(restored_graph)}
        if row.get("canonical_graph_hash") and row["canonical_graph_hash"] not in allowed_hashes:
            raise ValueError("canonical_graph_hash_mismatch")
        if row.get("semantic_hash") and row["semantic_hash"] not in allowed_hashes:
            raise ValueError("semantic_hash_mismatch")
        stage2 = row["stage2_source"]
        if source != stage2 and restore_url_placeholders(source,mapping) != stage2:
            raise ValueError("stage2_source_mismatch")
        from pipeline.source_quality import assess_source_quality
        source_recheck = assess_source_quality(row["query"],stage2)
        if source_recheck["status"] == "failed":
            event["reasons"] = sorted({"source_recheck:"+x["code"] for x in source_recheck["findings"] if x["severity"] == "error"})
            return None,event,None
        processed, mixed_reference_normalization = normalize_references(source,original_graph,mapping)
        source = processed.response_text
        checked = serialize_checked(express.encode(processed.canonical_graph,shorten_ids=False),"root-first")
        target = checked.text
        url_map = processed.url_map if mixed_reference_normalization else {**mapping,**processed.url_map}
        reasons.update(gates(source,target,checked,sha(row["response_text"]),url_map))
        if reasons:
            event["reasons"] = sorted(reasons)
            return None,event,None
        rid = f"{run}:{row['ui_id']}:{row['archive_record']['line']}"
        family = "space-source-sha256:"+sha(source)
        metadata = {"source_id":family,"query_id":run+":"+str(row.get("query_id")),"response_id":run+":"+str(row.get("response_id")),"ui_id":rid,
                    "intent":row.get("intent") or row.get("query",{}).get("intent"),"intent_bucket":row.get("intent_bucket","unknown"),
                    "query_text":row["query"]["query_text"],"difficulty":row["query"].get("difficulty"),"tags":row["query"].get("tags",[]),
                    "query_quality":row.get("query_quality") or {},"source_quality":quality,"scenario_family_id":row.get("scenario_family_id"),
                    "source_format":"a2ui_express_v1","target_format":"a2ui_express_v1","shared_prompt":CONFIG["scaffold"]["shared_prompt"],
                    "url_preprocessing":{"enabled":bool(processed.url_map),"binding_policy":"source_identity","source_closed":True,"url_map":url_map},
                    "space_import":{"policy":POLICY,"run":run,"archive_record":row["archive_record"],"original_source_sha256":sha(row["response_text"]),"original_target_sha256":sha(row["completion"]),"effective_source_sha256":sha(source),"effective_target_sha256":sha(target),"effective_semantic_sha256":checked.semantic_sha256,"gen":gen,"quality_metrics":row.get("quality_metrics",{}),"training_acceptance":accept,"source_content_changed":False,"target_semantics_changed":False}}
        output = {"id":rid,"row_id":rid,"response_id":metadata["response_id"],"source_id":family,"source_format":"a2ui_express_v1","target_format":"a2ui_express_v1",
                  "response_text":source,"completion":target,"messages":deepcopy(CONFIG["scaffold"]["messages"])+[{"role":"user","content":TASK_PREFIX+source},{"role":"assistant","content":target}],
                  "intent_bucket":metadata["intent_bucket"],"repair":{"applied":False,"changes":[]},"metadata":metadata}
        metadata["space_import"]["source_quality_recheck"] = source_recheck
        metadata["space_import"]["mixed_reference_normalization"] = mixed_reference_normalization
        metadata["space_import"]["created_at"] = row.get("created_at")
        metadata["space_import"]["phase_invocation_id"] = row.get("phase_invocation_id")
        profile = describe(output,checked,run)
        event.update({"source_sha":profile["source_sha"],"normalized_sha":profile["normalized_sha"],"semantic_sha":profile["semantic_sha"]})
        return output,event,profile
    except (ValueError,TypeError,KeyError,UnicodeError) as exc:
        event["reasons"] = [getattr(exc,"reason",str(exc).split(":")[0])]
        event["detail"] = str(exc)[:600]
        return None,event,None


def joined_rows(project, inventory):
    for directory in sorted(project.glob("dataset_muse_glimmer_100k_r*")):
        counts = Counter()
        stage3_ids,stage3_responses,eligible_responses = set(),set(),set()
        queries,responses = {},{}
        for name,key,store in (("queries.jsonl","query_id",queries),("responses.jsonl","response_id",responses)):
            path = directory/name
            if path.exists():
                for _,_,row in lines(path):
                    counts[name] += 1
                    if name == "queries.jsonl":
                        counts["query_intent:"+str(row.get("intent","unknown"))] += 1
                        counts["query_difficulty:"+str(row.get("difficulty","unknown"))] += 1
                        counts["query_modality:"+str((row.get("query_quality") or {}).get("modality","unknown"))] += 1
                    else:
                        counts["response_source_quality:"+str((row.get("source_quality") or {}).get("status","unknown"))] += 1
                    identity = row[key]
                    if identity in store and store[identity] != row:
                        raise ValueError(f"Conflicting {key} in {directory.name}: {identity}")
                    store[identity] = row
                counts["unique_"+key] = len(store)
        path = directory/"genui.training.jsonl"
        if path.exists():
            for _,_,row in lines(path):
                counts["genui_records"] += 1
                if row.get("ui_id"): stage3_ids.add(row["ui_id"])
                if row.get("response_id"):
                    stage3_responses.add(row["response_id"])
                    if (row.get("training_acceptance") or {}).get("eligible") is True:
                        eligible_responses.add(row["response_id"])
                counts["status:"+str(row.get("record_status","missing"))] += 1
                counts["eligible:"+str((row.get("training_acceptance") or {}).get("eligible","missing"))] += 1
                counts["source_quality:"+str((row.get("source_quality") or {}).get("status","missing"))] += 1
                counts["stage3_teacher:"+str((row.get("gen") or {}).get("model","unknown"))] += 1
                counts["stage3_prompt:"+str((row.get("gen") or {}).get("prompt_version","unknown"))] += 1
                counts["stage3_created_date:"+str(row.get("created_at","unknown")).split("T")[0]] += 1
                query = queries.get(row.get("query_id"))
                response = responses.get(row.get("response_id"))
                errors = []
                if not query: errors.append("missing_stage1_query")
                if not response: errors.append("missing_stage2_response")
                if response and response.get("query_id") != row.get("query_id"): errors.append("response_query_mismatch")
                if response and query and response.get("query_text") and response["query_text"] != query.get("query_text"):
                    errors.append("stage2_query_text_mismatch")
                query_hash = (row.get("source_quality") or {}).get("original_query_sha256")
                if query and query_hash and query_hash != sha(query.get("query_text","")):
                    errors.append("original_query_hash_mismatch")
                row["query"] = query
                row["stage2_source"] = response.get("response_text") if response else None
                row["join_errors"] = errors
                yield directory.name,row
        counts["unique_stage3_ui_ids"] = len(stage3_ids)
        counts["unique_stage3_response_ids"] = len(stage3_responses)
        counts["stage2_responses_without_stage3_record"] = len(set(responses)-stage3_responses)
        counts["stage2_responses_without_generator_eligible_target"] = len(set(responses)-eligible_responses)
        inventory[directory.name] = dict(counts)


def goldens(repo):
    from ir_training.data.archive_refinement import source_signature
    sources,hashes = {},{}
    for name,folder in (("golden32","golden32_archive_repeat_v1"),("golden35","golden35_v1"),("bixby50","bixby50_v1")):
        path = repo/"training/data/eval"/folder/f"{name}.jsonl"
        hashes[str(path)] = file_sha(path)
        for n,_,row in lines(path):
            source = row.get("response_text")
            if not source:
                from ir_training.data.golden_replacement import response_text
                source = response_text(row)
            sources[f"{name}:{n}"] = source_signature(source)
    return sources,hashes


def new(args):
    init(args.policy_repo)
    from ir_training.data.archive_refinement import NearSourceIndex,FamilyUnion,choose_validation,component_shape
    projection = json.loads((args.project_dir/"archive_manifest.json").read_text(encoding="utf-8"))
    if projection.get("status") != "verified" or not projection.get("gzip_crc_verified"):
        raise ValueError("Requires a completed, verified archive projection")
    base_report = json.loads((args.work_dir/"base_report.json").read_text())
    if base_report["implementation_sha256"] != policy_files(args.policy_repo):
        raise ValueError("Policy changed after v10 screening")
    args.audit_dir.mkdir(parents=True,exist_ok=False)
    v11s = args.output_root/"v11s.partial"
    v11 = args.output_root/"v11.partial"
    for path in (v11s,v11):
        if path.with_name(path.name.replace(".partial","" )).exists(): raise FileExistsError(path)
        path.mkdir(parents=True,exist_ok=False)
    scaffold = json.loads((args.work_dir/"scaffold.json").read_text())
    base_sources = {}
    for _,_,row in lines(args.work_dir/"base_index.jsonl"):
        base_sources[row["normalized_sha"]] = row["signature"]
    print(f"Indexing {len(base_sources):,} v10 normalized sources for exact/near exclusion",flush=True)
    base_near = NearSourceIndex(base_sources)
    golden_sources,golden_hashes = goldens(args.policy_repo)
    golden_near = NearSourceIndex(golden_sources)
    golden_exact = {sha(v) for v in golden_sources.values()}
    inventory,counts,reasons = {},Counter(),Counter()
    new_sources,profiles,scenario_keys,query_keys = {},{},defaultdict(set),defaultdict(set)
    families = FamilyUnion()
    seen_pairs,seen_semantics = set(),set()
    spool = args.audit_dir/"candidates.jsonl"
    start = time.monotonic()
    with spool.open("w",encoding="utf-8") as accepted, (args.audit_dir/"new_decisions.jsonl").open("w",encoding="utf-8") as decisions:
        for row,event,profile in bounded_map(new_item,joined_rows(args.project_dir,inventory),args.policy_repo,args.workers,{"scaffold":scaffold}):
            counts["input"] += 1
            if row is not None:
                sig,ns = profile["signature"],profile["normalized_sha"]
                if ns in golden_exact or list(golden_near.matches(sig,containment=True)):
                    event["reasons"].append("reserved_benchmark_source")
                elif ns in base_sources:
                    event["reasons"].append("v10_normalized_source_overlap")
                elif list(base_near.matches(sig)):
                    event["reasons"].append("v10_lexical_near_source_overlap")
                elif profile["pair_sha"] in seen_pairs or (ns,profile["semantic_sha"]) in seen_semantics:
                    event["reasons"].append("duplicate_source_target_pair")
                else:
                    seen_pairs.add(profile["pair_sha"])
                    seen_semantics.add((ns,profile["semantic_sha"]))
                    families.find(ns)
                    new_sources[ns] = sig
                    profiles.setdefault(ns,profile)
                    scenario = row["metadata"].get("scenario_family_id")
                    if scenario: scenario_keys[scenario].add(ns)
                    from ir_training.data.archive_refinement import source_signature
                    query_keys[sha(source_signature(row["metadata"]["query_text"]))].add(ns)
                    row["metadata"]["space_import"]["normalized_source_sha256"] = ns
                    accepted.write(dumps(row)+"\n")
                    counts["candidate"] += 1
            event["outcome"] = "QUARANTINE" if event["reasons"] else "KEEP"
            counts[event["outcome"]] += 1
            reasons.update(event["reasons"])
            decisions.write(dumps(event)+"\n")
            if counts["input"] % 1000 == 0:
                print(f"Space reviewed {counts['input']:,}; retained {counts['candidate']:,}; elapsed {time.monotonic()-start:.0f}s",flush=True)
    del base_near
    print(f"Grouping {len(new_sources):,} new source signatures before the 98/2 split",flush=True)
    for groups in (scenario_keys,query_keys):
        for members in groups.values():
            members = sorted(members)
            for member in members[1:]: families.union(members[0],member)
    near = NearSourceIndex(new_sources)
    near_edges = 0
    for index,(key,sig) in enumerate(new_sources.items(),1):
        for other,jaccard,containment in near.matches(sig):
            if other != key and families.union(key,other): near_edges += 1
        if index % 2000 == 0: print(f"Source family grouping {index:,}/{len(new_sources):,}",flush=True)
    groups = {}
    for key,p in profiles.items():
        group = families.find(key)
        if group not in groups: groups[group] = (component_shape(p["components"]),p["lengths"]["source_chars"])
    validation = choose_validation(groups,.02,args.seed)
    if not validation or len(validation) == len(groups): raise ValueError("Empty split")
    split_counts = Counter()
    new_stats = {s:Stats() for s in ("train","val","combined")}
    outputs = {s:(v11s/f"{s}.jsonl").open("w",encoding="utf-8",newline="\n") for s in ("train","val")}
    with (args.audit_dir/"new_family_assignments.jsonl").open("w",encoding="utf-8") as assignments:
        for key in sorted(new_sources):
            group = families.find(key)
            assignments.write(dumps({"normalized_source_sha256":key,"family":group,"split":"val" if group in validation else "train"})+"\n")
    from ir_training.data.express_preparation import serialize_checked
    for _,_,row in lines(spool):
        ns = row["metadata"]["space_import"]["normalized_source_sha256"]
        group = families.find(ns)
        split = "val" if group in validation else "train"
        row["source_id"] = "space-source-family-sha256:"+group
        row["metadata"]["source_id"] = row["source_id"]
        row["metadata"]["assigned_split"] = split
        checked = serialize_checked(row["completion"],"root-first")
        profile = describe(row,checked,row["metadata"]["space_import"]["run"])
        new_stats[split].add(profile);new_stats["combined"].add(profile)
        outputs[split].write(dumps(row)+"\n")
        split_counts[split] += 1
    for stream in outputs.values(): stream.close()
    for split in ("train","val"):
        with (v11/f"{split}.jsonl").open("xb") as output:
            for source in (args.work_dir/f"{split}.jsonl",v11s/f"{split}.jsonl"):
                with source.open("rb") as stream: shutil.copyfileobj(stream,output,1024*1024)
    common = {"policy":POLICY,"seed":args.seed,"source_family_validation_target_fraction":.02,"base_input_hashes":base_report["input_split_sha256"],"base_filtering":{"retained":base_report["retained_rows"],"quarantined":base_report["quarantined_rows"],"reasons_overlap":base_report["reasons_overlap"]},"new_input_inventory":inventory,"new_cleaning_counts":dict(counts),"new_quarantine_reasons_overlap":dict(reasons),"new_source_groups":len(groups),"new_lexical_near_union_edges":near_edges,"benchmark_sha256":golden_hashes,"archive_manifest_sha256":file_sha(args.project_dir/"archive_manifest.json"),"implementation_sha256":base_report["implementation_sha256"],"build_script_sha256":file_sha(Path(__file__)),"format":"a2ui_express_v1","tokenizer_preparation":"deferred_to_training_model; no records truncated","semantic_repairs_applied":0,"validation_usage":"v11s/val is an unchanged subset of v11/val; never use v11s/train for an unseen evaluation of a v11-trained model"}
    common["source_quality_recheck_implementation_sha256"] = file_sha(args.policy_repo/"dataset/src/pipeline/source_quality.py")
    history_path = args.policy_repo/"training/reports/offline_recovery_20260916_v10/summary.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    common["v10_creation_history"] = {"evidence_path":str(history_path),"evidence_sha256":file_sha(history_path),"policy":history["policy_version"],
        "original_archive_records":sum(history["categories"]["combined"].values()),"v9_input_rows":history["semantic_review"]["input_rows"],
        "v9_to_v10_outcomes":history["semantic_review"]["outcomes_from_v9"],"v10_output_rows":history["output_rows"],
        "historical_repairs_overlap":history["semantic_review"]["retained_repair_rows_by_kind_overlap"]}
    for name,path,rows in (("v11s",v11s,dict(split_counts)),("v11",v11,{s:split_counts[s]+base_report["retained_rows"][s] for s in ("train","val")})):
        manifest = {**common,"dataset":name,"status":"built_pending_independent_verification","rows":rows,"outputs":{s+".jsonl":{"sha256":file_sha(path/f"{s}.jsonl"),"bytes":(path/f"{s}.jsonl").stat().st_size} for s in ("train","val")}}
        dump(path/"manifest.json",manifest)
        dump(path/"prompt_scaffold.json",scaffold)
        path.rename(path.with_name(name))
    dump(args.audit_dir/"v10_statistics.json",base_report["statistics"])
    dump(args.audit_dir/"v11s_statistics.json",{s:p.result() for s,p in new_stats.items()})
    dump(args.audit_dir/"cleaning_summary.json",common)
    shutil.copyfile(args.work_dir/"decisions.jsonl",args.audit_dir/"v10_decisions.jsonl")
    shutil.copyfile(args.work_dir/"known_v10_holds.json",args.audit_dir/"known_v10_holds.json")
    print(dumps({"v11s":dict(split_counts),"v11":{s:split_counts[s]+base_report["retained_rows"][s] for s in ("train","val")},"new_counts":dict(counts),"reasons":dict(reasons)}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["base","new"])
    parser.add_argument("--policy-repo",type=Path,required=True)
    parser.add_argument("--work-dir",type=Path,required=True)
    parser.add_argument("--base-dir",type=Path)
    parser.add_argument("--project-dir",type=Path)
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--audit-dir",type=Path)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--seed",type=int,default=20260930)
    args = parser.parse_args()
    (base if args.stage == "base" else new)(args)


if __name__ == "__main__":
    main()
