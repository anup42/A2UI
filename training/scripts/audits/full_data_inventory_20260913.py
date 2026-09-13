"""Read-only, streaming inventory of supplied training rows and repository provenance.

Writes only audit artifacts, never changes training or benchmark inputs. Source text
is extracted from the final task turn, not the fixed in-context example.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "training/src"))
from ir_training.data.express_preparation import TASK_PREFIX
from ir_training.data.golden_replacement import response_text
from ir_training.data.url_preprocess import preprocess_training_urls


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.split())


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_hashes(source: str) -> tuple[str, str, str]:
    return sha(source), sha(normalized(source)), sha(normalized(preprocess_training_urls(source, {}, enabled=True).response_text))


def physical_lines(path: Path):
    """Yield true byte offsets and physical lines for UTF8 or BOM-tagged UTF16."""
    with path.open("rb") as stream:
        prefix = stream.read(2)
        stream.seek(0)
        if prefix == b"\xff\xfe":
            encoding, delimiter = "utf-16-le", b"\n\x00"
        elif prefix == b"\xfe\xff":
            encoding, delimiter = "utf-16-be", b"\x00\n"
        else:
            encoding, delimiter = "utf-8", b"\n"
        buffer, offset = b"", 0
        while chunk := stream.read(4*1024*1024):
            buffer += chunk
            pieces = buffer.split(delimiter)
            buffer = pieces.pop()
            for piece in pieces:
                raw = piece + delimiter
                yield offset, raw, encoding
                offset += len(raw)
        if buffer:
            yield offset, buffer, encoding


def index_inputs(source_dir: Path, db: sqlite3.Connection) -> dict:
    db.executescript("""
        CREATE TABLE rows (
          split TEXT NOT NULL, line INTEGER NOT NULL, byte_offset INTEGER NOT NULL,
          byte_length INTEGER NOT NULL, row_sha256 TEXT, canonical_row_sha256 TEXT,
          source_sha256 TEXT, source_normalized_sha256 TEXT, source_masked_sha256 TEXT,
          target_sha256 TEXT, scaffold_sha256 TEXT, source_chars INTEGER,
          target_chars INTEGER, prompt_chars INTEGER, message_count INTEGER,
          roles TEXT, schema TEXT, error TEXT, source_mojibake INTEGER,
          target_mojibake INTEGER, target_extra_placeholders INTEGER,
          target_missing_placeholders INTEGER, PRIMARY KEY(split,line));
        CREATE TABLE provenance (
          source_sha256 TEXT, source_normalized_sha256 TEXT, source_masked_sha256 TEXT,
          path TEXT, line INTEGER, query_id TEXT, response_id TEXT, ui_id TEXT,
          intent TEXT, model TEXT, provider TEXT, stage TEXT);
        CREATE TABLE golden_hashes (
          benchmark TEXT, membership TEXT, case_id TEXT, hash_kind TEXT, digest TEXT);
    """)
    files = {}
    for split in ("train", "val"):
        path = source_dir / f"{split}.jsonl"
        digest = hashlib.sha256()
        size_before = path.stat().st_size
        counts = Counter()
        schemas = Counter()
        roles_count = Counter()
        begin = time.monotonic()
        line = 0
        for offset, raw, encoding in physical_lines(path):
                line += 1
                digest.update(raw)
                values = {"split": split, "line": line, "byte_offset": offset,
                          "byte_length": len(raw)}
                try:
                    text = raw.decode(encoding).removeprefix("\ufeff")
                    values["row_sha256"] = sha(text.rstrip("\r\n"))
                    if not text.strip():
                        raise ValueError("blank_line")
                    row = json.loads(text)
                    if not isinstance(row, dict):
                        raise ValueError("non_object")
                    values["canonical_row_sha256"] = sha(dumps(row))
                    values["schema"] = ",".join(sorted(row))
                    schemas[values["schema"]] += 1
                    messages = row.get("messages")
                    if not isinstance(messages, list) or len(messages) < 2:
                        raise ValueError("invalid_messages")
                    if not all(isinstance(message, dict) and isinstance(message.get("content"), str) for message in messages):
                        raise ValueError("non_text_message")
                    values["message_count"] = len(messages)
                    values["roles"] = ",".join(str(message.get("role")) for message in messages)
                    roles_count[values["roles"]] += 1
                    if messages[-2].get("role") != "user" or messages[-1].get("role") != "assistant":
                        raise ValueError("invalid_final_roles")
                    task = messages[-2]["content"]
                    if not task.startswith(TASK_PREFIX):
                        raise ValueError("invalid_task_prefix")
                    source = task[len(TASK_PREFIX):]
                    target = messages[-1]["content"]
                    pattern = r"\[(?:IMAGE_URL|ICON_URL|ACTION_URL|SOURCE_URL|MEDIA_URL|URL|IMAGE_ASSET|ICON_ASSET|MEDIA_ASSET)_\d+\]"
                    source_tokens, target_tokens = set(re.findall(pattern, source)), set(re.findall(pattern, target))
                    suspicious = ("ΓÇ", "â€", "Ã", "\ufffd")
                    values.update(source_mojibake=int(any(token in source for token in suspicious)),
                                  target_mojibake=int(any(token in target for token in suspicious)),
                                  target_extra_placeholders=len(target_tokens-source_tokens),
                                  target_missing_placeholders=len(source_tokens-target_tokens))
                    for marker in suspicious:
                        counts[f"source_marker_{marker}_rows"] += int(marker in source)
                        counts[f"target_marker_{marker}_rows"] += int(marker in target)
                    try:
                        exact, norm, masked = source_hashes(source)
                    except ValueError as exc:
                        exact, norm, masked = sha(source), sha(normalized(source)), None
                        values["error"] = "url_preprocessing_error"
                        counts["url_preprocessing_error"] += 1
                        counts["url_preprocessing_error:" + str(exc)] += 1
                    values.update(source_sha256=exact, source_normalized_sha256=norm, source_masked_sha256=masked,
                                  target_sha256=sha(target), scaffold_sha256=sha(dumps(messages[:-2])),
                                  source_chars=len(source), target_chars=len(target),
                                  prompt_chars=sum(len(message["content"]) for message in messages[:-1]))
                    if not source.strip() or not target.strip():
                        raise ValueError("empty_source_or_target")
                    counts["valid_envelope_rows"] += 1
                except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                    category = type(exc).__name__ if isinstance(exc, (UnicodeError, json.JSONDecodeError)) else str(exc)
                    values["error"] = category
                    counts[category] += 1
                keys = list(values)
                db.execute(f"INSERT INTO rows ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})", list(values.values()))
                if line % 10000 == 0:
                    db.commit()
                    print(f"{split}: indexed {line:,} physical rows in {time.monotonic()-begin:.1f}s", flush=True)
        if path.stat().st_size != size_before:
            raise RuntimeError(f"Input size changed during audit: {path}")
        files[split] = {"path": str(path), "bytes": size_before, "sha256": digest.hexdigest(),
                        "encoding": encoding,
                        "physical_lines": line, "counts": dict(counts), "schemas": dict(schemas),
                        "role_sequences": dict(roles_count), "seconds": round(time.monotonic()-begin, 2)}
        db.commit()
    for column in ("source_sha256", "source_normalized_sha256", "source_masked_sha256", "target_sha256", "canonical_row_sha256"):
        db.execute(f"CREATE INDEX idx_{column} ON rows ({column})")
    db.commit()
    return files


def distribution(db, column, split=None):
    where = " WHERE " + column + " IS NOT NULL" + (" AND split=?" if split else "")
    vals = [row[0] for row in db.execute(f"SELECT {column} FROM rows{where} ORDER BY {column}", (split,) if split else ())]
    if not vals:
        return {}
    return {"count": len(vals), "min": vals[0], "mean": round(sum(vals)/len(vals), 2),
            **{f"p{p}": vals[min(len(vals)-1, round((len(vals)-1)*p/100))] for p in (25,50,75,90,95,99)}, "max": vals[-1]}


def complete_failed_url_mask_records(db, files):
    """Complete audit metadata from an older index that aborted on URL parsing.

    Only audit rows are updated; source files and targets remain unchanged.
    """
    candidates = db.execute("SELECT split,line,byte_offset,byte_length,error FROM rows WHERE error IS NOT NULL AND source_sha256 IS NULL AND message_count IS NOT NULL").fetchall()
    for split,line,offset,length,error in candidates:
        record = files[split]
        with Path(record["path"]).open("rb") as stream:
            stream.seek(offset)
            row = json.loads(stream.read(length).decode(record["encoding"]).removeprefix("\ufeff"))
        messages = row["messages"]
        if not messages[-2]["content"].startswith(TASK_PREFIX):
            continue
        source, target = messages[-2]["content"][len(TASK_PREFIX):], messages[-1]["content"]
        try:
            source_hashes(source)
        except ValueError:
            db.execute("""UPDATE rows SET source_sha256=?,source_normalized_sha256=?,target_sha256=?,
                       scaffold_sha256=?,source_chars=?,target_chars=?,prompt_chars=?,error='url_preprocessing_error'
                       WHERE split=? AND line=?""", (sha(source),sha(normalized(source)),sha(target),sha(dumps(messages[:-2])),
                       len(source),len(target),sum(len(message["content"]) for message in messages[:-1]),split,line))
            counts = record["counts"]
            counts[error] -= 1
            if counts[error] == 0:
                counts.pop(error)
            counts["valid_envelope_rows"] = counts.get("valid_envelope_rows", 0) + 1
            counts["url_preprocessing_error"] = counts.get("url_preprocessing_error", 0) + 1
            counts["url_preprocessing_error:"+error] = counts.get("url_preprocessing_error:"+error,0) + 1
    db.commit()


def duplicate_summary(db):
    result = {}
    for column in ("row_sha256", "canonical_row_sha256", "source_sha256", "source_normalized_sha256", "source_masked_sha256", "target_sha256", "scaffold_sha256"):
        metric = {}
        for split in ("train", "val", "combined"):
            where = f"{column} IS NOT NULL" + (" AND split=?" if split != "combined" else "")
            args = (split,) if split != "combined" else ()
            n, unique = db.execute(f"SELECT COUNT(*),COUNT(DISTINCT {column}) FROM rows WHERE {where}", args).fetchone()
            groups = db.execute(f"SELECT COUNT(*),SUM(n),MAX(n) FROM (SELECT COUNT(*) n FROM rows WHERE {where} GROUP BY {column} HAVING COUNT(*)>1)", args).fetchone()
            metric[split] = {"rows":n,"unique":unique,"excess_occurrences":n-unique,"duplicate_groups":groups[0],"rows_in_duplicate_groups":groups[1] or 0,"max_multiplicity":groups[2] or 1}
        overlap = db.execute(f"SELECT COUNT(*),SUM(t.n),SUM(v.n) FROM (SELECT {column} h,COUNT(*) n FROM rows WHERE split='train' AND {column} IS NOT NULL GROUP BY {column}) t JOIN (SELECT {column} h,COUNT(*) n FROM rows WHERE split='val' AND {column} IS NOT NULL GROUP BY {column}) v ON t.h=v.h").fetchone()
        metric["cross_split"] = dict(zip(("shared_groups","train_occurrences","val_occurrences"), [x or 0 for x in overlap]))
        result[column] = metric
    result["source_multiple_text_targets"] = {}
    for col in ("source_sha256", "source_normalized_sha256", "source_masked_sha256"):
        result["source_multiple_text_targets"][col] = {}
        for split in ("train", "val", "combined"):
            where = f"{col} IS NOT NULL" + (" AND split=?" if split != "combined" else "")
            row = db.execute(f"SELECT COUNT(*),SUM(n),MAX(targets) FROM (SELECT COUNT(*) n,COUNT(DISTINCT target_sha256) targets FROM rows WHERE {where} GROUP BY {col} HAVING COUNT(DISTINCT target_sha256)>1)", (split,) if split != "combined" else ()).fetchone()
            result["source_multiple_text_targets"][col][split] = dict(zip(("groups","occurrences","max_targets_per_source"), [x or 0 for x in row]))
    return result


def golden_audit(db):
    db.execute("DELETE FROM golden_hashes")
    files = {"golden32": REPO / "training/data/eval/golden32_archive_repeat_v1/golden32.jsonl",
             "golden35": REPO / "training/data/eval/golden35_v1/golden35.jsonl"}
    counts = {}
    for name, path in files.items():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            source = response_text(row)
            case = row.get("source_id") or row.get("query_id") or row.get("id")
            for kind, digest in zip(("exact", "normalized", "masked"), source_hashes(source)):
                db.execute("INSERT INTO golden_hashes VALUES (?,?,?,?,?)", (name,"accepted",case,kind,digest))
        manifest = json.loads((path.parent / "benchmark_manifest.json").read_text(encoding="utf-8"))
        for row in manifest.get("excluded_sources", []):
            for digest in {row.get("response_sha256"), *row.get("response_sha256s", [])} - {None}:
                db.execute("INSERT INTO golden_hashes VALUES (?,?,?,?,?)", (name,"excluded",row.get("source_id") or row.get("query_id"),"normalized_or_masked",digest))
        counts[name] = {"accepted_occurrences": len(rows), "excluded_sources": len(manifest.get("excluded_sources", [])), "benchmark_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    db.commit()
    overlaps = []
    for name in files:
        for membership in ("accepted", "excluded"):
            query = """SELECT r.split,COUNT(DISTINCT r.line),COUNT(DISTINCT r.source_normalized_sha256),COUNT(DISTINCT g.case_id)
                       FROM rows r JOIN golden_hashes g ON (g.digest=r.source_normalized_sha256 OR g.digest=r.source_masked_sha256 OR g.digest=r.source_sha256)
                       WHERE g.benchmark=? AND g.membership=? GROUP BY r.split"""
            for split, occurrences, sources, cases in db.execute(query, (name,membership)):
                overlaps.append({"benchmark":name,"membership":membership,"split":split,"occurrences":occurrences,"unique_sources":sources,"matched_golden_case_ids":cases})
    details = [{"benchmark":r[0],"membership":r[1],"case_id":r[2],"split":r[3],"occurrences":r[4],"first_line":r[5]}
               for r in db.execute("""SELECT g.benchmark,g.membership,g.case_id,r.split,COUNT(DISTINCT r.line),MIN(r.line)
                   FROM rows r JOIN golden_hashes g ON (g.digest=r.source_normalized_sha256 OR g.digest=r.source_masked_sha256 OR g.digest=r.source_sha256)
                   GROUP BY g.benchmark,g.membership,g.case_id,r.split ORDER BY 1,2,3,4""")]
    return {"benchmarks":counts,"overlaps":overlaps,"case_details":details}


def provenance_audit(db):
    db.execute("DELETE FROM provenance")
    wanted = {row[0] for row in db.execute("SELECT DISTINCT source_normalized_sha256 FROM rows WHERE source_normalized_sha256 IS NOT NULL")}
    wanted.update(row[0] for row in db.execute("SELECT DISTINCT source_masked_sha256 FROM rows WHERE source_masked_sha256 IS NOT NULL"))
    paths = sorted({* (REPO / "dataset/data/runs").rglob("responses.jsonl"), * (REPO / "dataset/data/runs").rglob("genui.jsonl")})
    stats = []
    for idx, path in enumerate(paths, 1):
        digest = hashlib.sha256()
        row_count = match_count = malformed = masking_errors = 0
        with path.open("rb") as stream:
            for line_no, raw in enumerate(stream, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                row_count += 1
                try:
                    row = json.loads(raw)
                except (ValueError, UnicodeError):
                    malformed += 1
                    continue
                if not isinstance(row, dict):
                    malformed += 1
                    continue
                source = row.get("response_text")
                if not isinstance(source, str) or not source:
                    continue
                try:
                    exact, norm, masked = source_hashes(source)
                except ValueError:
                    masking_errors += 1
                    exact, norm, masked = sha(source),sha(normalized(source)),None
                if norm not in wanted and masked not in wanted:
                    continue
                match_count += 1
                meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                gen = row.get("gen") if isinstance(row.get("gen"), dict) else {}
                db.execute("INSERT INTO provenance VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (exact,norm,masked,str(path.relative_to(REPO)),line_no,
                           row.get("query_id"),row.get("response_id"),row.get("ui_id"),row.get("intent_bucket") or meta.get("intent_bucket"),gen.get("model"),gen.get("provider"),path.stem))
        stats.append({"path":str(path.relative_to(REPO)),"bytes":path.stat().st_size,"sha256":digest.hexdigest(),"rows":row_count,"matched_rows":match_count,"malformed_rows":malformed,"url_masking_errors":masking_errors})
        if idx % 25 == 0:
            db.commit()
            print(f"Provenance: scanned {idx}/{len(paths)} files", flush=True)
    for col in ("source_normalized_sha256", "source_masked_sha256"):
        db.execute(f"CREATE INDEX IF NOT EXISTS provenance_{col} ON provenance ({col})")
    db.commit()
    summary = []
    for split in ("train", "val"):
        result = db.execute("""SELECT COUNT(*),COUNT(DISTINCT r.source_normalized_sha256) FROM rows r WHERE split=? AND EXISTS
           (SELECT 1 FROM provenance p WHERE p.source_normalized_sha256=r.source_normalized_sha256 OR p.source_masked_sha256=r.source_masked_sha256)""", (split,)).fetchone()
        summary.append({"split":split,"matched_occurrences":result[0],"matched_unique_sources":result[1]})
    return {"scanned_files":len(paths),"scanned_bytes":sum(s["bytes"] for s in stats),"matched_by_split":summary,"files":stats,
            "interpretation":"Source text matches do not prove target-generator provenance. Raw query/response IDs can be reused in separate dataset runs; retain the run path when restoring identity."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("C:/Users/anupk/Downloads/training_data"))
    parser.add_argument("--output-dir", type=Path, default=REPO/"training/reports/full_data_audit_20260913/inventory")
    parser.add_argument("--index", type=Path, default=REPO/"training/outputs/audits/full_data_20260913/inventory.sqlite")
    parser.add_argument("--skip-provenance", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true", help="Reuse completed row index; recompute only derived audit summaries")
    args = parser.parse_args()
    if args.index.exists() and not args.summarize_existing:
        raise SystemExit(f"Refusing to overwrite existing audit index: {args.index}")
    args.index.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.index)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    begin = time.monotonic()
    if args.summarize_existing:
        files = json.loads((args.output_dir/"files.json").read_text(encoding="utf-8"))
        for split, record in files.items():
            indexed = db.execute("SELECT COUNT(*) FROM rows WHERE split=?", (split,)).fetchone()[0]
            if indexed != record["physical_lines"] or Path(record["path"]).stat().st_size != record["bytes"]:
                raise SystemExit("Saved index is incomplete or source size changed; create a fresh audit index")
    else:
        files = index_inputs(args.source_dir, db)
    complete_failed_url_mask_records(db, files)
    write_json(args.output_dir/"files.json",files)
    duplicates = duplicate_summary(db)
    write_json(args.output_dir/"duplicates.json",duplicates)
    cross_split = [{"source_normalized_sha256":r[0],"train_lines":[int(x) for x in r[1].split(',')],"val_lines":[int(x) for x in r[2].split(',')]}
                   for r in db.execute("""SELECT t.h,t.lines,v.lines FROM
                     (SELECT source_normalized_sha256 h,GROUP_CONCAT(line) lines FROM rows WHERE split='train' GROUP BY source_normalized_sha256) t JOIN
                     (SELECT source_normalized_sha256 h,GROUP_CONCAT(line) lines FROM rows WHERE split='val' GROUP BY source_normalized_sha256) v ON t.h=v.h ORDER BY t.h""")]
    write_json(args.output_dir/"cross_split_sources.json",cross_split)
    print("UNIQUENESS " + json.dumps({key:value for key,value in duplicates.items() if key in ("source_sha256","target_sha256")}), flush=True)
    lengths = {split:{column:distribution(db,column,split) for column in ("source_chars","target_chars","prompt_chars","byte_length","target_extra_placeholders","target_missing_placeholders")} for split in ("train","val")}
    scaffolds = [{"sha256":r[0],"split":r[1],"occurrences":r[2],"first_line":r[3]} for r in db.execute("SELECT scaffold_sha256,split,COUNT(*),MIN(line) FROM rows GROUP BY scaffold_sha256,split")]
    placeholder_flags = [{"split":r[0],"rows_with_extra_target_placeholders":r[1],"rows_with_missing_target_placeholders":r[2],
                          "rows_with_both":r[3],"source_mojibake_marker_rows":r[4],"target_mojibake_marker_rows":r[5]}
                         for r in db.execute("""SELECT split,SUM(target_extra_placeholders>0),SUM(target_missing_placeholders>0),
                           SUM(target_extra_placeholders>0 AND target_missing_placeholders>0),SUM(source_mojibake),SUM(target_mojibake)
                           FROM rows GROUP BY split""")]
    write_json(args.output_dir/"structure.json",{"lengths":lengths,"scaffolds":scaffolds,"surface_quality_flags":placeholder_flags,
               "flag_limitations":"Placeholder-set differences flag missing references or ungrounded new references but are not a semantic correctness verdict. Mojibake patterns are heuristics; Ã may occur legitimately in some languages."})
    goldens = golden_audit(db)
    write_json(args.output_dir/"golden_overlap.json",goldens)
    print("GOLDEN_OVERLAP " + json.dumps(goldens["overlaps"]),flush=True)
    if not args.skip_provenance:
        write_json(args.output_dir/"provenance.json",provenance_audit(db))
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    write_json(args.output_dir/"audit_manifest.json",{"schema_version":1,"completed_at_utc":datetime.now(timezone.utc).isoformat(),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"source_directory":str(args.source_dir),"index":str(args.index),"elapsed_seconds_this_pass":round(time.monotonic()-begin,2),"summary_only_pass":args.summarize_existing,"read_only_source":True,"semantic_validation_performed":False,"tokenizer_validation_performed":False,"extraction":"Final user task response and final assistant target only; earlier few-shot turns excluded from source/target analysis.","normalization":"Whitespace folding; masked additionally uses production URL preprocessing. Four valid source strings raise URL masking errors and retain raw/normalized hashes with a NULL masked hash.","provenance_limit":"Hash matches prove source-text identity, not provenance of target generation; query IDs are run-scoped."})
    print(f"Audit complete in {time.monotonic()-begin:.1f}s: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
