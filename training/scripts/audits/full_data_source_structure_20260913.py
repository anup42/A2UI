"""Complete source markup/Unicode census with target Table-presence cross-tabs.

This is a lexical/character audit, not source intent or language classification.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

from full_data_inventory_20260913 import REPO, TASK_PREFIX, physical_lines, write_json

TABLE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
PATTERNS = {
    "markdown_heading": re.compile(r"^\s*#{1,6}\s+\S", re.M),
    "numbered_list_line": re.compile(r"^\s*\d+[.)]\s+\S", re.M),
    "bullet_list_line": re.compile(r"^\s*[-+*\u2022]\s+\S", re.M),
    "blockquote_line": re.compile(r"^\s*>\s*\S", re.M),
    "markdown_link": re.compile(r"(?<!!)\[[^\]\n]+\]\([^\)\n]+\)"),
    "markdown_image": re.compile(r"!\[[^\]\n]*\]\([^\)\n]+\)"),
    "media_declaration_line": re.compile(r"^\s*(?:Media|Image|Video|Audio|Icon)\s*[:=]", re.M|re.I),
    "named_action_line": re.compile(r"^\s*Action:\s*\[Button:", re.M|re.I),
    "literal_http_scheme": re.compile(r"https?://", re.I),
    "literal_other_action_scheme": re.compile(r"\b(?:mailto|tel|geo|intent|genuicraft):", re.I),
    "image_placeholder": re.compile(r"\[(?:IMAGE_URL|IMAGE_ASSET)_\d+\]"),
    "icon_placeholder": re.compile(r"\[(?:ICON_URL|ICON_ASSET)_\d+\]"),
    "action_placeholder": re.compile(r"\[ACTION_URL_\d+\]"),
    "source_placeholder": re.compile(r"\[SOURCE_URL_\d+\]"),
    "media_placeholder": re.compile(r"\[(?:MEDIA_URL|MEDIA_ASSET)_\d+\]"),
    "generic_url_placeholder": re.compile(r"\[URL_\d+\]"),
}
BLOCKS = {
    "latin_extended": ((0x00C0,0x024F),), "greek": ((0x0370,0x03FF),),
    "cyrillic": ((0x0400,0x052F),), "hebrew": ((0x0590,0x05FF),),
    "arabic": ((0x0600,0x06FF),(0x0750,0x077F),(0x08A0,0x08FF)),
    "devanagari": ((0x0900,0x097F),), "bengali": ((0x0980,0x09FF),),
    "gurmukhi": ((0x0A00,0x0A7F),), "gujarati": ((0x0A80,0x0AFF),),
    "tamil": ((0x0B80,0x0BFF),), "telugu": ((0x0C00,0x0C7F),),
    "kannada": ((0x0C80,0x0CFF),), "malayalam": ((0x0D00,0x0D7F),),
    "sinhala": ((0x0D80,0x0DFF),), "thai": ((0x0E00,0x0E7F),),
    "hangul": ((0x1100,0x11FF),(0x3130,0x318F),(0xAC00,0xD7AF)),
    "hiragana": ((0x3040,0x309F),), "katakana": ((0x30A0,0x30FF),(0x31F0,0x31FF)),
    "cjk_ideographs": ((0x3400,0x4DBF),(0x4E00,0x9FFF),(0xF900,0xFAFF),(0x20000,0x2EBEF)),
    "box_drawing": ((0x2500,0x257F),), "block_elements": ((0x2580,0x259F),),
    "general_punctuation": ((0x2000,0x206F),), "currency_symbols": ((0x20A0,0x20CF),),
    "emoji_or_dingbat_range": ((0x2600,0x27BF),(0x1F300,0x1FAFF)),
    "replacement_character": ((0xFFFD,0xFFFD),),
}


def inspect_source(source):
    flags = {name:bool(pattern.search(source)) for name,pattern in PATTERNS.items()}
    any_placeholder = any(value for key,value in flags.items() if key.endswith("_placeholder"))
    flags.update(any_typed_placeholder=any_placeholder,
                 literal_http_and_placeholder=flags["literal_http_scheme"] and any_placeholder,
                 literal_http_without_placeholder=flags["literal_http_scheme"] and not any_placeholder,
                 placeholder_without_literal_http=any_placeholder and not flags["literal_http_scheme"])
    fence_char, fence_size = None,0
    table_any = table_outside = pipe_rows = fence_starts = 0
    for line in source.splitlines():
        match = FENCE.match(line)
        is_fence = match is not None
        if match:
            token = match.group(1)
            if fence_char is None:
                fence_char, fence_size = token[0],len(token)
                fence_starts += 1
            elif token[0] == fence_char and len(token) >= fence_size:
                fence_char,fence_size = None,0
        separator = TABLE.fullmatch(line) is not None
        table_any += separator
        if fence_char is None and not is_fence:
            table_outside += separator
            pipe_rows += int("|" in line and len(line.strip().strip("|").split("|"))>=2 and bool(re.search(r"\w",line)))
    flags.update(table_separator_anywhere=bool(table_any),table_separator_outside_fences=bool(table_outside),
                 pipe_row_outside_fences=bool(pipe_rows),fenced_code=bool(fence_starts),unclosed_fence_at_end=fence_char is not None)
    nonascii = {ord(char) for char in source if ord(char)>127}
    flags["non_ascii_character"] = bool(nonascii)
    flags["ascii_only"] = not nonascii
    unicode_presence = {name for name,ranges in BLOCKS.items() if any(low<=cp<=high for cp in nonascii for low,high in ranges)}
    return flags,unicode_presence,table_outside


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,default=Path("C:/Users/anupk/Downloads/training_data"))
    parser.add_argument("--ir-index",type=Path,default=REPO/"training/outputs/audits/full_data_20260913/ir.sqlite")
    parser.add_argument("--inventory-manifest",type=Path,default=REPO/"training/reports/full_data_audit_20260913/inventory/files.json")
    parser.add_argument("--output-dir",type=Path,default=REPO/"training/reports/full_data_audit_20260913/source_structure")
    args=parser.parse_args()
    started=time.monotonic()
    connection=sqlite3.connect(args.ir_index.resolve().as_uri()+"?mode=ro",uri=True)
    statuses={split:bytearray(connection.execute("SELECT MAX(line) FROM rows WHERE split=?",(split,)).fetchone()[0]+1) for split in ("train","val")}
    target_hashes={split:[None]*len(values) for split,values in statuses.items()}
    inventory_manifest=json.loads(args.inventory_manifest.read_text(encoding="utf-8"))
    # 0 unknown graph; 1 valid/noTable; 2 valid/Table; 3 invalid/noTable; 4 invalid/Table.
    target_counts=defaultdict(Counter)
    for split,line,valid,parsed,has_table,target_sha in connection.execute("""SELECT r.split,r.line,t.valid,
      json_type(t.features,'$.components') IS NOT NULL,COALESCE(json_extract(t.features,'$.components.Table'),0)>0
      ,r.target_sha FROM rows r JOIN targets t ON r.target_sha=t.sha"""):
        statuses[split][line]=(1 if valid else 3)+int(has_table) if parsed else 0
        target_hashes[split][line]=target_sha
        target_counts[split]["strict_valid"]+=int(valid)
        target_counts[split]["graph_known"]+=int(parsed)
        target_counts[split]["graph_unknown"]+=int(not parsed)
        target_counts[split]["strict_valid_with_Table"]+=int(valid and has_table)
    connection.close()
    counts,unicode_counts,histograms,crosstabs,examples,file_hashes={},{},{},{},{},{}
    for split in ("train","val"):
        count,unicode_count,histogram=Counter(),Counter(),Counter()
        crosses=defaultdict(Counter)
        example=defaultdict(list)
        digest=hashlib.sha256()
        line=0
        for line,(offset,raw,encoding) in enumerate(physical_lines(args.source_dir/f"{split}.jsonl"),1):
            digest.update(raw)
            row=json.loads(raw.decode(encoding).removeprefix("\ufeff"))
            if hashlib.sha256(row["messages"][-1]["content"].encode("utf-8")).hexdigest()!=target_hashes[split][line]:
                raise ValueError(f"IR target hash mismatch at {split}:{line}")
            task=row["messages"][-2]["content"]
            if not task.startswith(TASK_PREFIX):
                raise ValueError(f"Task prefix mismatch at {split}:{line}")
            source=task[len(TASK_PREFIX):]
            flags,blocks,table_count=inspect_source(source)
            status=statuses[split][line]
            count["rows"]+=1
            count.update({key:int(value) for key,value in flags.items()})
            unicode_count.update(blocks)
            histogram[table_count]+=1
            has_target_table=status in (2,4)
            for key in ("table_separator_anywhere","table_separator_outside_fences","pipe_row_outside_fences","image_placeholder","markdown_image","media_declaration_line"):
                cell=f"source_{'yes' if flags[key] else 'no'}_target_Table_{'yes' if has_target_table else 'no'}"
                if status:
                    crosses[f"{key}:all_parsed_targets"][cell]+=1
                if status in (1,2):
                    crosses[f"{key}:strict_valid_targets"][cell]+=1
            for category,present in (("source_separator_no_target_Table",flags["table_separator_outside_fences"] and status==1),
                                     ("target_Table_no_source_separator",not flags["table_separator_outside_fences"] and status==2)):
                if present and len(example[category])<12:
                    example[category].append({"line":line,"byte_offset":offset,"byte_length":len(raw),"source_chars":len(source),
                                             "source_has_pipe_row":flags["pipe_row_outside_fences"],"source_has_bullets":flags["bullet_list_line"],
                                             "source_has_numbered_list":flags["numbered_list_line"]})
            if line%50000==0:
                print(f"{split}: {line:,} sources inspected in {time.monotonic()-started:.1f}s",flush=True)
        if line!=len(statuses[split])-1:
            raise ValueError(f"IR/source row count mismatch for {split}")
        counts[split],unicode_counts[split],histograms[split],crosstabs[split],examples[split]=count,unicode_count,histogram,crosses,example
        file_hashes[split]=digest.hexdigest()
        if file_hashes[split]!=inventory_manifest[split]["sha256"]:
            raise ValueError(f"Source file hash differs from inventory: {split}")
    report={"schema_version":1,"completed_at_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":round(time.monotonic()-started,2),
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"source_hashes":file_hashes,"counts":counts,
            "binding_checks":"Every decoded final target SHA256 matched its IR split/line coordinate; both complete source-file SHA256 hashes matched the original inventory.",
            "unicode_block_row_presence":unicode_counts,"source_table_separator_count_histogram":histograms,
            "target_graph_counts":target_counts,"source_target_Table_crosstabs":crosstabs,"bounded_examples":examples,
            "methods":{"table_separator_regex":TABLE.pattern,"source_extraction":"Final user TASK_PREFIX response only, no few-shot content.",
                       "table_detection":"Markdown-like separator rows with >=2 dash columns and >=3 dashes each; count anywhere and separately outside simple backtick/tilde fences.",
                       "pipe_row_detection":"Outside fenced blocks, a line with pipe-separated columns and at least one Unicode word character; weak signal, not proof of table semantics.",
                       "unicode_blocks":BLOCKS},
            "limits":["Markup presence is not semantic intent classification. Table-like data can occur without Markdown separator rows; pipe rows may instead be prose, code, or structured non-table content.",
                      "Headings/list/link flags can also match literal examples inside fenced blocks; only the separately named table/pipe metrics exclude detected fences.",
                      "Simple fence recognition is not a full Markdown parser and can miss nested/indented or malformed fence semantics.",
                      "Table is a renderer component and can legitimately present non-table source lists as cards. Source/target syntax disagreement alone does not demonstrate an unjustified Table habit.",
                      "Image component absence does not prove missing pictures: images may be bound in Table rows or other properties. Review actual asset bindings and rendered output.",
                      "Unicode-block presence is not language classification; names, symbols, and mojibake can trigger ranges (especially Greek, box drawing, and block elements).",
                      "Literal URL schemes are not verified reachable URLs. Placeholder presence is not proof of usable assets or correct role binding."]}
    write_json(args.output_dir/"summary.json",report)
    print(json.dumps({"completed":True,"counts":counts,"Table_crosstabs":{s:{k:v for k,v in c.items() if k.startswith('table_separator_outside_fences:')} for s,c in crosstabs.items()},"elapsed_seconds":report["elapsed_seconds"]},indent=2),flush=True)


if __name__=="__main__":
    main()
