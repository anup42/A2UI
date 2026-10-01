"""Complete v10 source-reference and retained-provenance census.

Run from the repository root: python training/reports/v10_full_audit_20260926/reference_census.py
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "training/outputs/datasets/full_data_archive_recovered_v10"
OUT = Path(__file__).resolve().parent
PLACEHOLDER = re.compile(r"\[(?:IMAGE_URL|ICON_URL|ACTION_URL|SOURCE_URL|MEDIA_URL|URL|IMAGE_ASSET|ICON_ASSET|MEDIA_ASSET)_\d+\]")
REAL_URL = re.compile(r"\bhttps?://[^\s<>\"']+", re.IGNORECASE)
OTHER_REFERENCE = re.compile(r"\b(?:mailto|tel|geo|intent|genuicraft|data|javascript|blob|urn|sms|market):[^\s<>\"']+", re.IGNORECASE)


def main():
    report = {"schema_version": 1, "definition": {
        "placeholder": PLACEHOLDER.pattern,
        "real_url": REAL_URL.pattern,
        "other_scheme": OTHER_REFERENCE.pattern,
        "placeholder_rich_threshold": "at least 3 distinct placeholder tokens in response_text"}, "splits": {}}
    for split in ("train", "val"):
        path = DATA / f"{split}.jsonl"
        counts: Counter[str] = Counter()
        placeholder_kinds: Counter[str] = Counter()
        missing_historical: Counter[str] = Counter()
        examples = {key: [] for key in ("placeholder_rich_no_map", "real_url_nonempty_map", "real_url_empty_map")}
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for line, raw in enumerate(stream, 1):
                digest.update(raw)
                row = json.loads(raw)
                source = row["response_text"]
                tokens = set(PLACEHOLDER.findall(source))
                real_urls = set(REAL_URL.findall(source))
                other_refs = set(OTHER_REFERENCE.findall(source))
                meta = row.get("metadata") or {}
                recovery = meta.get("archive_recovery") or {}
                url_metadata = meta.get("url_preprocessing") or {}
                url_map = url_metadata.get("url_map") or {}
                mapped_tokens = tokens & set(url_map)
                counts["rows"] += 1
                counts["has_placeholder"] += bool(tokens)
                counts["placeholder_rich_3plus"] += len(tokens) >= 3
                counts["has_real_http_url"] += bool(real_urls)
                counts["has_other_scheme_reference"] += bool(other_refs)
                counts["placeholder_only_no_real_http"] += bool(tokens) and not real_urls
                counts["real_http_only_no_placeholder"] += bool(real_urls) and not tokens
                counts["placeholder_and_real_http"] += bool(tokens) and bool(real_urls)
                counts["neither_placeholder_nor_real_http"] += not tokens and not real_urls
                counts["source_placeholder_occurrences"] += len(PLACEHOLDER.findall(source))
                counts["source_distinct_placeholder_tokens"] += len(tokens)
                counts["source_real_http_url_occurrences"] += len(REAL_URL.findall(source))
                counts["url_metadata_present"] += "url_preprocessing" in meta
                counts["url_map_nonempty"] += bool(url_map)
                counts["url_map_entries"] += len(url_map)
                counts["source_tokens_mapped_by_current_map"] += len(mapped_tokens)
                counts["source_tokens_unmapped_by_current_map"] += len(tokens - mapped_tokens)
                counts["placeholder_rows_all_tokens_mapped"] += bool(tokens) and tokens <= set(url_map)
                counts["placeholder_rows_partial_map"] += bool(mapped_tokens) and bool(tokens - mapped_tokens)
                counts["placeholder_rows_no_tokens_mapped"] += bool(tokens) and not mapped_tokens
                counts["map_entries_http_url"] += sum(str(item.get("url", "")).lower().startswith(("http://", "https://"))
                                                       for item in url_map.values() if isinstance(item, dict))
                counts["map_entries_local_asset"] += sum(item.get("kind") == "local_asset"
                                                         for item in url_map.values() if isinstance(item, dict))
                counts["has_placeholder_and_empty_map"] += bool(tokens) and not url_map
                counts["has_real_http_and_empty_map"] += bool(real_urls) and not url_map
                counts["url_preprocessing_enabled"] += url_metadata.get("enabled") is True
                counts["url_source_closed"] += url_metadata.get("source_closed") is True
                counts["original_url_map_marked_missing"] += "original_url_map" in (recovery.get("missing_historical_metadata") or [])
                counts["original_query_id_marked_missing"] += "original_query_id" in (recovery.get("missing_historical_metadata") or [])
                counts["generator_lineage_marked_missing"] += "generator_lineage" in (recovery.get("missing_historical_metadata") or [])
                counts["asset_manifest_marked_missing"] += "asset_manifest" in (recovery.get("missing_historical_metadata") or [])
                counts["source_model_family_present"] += bool(row.get("source_model_family") or meta.get("source_model_family"))
                for token in tokens:
                    placeholder_kinds[token[1:-1].rsplit("_", 1)[0]] += 1
                for item in recovery.get("missing_historical_metadata") or []:
                    missing_historical[item] += 1
                category = ("placeholder_rich_no_map" if len(tokens) >= 3 and not url_map else
                            "real_url_nonempty_map" if real_urls and url_map else
                            "real_url_empty_map" if real_urls else None)
                if category and len(examples[category]) < 5:
                    examples[category].append({"coordinate": f"{split}:{line}",
                                                "placeholder_tokens": len(tokens), "real_http_urls": len(real_urls),
                                                "url_map_entries": len(url_map),
                                                "source_preview": " ".join(source.split())[:160]})
                if line % 20000 == 0:
                    print(f"{split}: {line:,}", flush=True)
        report["splits"][split] = {"file_sha256": digest.hexdigest(), "counts": dict(counts),
                                     "placeholder_kind_distinct_token_totals": dict(placeholder_kinds),
                                     "missing_historical_metadata": dict(missing_historical), "examples": examples}
    report["limits"] = ["A nonempty current url_preprocessing.url_map can map newly recognized real references, but it does not prove recovery of the historical map for preexisting placeholders.",
                        "Presence of placeholders is a surface signal; it does not prove a URL is invalid or absent from the original response.",
                        "The HTTP regex is intentionally narrow. Other URI schemes are counted separately; local asset paths are not categorized as HTTP URLs."]
    (OUT / "reference_counts.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({split: result["counts"] for split, result in report["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
