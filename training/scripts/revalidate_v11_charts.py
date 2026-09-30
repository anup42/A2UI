"""Recheck immutable v11s charts against the current renderer; never rewrite labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'dataset/src'), str(ROOT / 'training/src')]
from pipeline.genui_quality.config_v5_4 import coerce_reward_config_v5_4
from pipeline.genui_quality.evidence_v5_4 import collect_output_evidence_v5_4
from pipeline.genui_quality.graph import audit_renderer_graph
from pipeline.ir_formats.active import decode_express_completion
from ir_training.data.express_preparation import PreparationError, prepare_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True, type=Path)
    parser.add_argument('--output-report', required=True, type=Path)
    args = parser.parse_args()
    source, destination = args.dataset_dir.resolve(), args.output_report.resolve()
    if destination.exists() or destination.is_relative_to(source):
        parser.error('Output must be a new file outside the supplied dataset directory')
    catalog = json.loads((ROOT / 'training/data/quality/v11_review_holds_20261001.json').read_text(encoding='utf-8'))
    implementation = [ROOT / 'dataset/schema/renderer_capabilities.json',
                      ROOT / 'dataset/src/pipeline/rich_chart_semantics.py',
                      ROOT / 'dataset/src/pipeline/renderer_effective_semantics_v5_4.py',
                      ROOT / 'training/src/ir_training/data/release_review.py']
    initial_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in implementation}
    prior = {row['id']: row for row in catalog['chart_rows']}
    joins = {row['id'] for row in catalog['join_rows']}
    diagnostics, subtypes, counts = Counter(), Counter(), Counter()
    results = []
    found = set()
    split_hashes = {}
    for split in ('train', 'val'):
        digest = hashlib.sha256()
        with (source / f'{split}.jsonl').open('rb') as stream:
            for raw in stream:
                digest.update(raw)
                row = json.loads(raw)
                counts['rows_scanned'] += 1
                if 'Chart(' not in row['completion']:
                    continue
                counts['rows_with_charts'] += 1
                graph = decode_express_completion(row['completion'])
                evidence = collect_output_evidence_v5_4(graph, audit_renderer_graph(graph), coerce_reward_config_v5_4(None))
                charts = [item for item in evidence.effective_components if item.get('kind') == 'chart']
                for chart in charts:
                    subtypes[chart['chart_type']] += 1
                bad = [{'component_id': item['component_id'], 'type': item['chart_type'], 'diagnostics': item['diagnostics']}
                       for item in charts if not item.get('complete')]
                for item in bad:
                    diagnostics.update(item['diagnostics'])
                reason = None
                try:
                    prepare_row(row, 'root-first')
                except PreparationError as error:
                    reason = error.reason
                counts[f'chart_preparation_{reason or "accepted"}'] += 1
                is_prior = row['id'] in prior
                target_hash = hashlib.sha256(row['completion'].encode('utf-8')).hexdigest()
                if is_prior:
                    found.add(row['id'])
                    if target_hash != prior[row['id']]['target_sha256']:
                        raise ValueError(f'Changed supplied target: {row["id"]}')
                    status = 'still_invalid' if bad else 'renderable_text_hold' if row['id'] in joins else 'renderable'
                    counts[f'previously_held_{status}'] += 1
                    counts[f'{split}_previously_held_{status}'] += 1
                elif bad:
                    counts['additional_invalid_chart_rows'] += 1
                results.append({'id': row['id'], 'split': split, 'source_id': row['source_id'],
                                'target_sha256': target_hash, 'previously_chart_held': is_prior,
                                'text_boundary_held': row['id'] in joins, 'chart_complete': not bad,
                                'preparation_reason': reason, 'issues': bad})
        split_hashes[split] = digest.hexdigest()
    if found != set(prior):
        raise ValueError(f'Incomplete prior-hold coverage: {len(found)}/{len(prior)}')
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != value for p, value in initial_hashes.items()):
        raise RuntimeError('Renderer contract changed during audit; rerun with a stable checkout')
    report = {'scope': 'Renderer/data-binding and standard preparation recheck, not a complete source-quality or tokenizer acceptance run. Supplied records are unchanged.',
              'renderer_capability_version': '2.1.0', 'counts': dict(counts), 'split_sha256': split_hashes,
              'implementation_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in implementation},
              'remaining_text_boundary_holds': len(joins), 'chart_subtypes': dict(subtypes),
              'diagnostic_counts': dict(diagnostics), 'rows': results}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in report.items() if key not in {'rows', 'implementation_sha256'}}, indent=2))


if __name__ == '__main__':
    main()
