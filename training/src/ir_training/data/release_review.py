"""Hold reviewed v11 targets without mutating the immutable release bytes."""
from __future__ import annotations

from functools import lru_cache
import json
from typing import Any, Mapping

from ir_training.common.config import repo_root


@lru_cache(maxsize=1)
def review_policy() -> set[tuple[str, str]]:
    root = repo_root()
    catalog = json.loads((root / 'training/data/quality/v11_review_holds_20261001.json').read_text(encoding='utf-8'))
    holds = set()
    for entry in catalog['join_rows']:
        identity, semantic = entry['source_id'], entry['semantic_sha256']
        if not isinstance(identity, str) or not identity or not isinstance(semantic, str) or len(semantic) != 64:
            raise ValueError('Invalid v11 reviewed target identity')
        holds.add((identity, semantic))
    return holds


def reviewed_hold(row: Mapping[str, Any], graph: Mapping[str, Any], semantic_sha256: str) -> tuple[str, str] | None:
    metadata = row.get('metadata')
    if not isinstance(metadata, dict) or not isinstance(metadata.get('space_import'), dict):
        return None
    holds = review_policy()
    identity = str(row.get('source_id') or metadata.get('source_id') or '')
    if (identity, semantic_sha256) in holds:
        return ('reviewed_text_boundary_hold', 'v11 legacy text-chunk repair deleted or lacks proof for a source boundary; regenerate through Stage 3')
    if any(element['type'] == 'Chart' for element in graph['elements'].values()):
        # The effective interpreter resolves state/item bindings and visibility;
        # inspecting raw props would reject a valid chartType="${/kind}" binding.
        from pipeline.genui_quality.config_v5_4 import coerce_reward_config_v5_4
        from pipeline.genui_quality.evidence_v5_4 import collect_output_evidence_v5_4
        from pipeline.genui_quality.graph import audit_renderer_graph
        evidence = collect_output_evidence_v5_4(graph, audit_renderer_graph(graph), coerce_reward_config_v5_4(None))
        for component in evidence.effective_components:
            if component.get('kind') == 'chart' and 'unsupported_chart_subtype' in component.get('diagnostics', ()):
                return ('unsupported_renderer_chart', f'{component["component_id"]}: Chart({component.get("chart_type")}) is not supported by the current Android renderer; regenerate without inventing chart semantics')
            if component.get('kind') == 'chart' and not component.get('complete'):
                return ('invalid_renderer_chart', f'{component["component_id"]}: Chart({component.get("chart_type")}) cannot render faithfully: {component.get("diagnostics")}; regenerate missing bindings through Stage 3')
    return None
