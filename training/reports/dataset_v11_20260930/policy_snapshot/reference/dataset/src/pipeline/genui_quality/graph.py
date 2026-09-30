"""Declared-root graph audits and JSON-pointer public surface."""

from __future__ import annotations

from typing import Any, Mapping

from ..renderer_semantics import iter_renderer_references
from ._core import GraphAudit, audit_graph, resolve_json_pointer


def audit_renderer_graph(spec: Mapping[str, Any]) -> GraphAudit:
    """Audit v5 reachability using the shared Android renderer inventory."""
    raw_elements = spec.get("elements")
    elements: Mapping[str, Any] = (
        raw_elements if isinstance(raw_elements, Mapping) else {}
    )
    root_value = spec.get("root")
    root = (
        root_value
        if isinstance(root_value, str) and root_value in elements
        else None
    )

    all_ids = {str(key) for key in elements}
    parents: dict[str, set[str]] = {key: set() for key in all_ids}
    for parent_id, raw in elements.items():
        if not isinstance(raw, Mapping):
            continue
        for reference in iter_renderer_references(raw):
            if reference.target_id in elements:
                parents.setdefault(reference.target_id, set()).add(str(parent_id))

    reachable: set[str] = set()
    missing: set[str] = set()
    cycles: set[tuple[str, str]] = set()
    depths: dict[str, int] = {}
    colors: dict[str, int] = {}

    def visit(node_id: str, depth: int) -> None:
        color = colors.get(node_id, 0)
        if color == 1:
            return
        depths[node_id] = min(depths.get(node_id, depth), depth)
        if color == 2:
            reachable.add(node_id)
            return
        colors[node_id] = 1
        reachable.add(node_id)
        raw = elements.get(node_id)
        references = (
            iter_renderer_references(raw)
            if isinstance(raw, Mapping)
            else ()
        )
        for reference in references:
            child = reference.target_id
            if child not in elements:
                missing.add(child)
                continue
            if colors.get(child, 0) == 1:
                cycles.add((node_id, child))
                continue
            visit(child, depth + 1)
        colors[node_id] = 2

    if root is not None:
        visit(root, 0)

    return GraphAudit(
        root_exists=root is not None,
        reachable_ids=reachable,
        unreachable_ids=all_ids - reachable,
        missing_references=missing,
        cycle_edges=cycles,
        multi_parent_ids={
            key
            for key, values in parents.items()
            if len(values) > 1 and key in reachable
        },
        depths=depths,
    )


__all__ = [
    "GraphAudit",
    "audit_graph",
    "audit_renderer_graph",
    "resolve_json_pointer",
]
