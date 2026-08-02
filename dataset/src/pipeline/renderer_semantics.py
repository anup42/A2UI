"""Active canonical renderer reference inventory.

The historical ``flat_spec_semantics`` module remains as a migration alias;
active Express validation imports this neutral name so reference traversal is
shared without making FlatSpec the production contract.
"""
from __future__ import annotations

from .flat_spec_semantics import (
    RENDERER_SEMANTICS_VERSION,
    RendererReference,
    iter_renderer_references,
    renderer_reference_inventory_hash,
)

__all__ = [
    "RENDERER_SEMANTICS_VERSION",
    "RendererReference",
    "iter_renderer_references",
    "renderer_reference_inventory_hash",
]
