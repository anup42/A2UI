"""Generate the strict standard catalog from the pinned Express profile.

The profile owns inference-only metadata; this generated JSON keeps standard
catalog component schemas strict and therefore prevents catalog/profile drift.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "schema" / "genuicraft_a2ui_catalog_v1.json"
PROFILE = ROOT / "schema" / "genuicraft_a2ui_express_profile_v1.json"


def _property_schema(name: str) -> dict[str, Any]:
    if name in {"children", "columns", "rows", "items", "options", "tabs", "highlightColumns", "numericColumns", "attachments", "entityMedia"}:
        return {"type": "array"}
    if name in {"repeat", "watch", "on"}:
        return {"type": "object"}
    if name in {"value", "result", "min", "max", "step", "width", "height", "aspectRatio", "size", "iconSize", "index"}:
        return {"type": ["string", "number", "boolean", "null"]}
    if name in {"visible", "decorative", "wantResponse"}:
        return {"type": ["boolean", "string", "object", "null"]}
    return {}


def main() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    catalog["catalogVersion"] = "genuicraft-a2ui-catalog-v1"
    catalog["profileVersion"] = profile["profileVersion"]
    components = catalog.get("components", {})
    profile_components = profile.get("components", {})
    profile_actions = profile.get("actions", {})
    common_properties = {str(value) for value in profile.get("commonProperties", ())}
    common_reference_properties = {
        str(value) for value in profile.get("commonReferenceProperties", ())
    }
    for name, descriptor in components.items():
        p = profile_components.get(name, {})
        allowed = set(descriptor.get("positional", ()))
        allowed.update(p.get("properties", ()))
        allowed.update(p.get("childProperties", ()))
        allowed.update(common_properties)
        allowed.update(common_reference_properties)
        allowed.discard("children[]")
        schema_properties: dict[str, Any] = {
            "id": {"type": "string", "minLength": 1},
            "component": {"const": name},
            "children": {"type": "array", "items": {"type": "string", "minLength": 1}},
            "repeat": {"type": "object"},
            "visible": {},
            "on": {"type": "object"},
            "watch": {"type": "object"},
        }
        for prop in sorted(allowed):
            if prop and prop not in schema_properties:
                schema_properties[prop] = _property_schema(prop)
        descriptor["allowAdditionalProps"] = False
        descriptor["allowedProperties"] = sorted(allowed)
        descriptor["schema"] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "component"],
            "properties": schema_properties,
        }
    for name, descriptor in (catalog.get("actions", {}) or {}).items():
        descriptor["allowAdditionalParams"] = False
        descriptor["required"] = list(profile_actions.get(name, {}).get("required", ()))
        descriptor["schema"] = {
            "type": "object",
            "additionalProperties": False,
            "required": descriptor["required"],
            "properties": {str(key): {} for key in descriptor.get("positional", ())},
        }
    payload = {key: value for key, value in catalog.items() if key != "catalogIdentityHash"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    catalog["catalogIdentityHash"] = hashlib.sha256(encoded).hexdigest()
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
