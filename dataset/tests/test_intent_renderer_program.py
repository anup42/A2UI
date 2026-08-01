from __future__ import annotations

import json
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[1]
SRC = DATASET_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.flat_spec_contract import coerce_and_validate  # noqa: E402


def _load(relative: str) -> dict:
    return json.loads((DATASET_ROOT / relative).read_text(encoding="utf-8"))


def test_all_32_intents_have_one_deterministic_recipe_and_fixture() -> None:
    declared = [line.strip() for line in (DATASET_ROOT / "intents.info").read_text(encoding="utf-8").splitlines() if line.strip()]
    recipes = _load("schema/intent_renderer_recipes.json")["recipes"]
    fixtures = _load("tests/fixtures/intent_flat_specs_v2.json")["fixtures"]
    assert len(declared) == len(recipes) == len(fixtures) == 32
    assert [entry["intent"] for entry in recipes] == declared
    assert [entry["intent"] for entry in fixtures] == declared
    assert len({entry["bucket"] for entry in recipes}) == 32


def test_recipes_only_claim_manifest_types_and_domains() -> None:
    manifest = _load("schema/renderer_capabilities.json")
    recipes = _load("schema/intent_renderer_recipes.json")["recipes"]
    types = {entry["canonical"] for entry in manifest["types"]}
    domains = set(manifest["table_domains"]["canonical"])
    for recipe in recipes:
        assert set(recipe["presentation"]) <= types, recipe["intent"]
        assert recipe["domain"] in domains, recipe["intent"]


def test_all_intent_fixtures_are_strict_valid_and_canonical() -> None:
    fixtures = _load("tests/fixtures/intent_flat_specs_v2.json")["fixtures"]
    for fixture in fixtures:
        result = coerce_and_validate(fixture["spec"])
        assert result.is_valid, f"{fixture['intent']}: {result.error}"
        assert result.spec == fixture["spec"], fixture["intent"]


def test_high_stakes_and_scanner_scope_are_explicit() -> None:
    recipes = {
        entry["bucket"]: entry
        for entry in _load("schema/intent_renderer_recipes.json")["recipes"]
    }
    for bucket in ("healthcare", "legal"):
        rule = recipes[bucket]["rule"].lower()
        assert "presentation-only" in rule
        assert "source" in rule and "disclaimer" in rule
        assert "conclusion" in rule
    scanner = recipes["qr_scanner"]["rule"].lower()
    assert "camera capture" in scanner and "recognition" in scanner
    assert "out of scope" in scanner
