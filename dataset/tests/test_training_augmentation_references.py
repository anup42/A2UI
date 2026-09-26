"""Offline reference transport through augmentation and canonical Stage3 masking."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.augmentation_references import (
    assert_reference_safe_source,
    bind_generated_source,
    build_reference_context,
)
from pipeline.reference_boundaries import reference_destination_errors
from pipeline.stage3_genui import _mask_model_references, _restore_model_references


def test_exact_donor_destinations_roundtrip_no_downloads():
    source = (
        "Source: https://en.wikipedia.org/wiki/Title_(part_(2))\n"
        'Media: Image="C:/device assets/flight (final).png"\n'
        "Action: [Button: Details] https://example.test/details?a=%28x%29&b=2"
    )
    with patch("pipeline.stage3_genui._download_response_asset", side_effect=AssertionError("download")), \
            patch("pipeline.stage3_genui._auto_download_response_assets", side_effect=AssertionError("download")):
        context = build_reference_context(source, "candidate")
        restored, evidence = bind_generated_source(context["donor_response_text"], context)
    assert restored == source
    assert len(context["reference_map"]) == 3
    assert evidence["asset_downloads_performed"] is False
    assert evidence["resource_existence_verified"] is False
    assert bind_generated_source(restored, context) == (restored, evidence)


def test_sidecar_restores_explicit_donor_map_without_changing_input():
    source = "Media: Icon=[ICON_URL_4]\nAction: [Button: Details] [ACTION_URL_2]"
    mapping = {"[ICON_URL_4]": "https://example.test/icon.svg", "[ACTION_URL_2]": "app://details/2"}
    context = build_reference_context(source, "candidate", donor_reference_map=mapping)
    restored, _ = bind_generated_source(context["donor_response_text"], context)
    assert restored == "Media: Icon=https://example.test/icon.svg\nAction: [Button: Details] app://details/2"
    assert source.endswith("[ACTION_URL_2]")
    assert mapping["[ICON_URL_4]"] == "https://example.test/icon.svg"


@pytest.mark.parametrize("token", ["[ICON_URL_1]", "[ACTION_URL_2]", "[SOURCE_URL_3]"])
def test_unknown_donor_tokens_never_get_invented_bindings(token):
    with pytest.raises(ValueError, match="unbound_donor_reference"):
        build_reference_context("Reference: " + token, "candidate")


@pytest.mark.parametrize("token", ["[ICON_URL_99]", "[ACTION_URL_99]", "[SOURCE_URL_99]"])
def test_unknown_generated_tokens_rejected(token):
    context = build_reference_context("Donor", "candidate", category="literal_media_references")
    with pytest.raises(ValueError, match="unknown_generated_reference"):
        bind_generated_source("Reference: " + token, context)


@pytest.mark.parametrize("token", ["[ICON_URL_", "[ACTION_URL_nope]", "[SOURCE_URL_3", "[url_1]"])
def test_malformed_tokens_rejected(token):
    context = build_reference_context("Donor", "candidate")
    with pytest.raises(ValueError, match="malformed_reference_placeholder"):
        bind_generated_source("Reference: " + token, context)


@pytest.mark.parametrize("mapping", [
    {"bad": "https://example.test"}, {"[URL_1]": "[URL_2]"},
    {"[URL_1]": {"url": "https://example.test"}}, {"[URL_1]": "not a resource"},
])
def test_invalid_donor_map_rejected(mapping):
    with pytest.raises(ValueError, match="invalid_reference_map"):
        build_reference_context("Donor", "candidate", donor_reference_map=mapping)


def test_synthetic_media_registry_is_reserved_optional_and_collision_free():
    context = build_reference_context("Media: Icon=https://example.test/donor.svg", "candidate", category="literal_media_references")
    entries = {item["role"]: item for item in context["synthetic_references"]}
    assert entries["icon"]["token"] != "[ICON_URL_1]"
    assert all(item["destination"].startswith("https://synthetic.invalid/") for item in entries.values())
    source = f"Synthetic illustration only; existence unverified.\nMedia: Image={entries['image']['token']}\nSource: {entries['source']['token']}"
    restored, evidence = bind_generated_source(source, context)
    assert "[IMAGE_URL_" not in restored
    assert evidence["synthetic_reference_count"] == 2
    assert entries["icon"]["destination"] not in restored  # An icon is not mandatory.
    assert context == build_reference_context("Media: Icon=https://example.test/donor.svg", "candidate", category="literal_media_references")
    assert context != build_reference_context("Media: Icon=https://example.test/donor.svg", "different", category="literal_media_references")


def test_synthetic_mock_action_binds_before_stage3_source_gate():
    context = build_reference_context("Donor", "candidate", category="forms_rare_controls")
    item = context["synthetic_references"][0]
    source = f"Synthetic mock form; no operation is performed.\nAction: [Button: Mock submit] {item['token']}"
    restored, evidence = bind_generated_source(source, context)
    assert "action://synthetic/" in restored
    assert evidence["synthetic_reference_count"] == 1
    masked, _, reverse = _mask_model_references(restored, [])
    assert item["destination"] not in masked
    graph = {"root": "b", "state": {}, "elements": {"b": {"type": "Button", "props": {"label": "Mock submit"},
        "on": {"press": {"action": "openUrl", "params": {"url": next(iter(reverse))}}}}}}
    assert reference_destination_errors(graph, reverse) == []
    assert _restore_model_references(graph, reverse)["elements"]["b"]["on"]["press"]["params"]["url"] == item["destination"]


@pytest.mark.parametrize("destination", ["app://details/item?id=2", "action://synthetic/mock", "app:mock", "action:mock"])
def test_canonical_internal_schemes_mask_and_restore(destination):
    source = "Action: [Button: Mock] " + destination
    masked, _, reverse = _mask_model_references(source, [])
    assert masked == "Action: [Button: Mock] [URL_1]"
    assert reverse == {"[URL_1]": destination}
    assert _restore_model_references(masked, reverse) == source
    assert_reference_safe_source(source)


def test_unrelated_category_has_no_synthetic_resources():
    assert build_reference_context("Donor", "candidate", category="coherent_numeric_entities")["synthetic_references"] == []


def test_invented_real_resource_rejected():
    context = build_reference_context("Donor", "candidate", category="literal_media_references")
    with pytest.raises(ValueError, match="undeclared_generated_reference"):
        bind_generated_source("Media: Icon=https://invented.example/icon.svg", context)


@pytest.mark.parametrize("suffix", ["/invented", "?redirect=other", "#fragment", "extra"])
def test_known_token_cannot_be_extended_into_new_destination(suffix):
    context = build_reference_context("Source: https://example.test/source", "candidate")
    with pytest.raises(ValueError, match="modified_generated_reference"):
        bind_generated_source("Source: [URL_1]" + suffix, context)


@pytest.mark.parametrize("declaration,role", [("Media: Icon={}", "source"), ("Media: Image={}", "icon"), ("Action: [Button: Open] {}", "image")])
def test_synthetic_role_swaps_rejected(declaration, role):
    context = build_reference_context("Donor", "candidate", category="literal_media_references")
    token = next(item["token"] for item in context["synthetic_references"] if item["role"] == role)
    with pytest.raises(ValueError, match="synthetic_reference_role_mismatch"):
        bind_generated_source(declaration.format(token), context)


@pytest.mark.parametrize("source,reason", [
    ("Action: [Button: Open] [ACTION_URL_1]", "unbound_reference_placeholder"),
    ("Action: [Button: Open] https://example.test/long...path", "truncated_http_destination"),
    ("Action: [Button: Open] javascript:alert(1)", "unsupported_destination_scheme"),
])
def test_reusable_source_cannot_bypass_reference_checks(source, reason):
    with pytest.raises(ValueError, match=reason):
        assert_reference_safe_source(source)
