"""Regression tests for source-grounded messages-only archive recovery."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ir_training.data.archive_recovery import recover_cp437_utf8, recover_pair
from ir_training.data.ir_targets import A2UI_EXPRESS_V1, materialize_completion_targets
from ir_training.data.url_preprocess import restore_url_placeholders


def target(*, text="Reference details", url=None, label="Open details"):
    elements = {
        "root": {"type": "Stack", "props": {}, "children": ["text"]},
        "text": {"type": "Text", "props": {"text": text}, "children": []},
    }
    if url is not None:
        elements["root"]["children"].append("button")
        elements["button"] = {
            "type": "Button",
            "props": {"label": label},
            "children": [],
            "on": {"press": {"action": "openUrl", "params": {"url": url}}},
        }
    graph = {"root": "root", "state": {}, "elements": elements}
    return materialize_completion_targets(graph)[A2UI_EXPRESS_V1]


def corrupt(value):
    return value.encode("utf-8").decode("cp437")


def test_cp437_utf8_recovery_is_exactly_reversible():
    clean = "Café – 20°C and €50"
    damaged = corrupt(clean)
    result = recover_cp437_utf8(damaged)
    assert result.applied and result.text == clean
    assert result.text.encode("utf-8").decode("cp437") == damaged


def test_ascii_and_normal_unicode_are_not_changed():
    assert not recover_cp437_utf8("plain ASCII").applied
    assert not recover_cp437_utf8("normal café").applied


def test_source_identity_url_repair_is_closed_and_reversible():
    raw = "https://example.org/details"
    source = f"Reference details at {raw}"
    result = recover_pair(source, target(url=raw))
    assert result.accepted and result.reason == "repaired"
    assert "source_identity_url_normalization" in result.transformations
    assert set(result.url_map) <= set(result.source_text.split())
    assert restore_url_placeholders(result.source_text, result.url_map) == source
    restored = restore_url_placeholders(result.graph, result.url_map)
    button = next(
        item for item in restored["elements"].values() if item["type"] == "Button"
    )
    assert button["on"]["press"]["params"]["url"] == raw


def test_instructional_bracketed_host_uses_normal_content_gates():
    raw = "https://[Your-Public-IP]"
    source = f"Router details at {raw}\n[Button: Open router] {raw}"
    result = recover_pair(source, target(url=raw, label="Open router"))

    assert result.accepted and result.reason == "repaired"
    assert "source_identity_url_normalization" in result.transformations
    assert restore_url_placeholders(result.source_text, result.url_map) == source


def test_target_only_literal_reference_element_is_removed_not_invented():
    result = recover_pair("Reference details", target(url="https://example.org/unseen"))
    assert result.accepted and result.reason == "repaired"
    assert dict(result.repair_metrics)["ungrounded_reference_elements_removed"] == 1
    assert not result.url_map
    assert not any(
        item["type"] == "Button" for item in result.graph["elements"].values()
    )


def test_source_only_symbolic_reference_can_remain_unused():
    result = recover_pair("Reference details [SOURCE_URL_1]", target())
    assert result.accepted and result.reason == "kept_unchanged"
    assert not result.url_map


def test_target_only_symbolic_reference_element_is_removed_not_aliased():
    result = recover_pair("Reference details", target(url="[ACTION_URL_1]"))
    assert result.accepted and result.reason == "repaired"
    assert dict(result.repair_metrics)["ungrounded_reference_elements_removed"] == 1


def test_button_reference_rebinds_only_by_exact_source_label():
    source = "Sources\n- Official documentation: <[SOURCE_URL_1]>"
    result = recover_pair(
        source, target(url="[ACTION_URL_7]", label="Official documentation")
    )
    assert result.accepted
    button = next(
        item for item in result.graph["elements"].values() if item["type"] == "Button"
    )
    assert button["on"]["press"]["params"]["url"] == "[SOURCE_URL_1]"
    assert dict(result.repair_metrics)["placeholder_label_rebindings"] == 1


def test_media_namespace_rebind_requires_same_kind_and_index():
    graph = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["text", "icon"]},
            "text": {"type": "Text", "props": {"text": "Achievement"}, "children": []},
            "icon": {
                "type": "Icon",
                "props": {"name": "[ICON_ASSET_1]"},
                "children": [],
            },
        },
    }
    completion = materialize_completion_targets(graph)[A2UI_EXPRESS_V1]
    result = recover_pair("Achievement\nMedia: Icon=[ICON_URL_1]", completion)
    assert result.accepted
    icon = next(
        item for item in result.graph["elements"].values() if item["type"] == "Icon"
    )
    assert icon["props"]["name"] == "[ICON_URL_1]"
    assert dict(result.repair_metrics)["placeholder_media_namespace_rebindings"] == 1


def test_unique_unmatched_same_kind_media_token_is_rebound_in_visible_text():
    result = recover_pair(
        "Save the report at [MEDIA_ASSET_1]",
        target(text="Save the report at [MEDIA_ASSET_2]"),
    )

    assert result.accepted and result.reason == "repaired"
    assert "[MEDIA_ASSET_1]" in result.target_text
    assert "[MEDIA_ASSET_2]" not in result.target_text
    assert dict(result.repair_metrics)["placeholder_unique_media_rebindings"] == 1


def test_ambiguous_unmatched_media_tokens_remain_quarantined():
    result = recover_pair(
        "Compare [MEDIA_ASSET_1] with [MEDIA_ASSET_2]",
        target(text="Compare [MEDIA_ASSET_3]"),
    )

    assert not result.accepted
    assert result.reason == "unresolved_target_reference"


def test_declared_media_cannot_disappear_from_an_otherwise_valid_target():
    result = recover_pair("Summary\nMedia: Image=[IMAGE_URL_1]", target(text="Summary"))
    assert not result.accepted
    assert result.reason == "unresolved_content_warning"
    assert "missing_declared_media" in result.warnings


def test_generic_fallback_open_source_button_is_removed_unless_requested():
    result = recover_pair(
        "Reference details\nSource: <[SOURCE_URL_1]>",
        target(url="[SOURCE_URL_1]", label="Open Source"),
    )
    assert result.accepted
    assert not any(
        item["type"] == "Button" for item in result.graph["elements"].values()
    )
    assert dict(result.repair_metrics)["generic_open_source_buttons_removed"] == 1


def test_unsupported_state_media_field_and_empty_table_column_are_removed():
    graph = {
        "root": "root",
        "state": {"rows": [{"name": "Item A", "image": "[IMAGE_URL_9]"}]},
        "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["table"]},
            "table": {
                "type": "Table",
                "props": {
                    "columns": [
                        {"key": "name", "label": "Name"},
                        {"key": "image", "label": "Image"},
                    ],
                    "statePath": "/rows",
                    "domain": "generic",
                    "preferredPresentation": "table",
                },
                "children": [],
            },
        },
    }
    result = recover_pair(
        "Item A", materialize_completion_targets(graph)[A2UI_EXPRESS_V1]
    )
    assert result.accepted
    assert result.graph["state"]["rows"] == [{"name": "Item A"}]
    table = next(
        item for item in result.graph["elements"].values() if item["type"] == "Table"
    )
    assert [column["key"] for column in table["props"]["columns"]] == ["name"]
    assert dict(result.repair_metrics)["ungrounded_reference_fields_removed"] == 1
    assert dict(result.repair_metrics)["empty_table_columns_removed"] == 1


def test_empty_non_root_layout_branches_are_removed_without_dropping_the_row():
    graph = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["text", "empty"]},
            "text": {
                "type": "Text",
                "props": {"text": "Useful content"},
                "children": [],
            },
            "empty": {
                "type": "Stack",
                "props": {"direction": "vertical", "gap": "sm"},
                "children": [],
            },
        },
    }
    result = recover_pair(
        "Useful content", materialize_completion_targets(graph)[A2UI_EXPRESS_V1]
    )
    assert result.accepted
    assert "empty" not in result.graph["elements"]
    children = result.graph["elements"][result.graph["root"]]["children"]
    assert (
        len(children) == 1 and result.graph["elements"][children[0]]["type"] == "Text"
    )
    assert dict(result.repair_metrics)["empty_layout_elements_removed"] == 1


def test_empty_root_layout_is_not_removed_and_remains_quarantined():
    graph = {
        "root": "root",
        "state": {},
        "elements": {"root": {"type": "Stack", "props": {}, "children": []}},
    }
    result = recover_pair(
        "Substantive requested content",
        materialize_completion_targets(graph)[A2UI_EXPRESS_V1],
    )
    assert not result.accepted
    assert result.reason == "unresolved_content_warning"
    assert "layout_only" in result.warnings


def test_legacy_midword_text_boundary_is_joined_without_changing_characters():
    left = "A" * 180
    right = "broken word continuation"
    graph = {
        "root": "root",
        "state": {},
        "elements": {
            "root": {"type": "Stack", "props": {}, "children": ["left", "right"]},
            "left": {
                "type": "Text",
                "props": {"text": left, "variant": "body"},
                "children": [],
            },
            "right": {
                "type": "Text",
                "props": {"text": right, "variant": "body"},
                "children": [],
            },
        },
    }
    result = recover_pair(
        left + right, materialize_completion_targets(graph)[A2UI_EXPRESS_V1]
    )
    assert result.accepted
    texts = [
        item["props"]["text"]
        for item in result.graph["elements"].values()
        if item["type"] == "Text"
    ]
    assert texts == [left + right]
    assert dict(result.repair_metrics)["midword_text_boundary_joins"] == 1


def test_mixed_pretokenized_and_literal_references_are_quarantined():
    raw = "https://example.org/details"
    result = recover_pair(f"Reference details [SOURCE_URL_1] {raw}", target(url=raw))
    assert (
        not result.accepted and result.reason == "mixed_symbolic_and_literal_references"
    )


def test_missing_named_action_is_not_repaired_by_guessing():
    result = recover_pair(
        "Reference details\nAction: [Button: Download report]", target()
    )
    assert not result.accepted
    assert result.reason == "unresolved_content_warning"
    assert "missing_named_action" in result.warnings
