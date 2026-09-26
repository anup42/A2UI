"""Live reference targets must bind exactly; quoted display text is inert."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.reference_boundaries import reference_destination_errors
from test_generation_quality_regressions import run_fake


@pytest.mark.parametrize("token", ["[URL_1...]", "[URL_999]", "[URL_1]/invented", "[url_1]"])
def test_malformed_or_unbound_action_token_cannot_be_accepted(tmp_path, monkeypatch, token):
    program = f'<a2ui>\nroot=Button("Details",onPress=openUrl("{token}"))\n</a2ui>'
    _, rows = run_fake(tmp_path, monkeypatch, [program], source="Action: [Button: Details] https://example.test/details")
    assert rows[0]["record_status"] == "format_rejected"
    assert "invalid_reference_destination" in str(rows[0]["validation"]["errors"])


def test_reference_repair_uses_existing_retry_path(tmp_path, monkeypatch):
    bad = '<a2ui>\nroot=Button("Details",onPress=openUrl("[URL_1...]"))\n</a2ui>'
    good = bad.replace("[URL_1...]", "[URL_1]")
    adapter, rows = run_fake(tmp_path, monkeypatch, [bad, good], repairs=1,
                            source="Action: [Button: Details] https://example.test/details")
    assert len(adapter.calls) == 2
    assert "invalid_reference_destination" in adapter.calls[1]["prompt"]
    assert rows[0]["record_status"] == "accepted"
    assert rows[0]["canonical_graph"]["elements"]["root"]["on"]["press"]["params"]["url"] == "https://example.test/details"


def test_reference_tokens_in_quoted_text_are_not_live_destinations():
    graph = {"elements": {"note": {"type": "Text", "props": {"text": "Example: [URL_999...]"}}}}
    assert reference_destination_errors(graph, {}) == []


def test_known_media_and_state_bound_references_are_checked():
    graph = {"elements": {"icon": {"type": "Icon", "props": {"url": "[ICON_URL_1]"}},
                          "photo": {"type": "Image", "props": {"url": "$/items/0/imageUrl"}}},
             "state": {"items": [{"imageUrl": "[IMAGE_URL_9]"}]}}
    errors = reference_destination_errors(graph, {"[ICON_URL_1]": "https://example.test/icon.svg"})
    assert errors == ["invalid_reference_destination:elements.photo.props.url"]


def test_inert_state_table_data_and_event_context_are_not_live_destinations():
    graph = {"state": {"url": "Quoted example [URL_999]"}, "elements": {
        "table": {"type": "Table", "props": {"rows": [{"url": "[URL_999]"}]}},
        "button": {"type": "Button", "on": {"press": {"action": "event", "params": {
            "context": {"url": "Quoted example [URL_999]"}}}}},
    }}
    assert reference_destination_errors(graph, {}) == []
