from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.reference_boundaries import split_reference_suffix
from pipeline.stage3_genui import _mask_model_references, _restore_model_references


@pytest.mark.parametrize(
    ("raw", "reference", "suffix"),
    [
        ("https://example.test/Anastasia_(1997_film)", "https://example.test/Anastasia_(1997_film)", ""),
        ("https://example.test/Title_(part_(2))).", "https://example.test/Title_(part_(2))", ")."),
        ("https://example.test/path).", "https://example.test/path", ")."),
        ("https://example.test/path],", "https://example.test/path", "],"),
        ("https://example.test/path?x=(2)", "https://example.test/path?x=(2)", ""),
        ("https://example.test/%28title%29.", "https://example.test/%28title%29", "."),
    ],
)
def test_reference_suffix_keeps_balanced_delimiters(raw, reference, suffix):
    assert split_reference_suffix(raw) == (reference, suffix)


def test_stage3_reference_map_preserves_destination_not_only_prose_roundtrip():
    url = "https://en.wikipedia.org/wiki/Anastasia_(1997_film)"
    source = f"Action: [Button: View details] {url}\nSource: [Film]({url})."
    masked, forward, reverse = _mask_model_references(source, [])
    assert reverse == {"[URL_1]": url}
    assert forward == {url: "[URL_1]"}
    assert _restore_model_references(masked, reverse) == source
    graph = {"elements": {"button": {"props": {"action": "open_url", "url": "[URL_1]"}}}}
    assert _restore_model_references(graph, reverse)["elements"]["button"]["props"]["url"] == url


def test_quoted_local_asset_preserves_terminal_parenthesis():
    path = "C:/device assets/flight (final)"
    source = f"Use '{path}'."
    masked, forward, reverse = _mask_model_references(source, [])
    assert path in forward
    assert _restore_model_references(masked, reverse) == source
