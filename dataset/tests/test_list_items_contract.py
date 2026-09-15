from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.ir_formats import decode_express_completion, encode_express_completion
from pipeline.list_items import validate_list_items
import pytest


def test_list_literals_round_trip_without_losing_instructions():
    raw = '<a2ui>root=List(items=["First instruction","Second instruction"])</a2ui>'
    graph = decode_express_completion(raw)
    assert graph["elements"]["root"]["props"]["items"] == ["First instruction", "Second instruction"]
    assert decode_express_completion(encode_express_completion(graph)) == graph


def test_list_source_objects_preserve_all_visible_fields():
    assert validate_list_items([{"label": "Manual", "description": "Details", "url": "[SOURCE_URL_1]"}]) is None


def test_opaque_component_calls_in_list_data_are_rejected():
    with pytest.raises(ValueError, match="use children"):
        decode_express_completion('<a2ui>root=List(items=[Text("Hidden instruction")])</a2ui>')
    valid = decode_express_completion('<a2ui>root=List(children=[Text("Visible instruction")])</a2ui>')
    assert len(valid["elements"]["root"]["children"]) == 1


@pytest.mark.parametrize("items", ["not-an-array", [3], [{"label": "Name", "secretField": "lost"}], [{"text": ["nested"]}]])
def test_unsupported_list_data_fails_instead_of_silently_losing_content(items):
    assert validate_list_items(items)
