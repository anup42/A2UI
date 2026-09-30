from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training/src'))
from ir_training.data.express_preparation import PreparationError, TASK_PREFIX, prepare_row, prepare_splits
from ir_training.data.release_review import reviewed_hold


def row(completion, source='A: 12'):
    return {
        'id': 'fixture', 'source_id': 'fixture', 'response_text': source,
        'completion': completion, 'target_format': 'a2ui_express_v1',
        'messages': [{'role': 'system', 'content': 'fixture'},
                     {'role': 'user', 'content': TASK_PREFIX + source},
                     {'role': 'assistant', 'content': completion}],
        'metadata': {'space_import': {'run': 'fixture'}},
    }


@pytest.mark.parametrize('subtype', ['line', 'pie', 'stackedbar', 'scatter', 'groupedbar', 'area', 'combo', 'radar'])
def test_unsupported_charts_quarantined_without_rewriting(subtype):
    candidate = row(f'<a2ui>\nroot=Chart(chartType="{subtype}",rows=[{{label:"A",value:12}}])\n</a2ui>')
    original = deepcopy(candidate)
    with pytest.raises(PreparationError) as failure:
        prepare_row(candidate, 'root-first')
    assert failure.value.reason == 'unsupported_renderer_chart'
    assert candidate == original


@pytest.mark.parametrize('subtype', ['bar', 'column'])
def test_renderer_supported_charts_remain_eligible(subtype):
    candidate = row(f'<a2ui>\nroot=Chart(chartType="{subtype}",rows=[{{label:"A",value:12}}])\n</a2ui>')
    prepared, _, _ = prepare_row(candidate, 'root-first')
    assert prepared['response_text'] == candidate['response_text']


@pytest.mark.parametrize('subtype', ['bar', 'column', 'bar_chart', ' BAR ', ''])
def test_chart_subtype_bindings_resolve_to_renderer_supported_value(subtype):
    candidate = row('<a2ui>\n$/={kind:' + json.dumps(subtype) + '}\nroot=Chart(chartType="${/kind}",columns=["Label","Value"],rows=[["A",12]])\n</a2ui>')
    prepared, _, _ = prepare_row(candidate, 'root-first')
    assert prepared['response_text'] == candidate['response_text']


def test_bound_unsupported_chart_is_held():
    candidate = row('<a2ui>\n$/={kind:"radar"}\nroot=Chart(chartType="${/kind}",columns=["Label","Value"],rows=[["A",12]])\n</a2ui>')
    with pytest.raises(PreparationError) as failure:
        prepare_row(candidate, 'root-first')
    assert failure.value.reason == 'unsupported_renderer_chart'


def test_exact_reviewed_target_held_but_regenerated_target_not_blanket_blocked():
    catalog = json.loads((ROOT / 'training/data/quality/v11_review_holds_20261001.json').read_text())
    known = catalog['join_rows'][0]
    candidate = {'source_id': known['source_id'], 'metadata': {'space_import': {}}}
    graph = {'elements': {'root': {'type': 'Text', 'props': {'text': 'fixture'}}}}
    assert reviewed_hold(candidate, graph, known['semantic_sha256'])[0] == 'reviewed_text_boundary_hold'
    assert reviewed_hold(candidate, graph, 'f' * 64) is None
    assert reviewed_hold({'source_id': 'independent', 'metadata': {'space_import': {}}}, graph, known['semantic_sha256']) is None


def test_normal_preparation_records_release_holds_and_policy_hash(tmp_path):
    good = row('<a2ui>\nroot=Text("A: 12")\n</a2ui>')
    bad = row('<a2ui>\nroot=Chart(chartType="line",rows=[{label:"A",value:12}])\n</a2ui>')
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps(good) + '\n' + json.dumps(bad) + '\n')
    destination = tmp_path / 'prepared'
    manifest = prepare_splits({'train': source}, destination)
    stats = manifest['splits']['train']
    assert stats['accepted_rows'] == 1
    assert stats['quarantined_rows'] == 1
    assert stats['quarantine_reasons'] == {'unsupported_renderer_chart': 1}
    assert 'training/data/quality/v11_review_holds_20261001.json' in manifest['implementation_sha256']
    evidence = json.loads((destination / 'quarantine.jsonl').read_text())
    assert evidence['row'] == bad
