import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'dataset/src'))
from pipeline.genui_quality import generation_reward_a2ui_express_v1
from pipeline.renderer_effective_semantics_v5_4 import effective_chart, output_table_from_chart

FIXTURES = json.loads((ROOT / 'dataset/tests/fixtures/chart_contract_v2_1.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('fixture', FIXTURES, ids=lambda f: f['name'])
def test_android_python_shared_contract(fixture):
    chart = effective_chart(fixture['props'], fixture['state'])
    expected = fixture['expected']
    assert chart.complete == expected['complete'], chart.diagnostics
    if 'diagnostic' in expected:
        assert expected['diagnostic'] in chart.diagnostics
    if 'series_values' in expected:
        assert [s['values'] for s in chart.series] == expected['series_values']
    if 'x_values' in expected:
        assert chart.x_values == tuple(expected['x_values'])
    if 'sizes' in expected:
        assert chart.sizes == tuple(expected['sizes'])
    if 'axes' in expected:
        assert [s['axis'] for s in chart.series] == expected['axes']
    assert (output_table_from_chart('chart', chart) is not None) == chart.complete


def test_all_grouped_series_contribute_to_fidelity_evidence():
    chart = effective_chart({'chartType': 'stackedbar', 'columns': ['day', 'solar', 'wind', 'hydro'],
                             'rows': [['Monday', 45, 120, 200], ['Tuesday', 50, 140, 180]]}, {})
    table = output_table_from_chart('chart', chart)
    assert table.keys == ['day', 'solar', 'wind', 'hydro']
    assert table.rows[0] == {'day': 'Monday', 'solar': '45', 'wind': '120', 'hydro': '200'}


@pytest.mark.parametrize('types', [('combo', 'line'), ('line', 'combo')])
def test_incomplete_repeated_chart_cannot_hide_behind_valid_instance(types):
    completion = ('<a2ui>\n$/items=[{kind:"' + types[0] + '"},{kind:"' + types[1] + '"}]\n'
                  'root=Column([],repeat={statePath:"/items",template:c})\n'
                  'c=Chart(chartType="${$item.kind}",columns=["Month","Value"],rows=[["Jan",7]])\n</a2ui>')
    result = generation_reward_a2ui_express_v1(completion, '| Month | Value |\n|---|---|\n| Jan | 7 |')
    assert result.evidence['type_contract']['per_element']['c'] == 0
    assert 'renderer_component_contract' in result.evidence['training_acceptance']['blocking_reasons']


def test_invalid_chart_does_not_claim_hidden_values_as_visible():
    chart = effective_chart({'chartType': 'scatter', 'rows': [['nonnumeric', 991]]}, {})
    assert not chart.complete
    assert output_table_from_chart('chart', chart) is None


def test_pie_data_key_preserves_ancillary_share_and_caveat():
    chart = effective_chart({'chartType': 'pie', 'columns': ['category', 'revenue', 'share', 'caveat'],
                             'rows': [['A', 100, '40%', 'estimated'], ['B', 150, '60%', 'confirmed']]}, {})
    assert chart.complete
    assert len(chart.series) == 1
    table = output_table_from_chart('chart', chart)
    assert table.rows[0]['share'] == '40%'
    assert table.rows[0]['caveat'] == 'estimated'


def test_new_series_props_survive_real_training_serialization():
    sys.path.insert(0, str(ROOT / 'training/src'))
    from ir_training.data.express_preparation import prepare_row, TASK_PREFIX
    from pipeline.ir_formats.active import decode_express_completion
    source = 'W1: study 8 h, quiz 78%. W2: study 10 h, quiz 90%.'
    completion = '<a2ui>\nroot=Chart("combo",["week","hours","quiz"],rows=[["W1",8,78],["W2",10,90]],xKey="week",series=[{yKey:"hours",type:"column",axis:"left",unit:"h"},{yKey:"quiz",type:"line",axis:"right",unit:"%"}])\n</a2ui>'
    row = {'id': 'fixture', 'source_id': 'fixture', 'response_text': source,
           'completion': completion, 'target_format': 'a2ui_express_v1',
           'messages': [{'role': 'system', 'content': 'fixture'}, {'role': 'user', 'content': TASK_PREFIX + source},
                        {'role': 'assistant', 'content': completion}], 'metadata': {'space_import': {'run': 'fixture'}}}
    prepared, _, _ = prepare_row(row, 'root-first')
    graph = decode_express_completion(prepared['completion'])
    chart = effective_chart(graph['elements'][graph['root']]['props'], graph.get('state', {}))
    assert chart.complete
    assert [s['axis'] for s in chart.series] == ['left', 'right']
