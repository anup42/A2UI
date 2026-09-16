"""CPU-only regression checks. Every provider call uses a scripted fake."""
import hashlib
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm.base import BaseLLMAdapter, LLMResult, ModelSpec, split_reasoning_from_text
from llm.local_adapter import LocalAdapter
from pipeline.cache import PromptCache
from pipeline.generation_audit import compact_exception, graph_acceptance_errors
from pipeline.ir_formats import A2UI_EXPRESS_V1, decode_express_completion, semantic_hash
from pipeline.stage3_genui import run_stage3, _restore_model_references
from utils.rate_limit import RateLimiter
from utils.versioning import build_run_manifest, write_run_manifest
from main import _apply_generation_env_overrides

VALID = '<a2ui>\nroot=Text("- Fit: comfortable")\n</a2ui>'


class ScriptedAdapter(BaseLLMAdapter):
    def __init__(self, outputs):
        super().__init__(ModelSpec('fake', 'fake', 'scripted'))
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        text, reason = output if isinstance(output, tuple) else (output, 'stop')
        index = len(self.calls)
        return LLMResult(text=text, raw={'choices':[{'finish_reason':reason}]},
                         latency_ms=index * 2, input_tokens=index * 10, output_tokens=index * 20,
                         cost_usd=None, model='scripted', provider='fake')


def run_fake(tmp_path, monkeypatch, outputs, *, repairs=0, regens=0, budget=None, source=None, metric='legacy', source_quality=None):
    monkeypatch.setenv('STAGE3_FINAL_REGEN_ATTEMPTS', str(regens))
    monkeypatch.setenv('DATASET_OFFLINE_MODE', '1')
    monkeypatch.setattr('pipeline.stage3_genui._auto_download_response_assets', lambda **kw: pytest.fail('Stage3 must not download assets'))
    response = {'response_id':'r1', 'query_id':'q1', 'n_idx':1,
                'response_text': source or 'Fit: comfortable',
                'source_quality': source_quality or {'status':'needs_review'}, 'query_quality': {'status':'checks_passed'},
                'scenario_family_id': 'family-1'}
    responses = tmp_path/'responses.jsonl'; responses.write_text(json.dumps(response)+'\n',encoding='utf-8')
    queries = tmp_path/'queries.jsonl'; queries.write_text(json.dumps({'query_id':'q1','intent':'generic'})+'\n',encoding='utf-8')
    adapter = ScriptedAdapter(outputs)
    output = tmp_path/'genui.jsonl'
    run_stage3(queries_path=queries,responses_path=responses,
               prompt_path=ROOT/'prompts/genui_gen_mobile_a2ui_express_v1.md',adapter=adapter,
               genui_path=output,schema_path=ROOT/'schema/canonical_ui_graph_v1.schema.json',
               artifacts_dir=tmp_path/'artifacts',candidates_per_response=1,max_repair_attempts=repairs,
               max_tokens=2048,prompt_max_tokens=budget,seed=4,rate_limiter=RateLimiter(0),
               cache=PromptCache(tmp_path/'cache',enabled=False),logger=logging.getLogger('mock-generation'),
               batch_size=1,max_attempts=1,metric_version=metric,ir_formats=[A2UI_EXPRESS_V1],phase_invocation_id="test-phase")
    rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()] if output.exists() else []
    return adapter, rows


def test_compact_email_and_target_alignment(tmp_path, monkeypatch):
    program = '<a2ui>\nroot=EmailPreview("Meeting",["Please attend at 10 AM."],from="Alex")\n</a2ui>'
    adapter, rows = run_fake(tmp_path,monkeypatch,[program])
    assert rows[0]['record_status']=='accepted'
    assert len(rows[0]['canonical_graph']['elements']) == 1
    assert rows[0]['canonical_graph'] == decode_express_completion(program)
    assert rows[0]['source_quality']=={'status':'needs_review'}
    assert rows[0]['scenario_family_id']=='family-1'


def test_empty_email_is_rejected(tmp_path,monkeypatch):
    _, rows=run_fake(tmp_path,monkeypatch,['<a2ui>\nroot=EmailPreview("Meeting",[])\n</a2ui>'])
    assert rows[0]['record_status']=='format_rejected'
    assert 'incomplete_role' in str(rows[0]['validation']['errors'])


def test_all_renderer_edges_reachable_and_orphans_rejected():
    graph={'root':'root','elements':{'root':{'type':'Modal','props':{'trigger':'button','content':'tabs'}},
          'button':{'type':'Button','props':{'label':'Open'}},
          'tabs':{'type':'Tabs','props':{'tabs':[{'child':'details'}]}},
          'details':{'type':'Text','props':{'text':'Details'}}}}
    assert graph_acceptance_errors(graph)==[]
    graph['elements']['orphan']={'type':'Text','props':{'text':'Lost'}}
    assert graph_acceptance_errors(graph)==['unreachable_components: orphan']


def test_retry_ledger_full_source_and_selected_usage(tmp_path,monkeypatch):
    source='Unique source\n- Fit: comfortable\n- Keep the complete second item.'
    adapter,rows=run_fake(tmp_path,monkeypatch,['invalid','still invalid',VALID],repairs=1,regens=1,source=source)
    row=rows[0]
    assert row['record_status']=='accepted'
    assert source in adapter.calls[1]['prompt'] and source in adapter.calls[2]['prompt']
    assert row['validation']['repair_attempts']==2
    assert row['validation']['schema_repair_attempts']==1
    assert row['validation']['regeneration_attempts']==1
    assert row['gen']['input_tokens']==30 and row['gen']['output_tokens']==60
    assert row['generation_totals']['input_tokens']==60 and row['generation_totals']['output_tokens']==120
    assert row['generation_totals']['provider_call_count']==3
    assert row['phase_invocation_id']=='test-phase'
    assert all(attempt['phase_invocation_id']=='test-phase' for attempt in row['generation_attempts'])
    assert [a['phase'] for a in row['generation_attempts']]==['initial','schema_repair_1','regeneration_1']
    first=tmp_path/'artifacts'/row['generation_attempts'][0]['artifact']
    assert json.loads(first.read_text(encoding='utf-8'))['completion']=='invalid'
    assert row['canonical_graph'] == decode_express_completion(VALID)


def test_explicit_finish_length_cannot_accept_complete_looking_output(tmp_path,monkeypatch):
    _,rows=run_fake(tmp_path,monkeypatch,[(VALID,'length')])
    assert rows[0]['record_status']=='format_rejected'
    assert rows[0]['generation_attempts'][0]['completion_complete'] is False


def test_full_source_budget_fail_closed_before_model_call(tmp_path,monkeypatch):
    adapter,rows=run_fake(tmp_path,monkeypatch,[],budget=10,source='one\n- two https://example.test/a')
    assert adapter.calls==[] and rows==[]
    error=json.loads(next((tmp_path/'artifacts').glob('error_*.json')).read_text(encoding='utf-8'))
    assert 'source_budget_exceeded' in error['error']
    assert 'one\n- two' in error['prompt'] and 'https://' not in error['prompt']


def test_explicit_reference_map_binds_original_source_and_raw_target(tmp_path,monkeypatch):
    source='Open https://example.test/a'
    program='<a2ui>\nroot=Button("Open",onPress=openUrl("[URL_1]"))\n</a2ui>'
    _,rows=run_fake(tmp_path,monkeypatch,[program],source=source)
    row=rows[0]
    assert row['reference_map']=={'[URL_1]':'https://example.test/a'}
    assert row['reference_source_sha256']==hashlib.sha256(source.encode()).hexdigest()
    restored=_restore_model_references(decode_express_completion(program),row['reference_map'])
    assert row['canonical_graph']==restored and row['canonical_graph_hash']==semantic_hash(restored)
    assert row['asset_downloads_required_for_training'] is False


def test_concise_wire_diagnostic_has_property_path():
    from pipeline.ir_formats import compile_express_to_wire
    with pytest.raises(Exception) as captured:
        compile_express_to_wire('<a2ui>\nroot=Text("Hi",visible="bad")\n</a2ui>')
    message=compact_exception(captured.value)
    assert len(message)<=1200
    assert 'visible' in message


@pytest.mark.parametrize('reason,expected',[('length',False),('content_filter',False),('stop',True),(None,None)])
def test_http_finish_reason_is_captured_without_model(monkeypatch,reason,expected):
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self): return json.dumps({'choices':[{'message':{'content':'Final answer'},'finish_reason':reason}],
                                         'usage':{'prompt_tokens':3,'completion_tokens':4}}).encode()
    monkeypatch.setattr('llm.local_adapter.urlopen',lambda *args,**kwargs: Response())
    monkeypatch.delenv('LOCAL_VLLM_ENDPOINTS',raising=False)
    result=LocalAdapter(ModelSpec('fake','local','gemma',endpoint='http://unused.test/v1'))._http_generate(
        prompt='test',system=None,temperature=0.2,max_tokens=20,seed=1,json_mode=False)
    assert result.finish_reason==reason and result.completion_complete==expected
    assert bool(result.error)==(expected is False)
    assert result.input_tokens==3 and result.output_tokens==4


def test_reasoning_delimiters_keep_prose_and_hide_complete_thought_json():
    assert LocalAdapter._strip_thinking_text('Thinking: choose a route.')=='Thinking: choose a route.'
    reasoning,final=split_reasoning_from_text('<|think|>draft {"x":1} [2]<|end_think|>Final')
    assert reasoning=='draft {"x":1} [2]' and final=='Final'
    literal='Explain the literal <think>example</think> tags.'
    assert split_reasoning_from_text(literal)==(None,literal)


@pytest.mark.parametrize("thinking,send,expected", [
    ("0", "1", {"enable_thinking": False}),
    ("1", "1", {"enable_thinking": True}),
    ("0", "0", None),
    ("1", "0", None),
    (None, "1", None),
    ("", "1", None),
])
def test_http_thinking_control_is_explicit_when_enabled(monkeypatch, thinking, send, expected):
    requests = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "Final answer"}, "finish_reason": "stop"}]}).encode()
    def capture(request, **kwargs):
        requests.append(json.loads(request.data))
        return Response()
    monkeypatch.setattr("llm.local_adapter.urlopen", capture)
    if thinking is None:
        monkeypatch.delenv("LOCAL_VLLM_ENABLE_THINKING", raising=False)
    else:
        monkeypatch.setenv("LOCAL_VLLM_ENABLE_THINKING", thinking)
    monkeypatch.setenv("LOCAL_VLLM_SEND_CHAT_TEMPLATE_KWARGS", send)
    monkeypatch.delenv("LOCAL_VLLM_ENDPOINTS", raising=False)
    result = LocalAdapter(ModelSpec("fake", "local", "gemma", endpoint="http://unused.test/v1"))._http_generate(
        prompt="test", system=None, temperature=0.2, max_tokens=20, seed=1, json_mode=False)
    assert not result.error
    assert len(requests) == 1
    assert requests[0].get("chat_template_kwargs") == expected


def test_environment_overrides_clamp(monkeypatch):
    for key,value in {'A2UI_CALL_SLEEP_SECONDS':'-2','A2UI_STAGE1_INTENT_BATCH_SIZE':'0',
                      'A2UI_MAX_REPAIR_ATTEMPTS':'-4','A2UI_MAX_ATTEMPTS':'0'}.items(): monkeypatch.setenv(key,value)
    cfg={};_apply_generation_env_overrides(cfg)
    assert cfg=={'call_sleep_seconds':0.0,'stage1_intent_batch_size':1,'max_repair_attempts':0,'max_attempts':1}
    monkeypatch.setenv('A2UI_CALL_SLEEP_SECONDS','nan')
    with pytest.raises(SystemExit): _apply_generation_env_overrides({})


def test_reasoning_budget_is_not_shrunk_after_context_error(monkeypatch):
    monkeypatch.setenv("LOCAL_VLLM_MIN_RETRY_OUTPUT_TOKENS", "8192")
    message = "maximum context length is 16384 tokens (9000 in the messages)"
    assert LocalAdapter._context_retry_max_tokens(message, 8192) is None


def test_phase_manifest_is_immutable_and_records_effective_config(tmp_path,monkeypatch):
    monkeypatch.setenv('A2UI_CALL_SLEEP_SECONDS','0')
    monkeypatch.setenv('FAKE_API_KEY','must-never-appear')
    cfg=tmp_path/'run.yaml';cfg.write_text('run: {}\n',encoding='utf-8')
    paths=SimpleNamespace(run_dir=tmp_path,manifest_path=tmp_path/'run_manifest.json')
    first=build_run_manifest(tmp_path,'test',1,ModelSpec('fake','fake','fake'),paths,cfg,cfg,[],{'call_sleep_seconds':0})
    write_run_manifest(paths.manifest_path,first)
    frozen=json.loads(paths.manifest_path.read_text(encoding='utf-8'))
    snapshot=tmp_path/frozen['phase_manifest_path']; before=snapshot.read_bytes()
    second=build_run_manifest(tmp_path,'test',3,ModelSpec('fake','fake','fake'),paths,cfg,cfg,[],{})
    write_run_manifest(paths.manifest_path,second)
    assert snapshot.read_bytes()==before
    assert json.loads(paths.manifest_path.read_text(encoding='utf-8'))['stage']==3
    assert frozen['config']['effective_run_config']=={'call_sleep_seconds':0}
    assert frozen['runtime_overrides']['A2UI_CALL_SLEEP_SECONDS']=='0'
    assert 'must-never-appear' not in before.decode()


def test_prompt_has_typed_enums_and_quoted_placeholders():
    text=(ROOT/'prompts/genui_gen_mobile_a2ui_express_v1.md').read_text(encoding='utf-8')
    for fragment in ['Icon(url="[ICON_URL_1]")','wrap: "nowrap"|"wrap"','visible={path:"/consent_agreed"}',
                     'fewer than\n  five components','highlightColumns: array']:
        assert fragment in text

def test_failed_source_contract_skips_provider_with_explicit_rejection(tmp_path,monkeypatch):
    adapter,rows=run_fake(tmp_path,monkeypatch,[],source_quality={"status":"failed","training_eligibility":"exclude"})
    assert adapter.calls==[]
    assert rows[0]["record_status"]=="quality_rejected"
    assert rows[0]["validation"]["generation_attempted"] is False
    assert rows[0]["training_acceptance"]["blocking_reasons"]==["source_contract_failed"]


@pytest.mark.parametrize("blockers,review,expected",[(["renderer_missing_reference"],[],"quality_rejected"),([], ["possible_content_difference"],"accepted")])
def test_v5_reference_callers_and_acceptance_gate(tmp_path,monkeypatch,blockers,review,expected):
    import pipeline.stage3_genui as stage3
    captured=[]
    original_generation=stage3.generation_reward_a2ui_express_v1
    original_artifact=stage3.render_artifact_quality_v5_4
    def generation(*args,**kwargs):
        captured.append(kwargs.get("reference_map"))
        return original_generation(*args,**kwargs)
    def artifact(*args,**kwargs):
        captured.append(kwargs.get("reference_map"))
        result=original_artifact(*args,**kwargs)
        # Test the caller gate with deterministic injected evidence, independent
        # of the evaluator's own separate source-matching regressions.
        result.evidence["training_acceptance"]={"eligible":not blockers and not review,
            "blocking_reasons":blockers,"review_reasons":review}
        return result
    monkeypatch.setattr(stage3,"generation_reward_a2ui_express_v1",generation)
    monkeypatch.setattr(stage3,"render_artifact_quality_v5_4",artifact)
    _,rows=run_fake(tmp_path,monkeypatch,['<a2ui>\nroot=Button("Open",onPress=openUrl("[URL_1]"))\n</a2ui>'],
                    source="Open https://example.test/a",metric="v5_4")
    assert len(captured)>=2
    assert all(mapping=={"[URL_1]":"https://example.test/a"} for mapping in captured)
    assert rows[0]["record_status"]==expected
    assert rows[0]["training_acceptance"]["eligible"] is False
    assert rows[0]["validation"]["semantic_review_reasons"]==review
