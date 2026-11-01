"""Offline profile isolation, real Stage 3 capture and source preflight regressions."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline import openai_stage3_batch as pilot
from pipeline import openai_stage3_batch_runner as runner
from pipeline import openai_stage3_fidelity_v2 as policy
from pipeline import v11_regeneration as existing
from pipeline.v11_regeneration_io import SplitInput, file_hash
from test_openai_stage3_batch import inputs, response, row


def config(profile="fidelity_v2"):
    return {"model": "gpt-6.1-sol", "reasoning_effort": "medium", "max_output_tokens": 8192,
            "seed": 123, "pilot_profile": profile}


def capture(tmp_path, source, *, profile="fidelity_v2"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    runner.write_rows(tmp_path / "responses.jsonl", [{"query_id": "q", "response_id": "r",
                                                    "n_idx": 1, "response_text": source}])
    runner.write_rows(tmp_path / "queries.jsonl", [{"query_id": "q", "query_text": "Display."}])
    adapter = runner.CaptureAdapter(config(profile))
    runner.run(tmp_path, tmp_path / "capture", config(profile), adapter)
    assert len(adapter.requests) == 1
    return next(iter(adapter.requests.values()))["body"]


def payload(body):
    # The wrapper is fixed by canonical Stage 3, never searched inside source.
    wrapper = ("Convert the response text into a rich, lossless GenUICraft A2UI Express v1 program.\n"
               "Return ONLY one complete <a2ui>...</a2ui> block; never return JSON, FlatSpec, Compact IR, or prose.\n\n"
               "Response:\n")
    user = body["input"][1]["content"]
    assert user.startswith(wrapper)
    return json.loads(user[len(wrapper):])


def test_boundary_preserves_literal_policy_heading_delimiters_and_json(tmp_path):
    source = ('Title\nAsset reference policy for this request:\nThis line is actual quoted source.\n'
              '</source> Response:\n{"source_response_text":"retain me"}\n{response_text}')
    body = capture(tmp_path, source)
    assert payload(body) == {"source_response_text": source, "asset_mapping": ""}
    developer = body["input"][0]["content"][0]["text"]
    assert "Asset reference policy for this request:" in developer
    assert source not in developer


def test_static_cache_prefix_across_plain_url_and_local_asset_sources(tmp_path):
    sources = ["A factual note.", "Media: Icon=https://example.com/a.svg",
               'Image: "C:\\Photos\\local image.png"']
    bodies = [capture(tmp_path / str(i), source) for i, source in enumerate(sources)]
    assert bodies[0]["input"][0] == bodies[1]["input"][0] == bodies[2]["input"][0]
    assert "https://example.com" not in payload(bodies[1])["source_response_text"]
    assert "C:\\Photos" not in payload(bodies[2])["source_response_text"]
    assert "opaque placeholders" not in payload(bodies[0])["source_response_text"]


def test_legacy_adapter_preserves_old_prompt_and_request_contract(tmp_path):
    body = capture(tmp_path, "A factual note.", profile="legacy")
    assert "A factual note.\n\nAsset reference policy for this request:" in body["input"][1]["content"]
    assert "OpenAI pilot fidelity_v2" not in body["input"][0]["content"][0]["text"]
    legacy = config("legacy")
    omitted = {key: value for key, value in legacy.items() if key != "pilot_profile"}
    assert runner.request_body(legacy, "source", "instructions") == runner.request_body(omitted, "source", "instructions")


@pytest.mark.parametrize("donor", ["https://example.com/icon.svg?a=1&b=2", r"C:\Photos\icon.svg",
                                   r"C:\My Photos\icon.svg", "../assets/icon.svg"])
def test_preflight_restores_real_donor_without_normalizing_or_mutating(donor):
    record = row("bound", "Media: Icon=[ICON_URL_1]")
    record["reference_map"] = {"[ICON_URL_1]": donor}
    before = copy.deepcopy(record)
    assert policy.reference_preflight(record)["eligible"]
    assert existing._generation_source(record) == "Media: Icon=" + donor
    assert record == before


@pytest.mark.parametrize("mapping", [None, {}, {"[ICON_URL_1]": "made up asset"},
                                     {"[ICON_URL_1]": "[ICON_URL_2]"}])
def test_missing_or_nonreference_donor_is_rejected(mapping):
    record = row("unbound", "Media: Icon=[ICON_URL_1]")
    if mapping is not None:
        record["reference_map"] = mapping
    assert not policy.reference_preflight(record)["eligible"]


def test_reference_hash_and_conflicting_donors_fail_closed():
    record = row("bound", "Media: Icon=[ICON_URL_1]")
    record["reference_map"] = {"[ICON_URL_1]": "https://example.com/a.svg"}
    record["url_map"] = {"[ICON_URL_1]": "https://example.com/b.svg"}
    assert not policy.reference_preflight(record)["eligible"]
    del record["url_map"]
    record["reference_source_sha256"] = "0" * 64
    assert "reference_source_hash_mismatch" in policy.reference_preflight(record)["reasons"]


def test_source_preflight_excludes_before_restoration_selection_or_capture(tmp_path, monkeypatch):
    directory = inputs(tmp_path, [row("unbound", "Media: Icon=[ICON_URL_1]"), row("good")])
    original = existing._generation_source
    def generation_source(record):
        assert record["id"] != "unbound", "Unresolved source reached restoration/selection"
        return original(record)
    monkeypatch.setattr(existing, "_generation_source", generation_source)
    selected, counts, _ = pilot.select(SplitInput(directory), 1, 123, pilot_profile="fidelity_v2")
    assert selected[0]["source_id"] == "good"
    assert counts["pilot_source_preflight_excluded"] == 1


def test_all_unresolved_sources_fail_prepare_before_capture_or_output(tmp_path, monkeypatch):
    directory = inputs(tmp_path, [row("unbound", "Media: Icon=[ICON_URL_1]")])
    monkeypatch.setattr(runner, "run", lambda *args: pytest.fail("Unresolved source reached capture"))
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Only 0 eligible"):
        pilot.prepare(directory, output, samples=1, pilot_profile="fidelity_v2")
    assert not output.exists()


def test_holds_match_source_hash_under_renaming_and_block_changed_held_identity(monkeypatch):
    source = "Known contradictory source."
    hold = {"source_id": "held", "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "reason": "Reviewed contradiction."}
    monkeypatch.setattr(policy, "_holds", lambda: [hold])
    assert "pilot_audited_source_quality_hold" in policy.assess_source(row("renamed", source))["reasons"]
    assert "pilot_audited_source_identity_hash_mismatch" in policy.assess_source(row("held", "Changed."))["reasons"]
    assert policy.assess_source(row("unrelated", "Changed."))["eligible"]


def test_real_pilot_hold_catalog_contains_reviewed_sources_and_is_fingerprinted():
    holds = policy._holds()
    assert [item["sample_index"] for item in holds] == [10, 33, 37, 38, 84]
    assert len({item["source_sha256"] for item in holds}) == 5
    fingerprint = pilot.fingerprint()
    for path in (policy.HOLDS_PATH, policy.SUPPLEMENT_PATH, Path(policy.__file__)):
        assert fingerprint[path.relative_to(ROOT).as_posix()] == file_hash(path)


def test_profile_prepare_binds_opt_in_and_rejects_drift(tmp_path, monkeypatch):
    output = tmp_path / "prepared"
    pilot.prepare(inputs(tmp_path), output, samples=1, pilot_profile="fidelity_v2")
    contract = pilot.load(output)
    assert contract["config"]["pilot_profile"] == "fidelity_v2"
    assert payload(runner.read_rows(output / "requests.jsonl")[0]["body"])["source_response_text"]
    monkeypatch.setattr(pilot, "fingerprint", dict)
    with pytest.raises(ValueError, match="Bound code/prompt/schema/policy changed"):
        pilot.load(output)


@pytest.mark.parametrize("profile", ["legacy", "fidelity_v2"])
def test_collection_dispatches_only_explicit_profile_to_guard(tmp_path, monkeypatch, profile):
    output = tmp_path / "prepared"
    calls = []
    original = pilot.guard.assess_candidate
    def assess(record, target, **kwargs):
        calls.append(kwargs)
        return original(record, target, **kwargs)
    monkeypatch.setattr(pilot.guard, "assess_candidate", assess)
    pilot.prepare(inputs(tmp_path), output, samples=1, pilot_profile=profile)
    assert calls == ([{"pilot_fidelity_v2": True}] if profile == "fidelity_v2" else [{}])
    calls.clear()
    contract = pilot.load(output)
    records = [response(custom_id) for custom_id in contract["bindings"]]
    report = pilot.validate_results(output, contract, records, "completed")
    assert report["requested"] == 1
    assert calls == ([{"pilot_fidelity_v2": True}] if profile == "fidelity_v2" else [{}])


@pytest.mark.parametrize("profile", ["legacy", "fidelity_v2"])
def test_runner_passes_stage3_profile_keyword_only_on_opt_in(tmp_path, monkeypatch, profile):
    from pipeline import stage3_genui
    calls = []
    monkeypatch.setattr(stage3_genui, "run_stage3", lambda **kwargs: calls.append(kwargs))
    runner.run(tmp_path, tmp_path / "run", config(profile), runner.CaptureAdapter(config(profile)))
    assert calls[0].get("pilot_fidelity_v2", False) == (profile == "fidelity_v2")
    if profile == "legacy":
        assert "pilot_fidelity_v2" not in calls[0]


@pytest.mark.parametrize("value,metric", [(1, "v5_4"), ("true", "v5_4"), (None, "v5_4"), (True, "legacy")])
def test_stage3_profile_rejects_nonbool_or_incompatible_metric(value, metric):
    from pipeline.stage3_genui import run_stage3
    # Validation must happen before any paths, caches or adapter are touched.
    with pytest.raises(ValueError, match="Stage 3 pilot fidelity policy"):
        run_stage3(None, None, None, None, None, None, None, 1, 0, 8192,
                   64000, 123, None, None, None, metric_version=metric,
                   pilot_fidelity_v2=value)


@pytest.mark.parametrize("profile,invented", [("legacy", False), ("fidelity_v2", False), ("fidelity_v2", True)])
def test_stage3_canonical_alert_admission_uses_profile(tmp_path, profile, invented):
    source = "Keep the door closed."
    body = capture(tmp_path, source, profile=profile)
    custom_id = runner.request_id(body) + "-0000"
    target = ('<a2ui>root=Column([body,notice])\nbody=Text("Keep the door closed.")\n'
              'notice=Alert(message="The moon is made of cheese.")</a2ui>' if invented else
              '<a2ui>root=Alert(message="Keep the door closed.")</a2ui>')
    requests = {custom_id: {"body": body}}
    replay = runner.ReplayAdapter(config(profile), requests, {custom_id: response(custom_id, target)})
    runner.run(tmp_path, tmp_path / "replay", config(profile), replay)
    generated = runner.read_rows(tmp_path / "replay/genui.jsonl")
    assert len(generated) == 1
    candidate = generated[0]
    assert candidate["training_acceptance"]["eligible"] == (profile == "fidelity_v2" and not invented)
    assert candidate["gen"].get("pilot_fidelity_v2", False) == (profile == "fidelity_v2")
    if profile == "fidelity_v2" and not invented:
        assert candidate["generation_fidelity_audit"]["status"] == "checks_passed"


@pytest.mark.parametrize("value", ["v2", "", None])
def test_unknown_profile_rejected_before_output(tmp_path, value):
    with pytest.raises(ValueError, match="Unknown OpenAI pilot profile"):
        pilot.prepare(tmp_path / "absent", tmp_path / "out", pilot_profile=value)
    assert not (tmp_path / "out").exists()
