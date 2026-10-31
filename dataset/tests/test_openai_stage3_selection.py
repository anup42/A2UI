"""Offline coverage-profile tests; production Stage 3 admission remains shared."""
from __future__ import annotations

import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "dataset/src"), str(REPO / "training/src")]

from pipeline import openai_stage3_batch as pilot
from pipeline import openai_stage3_batch_api as api
from pipeline import openai_stage3_batch_runner as runner
from pipeline import openai_stage3_selection as selection
from pipeline.v11_regeneration_io import ROOT, SplitInput, digest, file_hash
from test_openai_stage3_batch import SOURCE, inputs, row


def evidence(*tags):
    return {"tags": list(tags), "numbered_citation_markers": int("numbered_citations" in tags),
            "numeric_bracket_patterns": int("numbered_citations" in tags),
            "words": 300, "characters": 1800}


def quotas(**overrides):
    return {key: overrides.get(key, 1) for key in selection.QUOTAS}


def sample(items, *, requested=None, citation_target=1):
    sampler = selection.MixSampler(quotas(references=2) if requested is None else requested,
                                   citation_target=citation_target)
    for priority, identifier, tags in items:
        hints = evidence(*tags)
        sampler.observe(hints)
        if sampler.wants(priority, hints):
            sampler.offer(priority, {"source_id": identifier, "row": {"response_text": identifier}}, hints)
    return sampler.finish()


def test_exact_quota_and_overlapping_features_have_unique_primary_membership():
    items = [(1, "citation", ("numbered_citations", "references", "controls")),
             (2, "url", ("references", "numerical"))]
    items.extend((10 + i, bucket, tuple(dict.fromkeys((bucket, "numerical"))))
                 for i, bucket in enumerate(selection.QUOTAS) if bucket != "references")
    selected, report = sample(items)
    assert len(selected) == len({entry["source_id"] for entry in selected}) == 8
    assert Counter(entry["selection_bucket"] for entry in selected) == quotas(references=2)
    assert report["requested"] == report["achieved"] == quotas(references=2)
    assert not any(report["category_shortfalls"].values())
    assert not any(report["coverage_backfill"].values())
    assert report["numbered_citations"]["shortfall"] == 0
    assert all(entry["selection_replacement_for"] is None for entry in selected)


def test_pool_selection_deterministic_independent_of_candidate_order_and_bounded():
    requested = quotas(references=2)
    items = []
    for i in range(80):
        tags = tuple(selection.QUOTAS) + (("numbered_citations",) if i % 5 == 0 else ())
        items.append((int(digest({"seed": 123, "id": f"row-{i}"}), 16), f"row-{i}", tags))
    a = sample(items, requested=requested)
    b = sample(list(reversed(items)), requested=requested)
    assert a == b
    assert len(a[0]) == sum(requested.values())
    sampler = selection.MixSampler(requested, citation_target=1)
    for priority, identifier, tags in items:
        sampler.offer(priority, {"source_id": identifier}, evidence(*tags))
    assert all(len(pool) <= sampler.size for pool in sampler.pools.values())


@pytest.mark.parametrize("source", [
    "The result is read from the array entry arr[1].",
    "The result is read from the array entry `arr[1]`.",
    "```python\nThis looks like a sentence with a citation [1]\n```",
    "~~~text\nThis looks like a sentence with a citation [1]\n~~~",
    "The relevant court case was decided in the year [2010].",
    "| Row | Description |\n| --- | --- |\n| [1] | The first shipment requires careful handling |",
    "[1] Enter the applicant name into the first field.",
    "The original information was obtained from [SOURCE_URL_1].",
    "Use [ACTION_URL_1] and [ICON_URL_1] for the supplied references.",
    "Loop buttons: [1/4] [1/2] [1] [2]",
    "* Mid layer fleece or grid fleece: 2 [4]",
    "* Liner socks for ski boots: 3 pairs [6]",
])
def test_citation_detector_rejects_code_forms_years_and_reference_tokens(source):
    hints = selection.features(source)
    assert hints["numbered_citation_markers"] == 0
    assert "numbered_citations" not in hints["tags"]


def test_numeric_claim_citations_and_urls_remain_distinct_features():
    citation = "The measured annual rainfall was substantially higher.[1][2]"
    hints = selection.features(citation)
    assert hints["numbered_citation_markers"] == 2
    assert "numbered_citations" in hints["tags"]
    assert "references" not in hints["tags"]
    url_hints = selection.features("The measured annual rainfall was substantially higher. https://example.org/report")
    assert "references" in url_hints["tags"]
    assert "numbered_citations" not in url_hints["tags"]


def test_reference_detection_uses_restored_bindings_without_modifying_original_row():
    source_row = row("bound-reference", "Read the supplied guide at [SOURCE_URL_1].")
    source_row["metadata"]["url_preprocessing"] = {"url_map": {
        "[SOURCE_URL_1]": {"url": "https://example.org/guide"}}}
    before = deepcopy(source_row)
    assert "references" not in selection.features(source_row["response_text"])["tags"]
    assert "references" in selection.features(pilot.existing._generation_source(source_row))["tags"]
    assert source_row == before


def test_structural_features_do_not_force_checklists_into_entity_bucket():
    assert "entity_lists" in selection.features(SOURCE)["tags"]
    assert "entity_lists" not in selection.features("Checklist\n" + SOURCE)["tags"]
    assert "procedures" in selection.features("Checklist\n" + SOURCE)["tags"]
    matrix = "| Feature | A | B |\n| --- | --- | --- |\n| Battery | 8 hours | 10 hours |"
    assert "comparisons" in selection.features(matrix)["tags"]
    assert "comparisons" not in selection.features("A | B is a single prose line.")["tags"]


def test_comma_separated_entity_fields_and_percentages_are_recognized():
    source = "- Alder, Price: $120, Breakfast: not included\n- Birch, Price: $80, Breakfast: included"
    assert "entity_lists" in selection.features(source)["tags"]
    percentages = "Plan A returns 3%, plan B returns 4%, and plan C returns 5%."
    assert "numerical" in selection.features(percentages)["tags"]
    assert "controls" not in selection.features(percentages)["tags"]


@pytest.mark.parametrize("source", [
    "- Gate opens: 09:00\n- Door closes: 10:00",
    "- Speaker: Name: [Name], Role: [Role]\n- Moderator: Name: [Name], Role: [Role]",
])
def test_times_and_template_bullets_are_not_multiple_entity_fields(source):
    assert "entity_lists" not in selection.features(source)["tags"]


def test_heading_entities_and_pipe_tables_without_separator_are_recognized():
    source = ("## Hotel: Alder\n- Price: $120\n- Breakfast: not included\n\n"
              "## Hotel: Birch\n- Price: $80\n- Breakfast: included")
    assert "entity_lists" in selection.features(source)["tags"]
    table = "Feature | Alder | Birch\nPrice | $120 | $80\nBreakfast | Not included | Included"
    hints = selection.features(table)
    assert "comparisons" in hints["tags"]
    assert "controls" not in hints["tags"]


@pytest.mark.parametrize("title", ["Packing", "Run of show"])
def test_packing_and_run_of_show_remain_procedural(title):
    hints = selection.features(title + "\n" + SOURCE)
    assert "procedures" in hints["tags"]
    assert "entity_lists" not in hints["tags"]
    assert "controls" not in hints["tags"]


def test_citation_shortage_uses_actual_url_sources_and_reports_gap():
    items = [(1, "url-a", ("references",)), (2, "url-b", ("references",))]
    items.extend((10 + i, key, (key,)) for i, key in enumerate(selection.QUOTAS) if key != "references")
    selected, report = sample(items)
    assert report["achieved"]["references"] == 2
    assert report["numbered_citations"]["requested"] == 1
    assert report["numbered_citations"]["selected_heuristic_candidates"] == 0
    assert report["numbered_citations"]["shortfall"] == 1
    assert report["numbered_citations"]["reference_slots_replaced_by_url_sources"] == 1
    assert report["numbered_citations"]["independently_verified"] is False
    assert all("[1]" not in entry["row"]["response_text"] for entry in selected)


def test_overlap_reassignment_avoids_false_shortfall_with_sufficient_candidates():
    requested = {key: 0 for key in selection.QUOTAS}
    requested.update(entity_lists=1, comparisons=1)
    selected, report = sample([(1, "shared", ("entity_lists", "comparisons")),
                               (2, "entity-only", ("entity_lists",))],
                              requested=requested, citation_target=0)
    assert {(item["source_id"], item["selection_bucket"]) for item in selected} == {
        ("entity-only", "entity_lists"), ("shared", "comparisons")}
    assert not any(report["category_shortfalls"].values())


def test_overlap_reassignment_preserves_citation_reservation():
    requested = {key: 0 for key in selection.QUOTAS}
    requested.update(entity_lists=1, references=1)
    selected, report = sample([(1, "shared", ("entity_lists", "numbered_citations")),
                               (2, "citation-only", ("numbered_citations",))],
                              requested=requested, citation_target=1)
    assert {(item["source_id"], item["selection_bucket"]) for item in selected} == {
        ("shared", "entity_lists"), ("citation-only", "references")}
    assert report["numbered_citations"]["shortfall"] == 0
    assert not any(report["category_shortfalls"].values())


def test_partial_url_pool_does_not_count_normal_url_slots_as_citation_replacements():
    items = [(1, "url-a", ("references",))]
    items.extend((10 + i, key, (key,)) for i, key in enumerate(selection.QUOTAS) if key != "references")
    items.append((100, "plain-extra", ("controls",)))
    _, report = sample(items)
    assert report["numbered_citations"]["shortfall"] == 1
    assert report["achieved"]["references"] == 1
    assert report["numbered_citations"]["reference_slots_replaced_by_url_sources"] == 0


def test_nonreference_shortage_is_reported_as_backfill_not_achieved_coverage():
    requested = quotas(references=2)
    items = [(i, f"plain-{i}", ("controls",)) for i in range(sum(requested.values()) + 3)]
    selected, report = sample(items, requested=requested)
    assert len(selected) == 8
    assert report["achieved"]["controls"] == 1
    assert report["achieved"]["entity_lists"] == 0
    assert report["category_shortfalls"]["entity_lists"] == 1
    assert report["coverage_backfill"]["entity_lists"] == 1
    backfill = [entry for entry in selected if entry["selection_bucket"] == "coverage_backfill"]
    assert len(backfill) == 7
    assert all(entry["selection_replacement_for"] in requested for entry in backfill)
    assert all(entry["selection_features"]["tags"] == ["controls"] for entry in backfill)


def test_insufficient_distinct_valid_sources_fail_instead_of_short_pilot():
    with pytest.raises(ValueError, match="Only 1 eligible distinct training sources; requested 8"):
        sample([(1, "only-one", ("controls",))])


def test_random_default_has_identical_selection_and_no_profile_metadata(tmp_path):
    directory = inputs(tmp_path, [row("a"), row("b", "Rain expected tomorrow."),
                                 row("c", "The answer remains uncertain.")])
    implicit = pilot.select(SplitInput(directory), 2, 123)
    explicit = pilot.select(SplitInput(directory), 2, 123, selection_profile="random")
    assert implicit == explicit
    assert "selection_report" not in implicit[1]
    assert all("selection_bucket" not in entry for entry in implicit[0])


def test_profile_preserves_real_holdouts_validation_dedup_and_quality_gates(tmp_path, monkeypatch):
    train = [row(f"train-{i}", SOURCE + f"\nRecord number {i}.") for i in range(1000)]
    # Use the actual pinned Bixby source under a renamed training identity.
    golden_path = ROOT / "training/data/eval/bixby50_v1/bixby50.jsonl"
    bixby = json.loads(golden_path.read_text(encoding="utf-8").splitlines()[0])
    train.extend([row("renamed-bixby", bixby["response_text"]),
                  row("renamed-validation", "Meeting   at 11:00."),
                  row("duplicate-valid-source", SOURCE + "\nRecord number 0."),
                  row("unbound-action", "Action: [Button: Open] [ACTION_URL_1]"),
                  row("held-source", "This deliberately held source must never be selected.")])
    restored_duplicate = row("renamed-url-validation", "Read the supplied guide at [SOURCE_URL_1].")
    restored_duplicate["metadata"]["url_preprocessing"] = {"url_map": {
        "[SOURCE_URL_1]": {"url": "https://example.org/guide"}}}
    alias_duplicate = row("renamed-validation-alias", "Different wording with the same held-out source identity.")
    alias_duplicate["source_id"] = "reference-val"
    train.extend([restored_duplicate, alias_duplicate])
    directory = inputs(tmp_path, train)
    runner.write_rows(directory / "val.jsonl", [row("validation", "Meeting at 11:00.", "val"),
        row("reference-val", "Read the supplied guide at https://example.org/guide.", "val")])
    original_guard = pilot.guard.assess_source
    monkeypatch.setattr(pilot.guard, "assess_source", lambda item: (
        {"eligible": False, "reasons": ["fixture_hold"]} if item["id"] == "held-source"
        else original_guard(item)))
    before = file_hash(directory / "train.jsonl")
    selected, counts, benchmark_files = pilot.select(SplitInput(directory), 1000, 123, selection.PROFILE)
    assert {item["source_id"] for item in selected} == {f"train-{i}" for i in range(1000)}
    assert counts["heldout_overlap_excluded"] == 4
    assert counts["duplicate_sources_excluded"] == 1
    assert counts["sampled_source_holds_excluded"] == 1
    assert counts["sampled_source_contract_excluded"] == 1
    assert benchmark_files
    assert file_hash(directory / "train.jsonl") == before
    assert all(item["original_split"] == "train" and item["original_row_sha256"] for item in selected)


@pytest.mark.parametrize("bad", [row("BXP-001"), row("golden35-fixture"),
                                  dict(row("explicit-eval"), evaluation_only=True)])
def test_mix_profile_rejects_explicit_benchmark_identity(tmp_path, bad):
    directory = inputs(tmp_path, [bad])
    with pytest.raises(ValueError, match="evaluation-only"):
        pilot.select(SplitInput(directory), 1000, 123, selection.PROFILE)


@pytest.mark.parametrize("profile,count", [("unknown-profile", 1000),
                                           (selection.PROFILE, 99), (selection.PROFILE, 999)])
def test_invalid_profile_or_wrong_mix_size_fails_before_output(tmp_path, profile, count):
    with pytest.raises(ValueError, match="profile|100.*1000"):
        pilot.prepare(tmp_path / "absent", tmp_path / "output", samples=count, selection_profile=profile)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("count,scale", [(100, 10), (1000, 1)])
def test_profile_quotas_scale_without_changing_original_defaults(count, scale):
    expected = {"entity_lists": 300, "comparisons": 200, "long_sections": 150,
                "procedures": 100, "numerical": 100, "references": 100, "controls": 50}
    assert selection.QUOTAS == expected
    actual = selection.quotas_for_samples(count)
    assert actual == {key: value // scale for key, value in expected.items()}
    assert sum(actual.values()) == count
    actual["entity_lists"] = 0
    assert selection.QUOTAS == expected
    assert selection.MixSampler().quotas == expected
    assert selection.MixSampler().citation_target == 60


@pytest.mark.parametrize("count", [1, 99, 101, 999, 1001, 100.0, True])
def test_profile_select_rejects_unsupported_size(tmp_path, count):
    source = SplitInput(inputs(tmp_path))
    with pytest.raises(ValueError, match="100.*1000"):
        pilot.select(source, count, 123, selection.PROFILE)
    with pytest.raises(ValueError, match="100.*1000"):
        selection.quotas_for_samples(count)


def test_prepare_100_uses_scaled_sampler_and_captures_every_request_without_api(tmp_path, monkeypatch):
    requested = {"entity_lists": 30, "comparisons": 20, "long_sections": 15,
                 "procedures": 10, "numerical": 10, "references": 10, "controls": 5}
    train, hints_by_source = [], {}
    for bucket, count in requested.items():
        for index in range(count):
            source = SOURCE + f"\nFixture bucket {bucket}, record {index}."
            tags = ("numbered_citations",) if bucket == "references" and index < 6 else (bucket,)
            hints_by_source[source] = evidence(*tags)
            train.append(row(f"{bucket}-{index}", source))
    directory = inputs(tmp_path, train)
    before = file_hash(directory / "train.jsonl")
    # Keep admission, sampling, report binding and Stage 3 capture real; only the
    # source classifier uses controlled coverage fixtures.
    monkeypatch.setattr(selection, "features", lambda source: deepcopy(hints_by_source[source]))
    monkeypatch.setattr(api, "client", lambda: pytest.fail("Unexpected paid API client"))
    output = tmp_path / "profile-100"
    result = pilot.prepare(directory, output, samples=100, selection_profile=selection.PROFILE)
    contract = pilot.load(output)
    selected = runner.read_rows(output / "selected.jsonl")
    report = result["selection_report"]
    assert report["requested"] == report["achieved"] == requested
    assert report["numbered_citations"]["requested"] == 6
    assert report["numbered_citations"]["selected_heuristic_candidates"] == 6
    assert report["numbered_citations"]["shortfall"] == 0
    assert not any(report["category_shortfalls"].values())
    assert not any(report["coverage_backfill"].values())
    assert len(selected) == len({entry["source_id"] for entry in selected}) == 100
    assert Counter(entry["selection_bucket"] for entry in selected) == requested
    assert len(runner.read_rows(output / "requests.jsonl")) == len(contract["bindings"]) == 100
    assert contract["config"]["samples"] == 100
    assert contract["config"]["selection_profile"] == selection.PROFILE
    assert contract["files"]["selection_report.json"] == file_hash(output / "selection_report.json")
    assert file_hash(directory / "train.jsonl") == before


def test_prepare_binds_selection_report_and_selected_evidence_without_api(tmp_path, monkeypatch):
    directory = inputs(tmp_path)
    selected, counts, benchmarks = pilot.select(SplitInput(directory), 1, 123)
    hints = selection.features(SOURCE)
    selected[0].update(selection_features=hints, selection_bucket="entity_lists",
                       selection_replacement_for=None)
    report = {"profile": selection.PROFILE, "selected": 1,
              "numbered_citations": {"requested": 60, "selected_heuristic_candidates": 0, "shortfall": 60}}
    # Isolate preparation/hash binding from the separately tested 1000-row
    # sampler, avoiding 1000 identical canonical Stage 3 captures in this test.
    monkeypatch.setattr(pilot, "select", lambda *_: (deepcopy(selected),
                        {**counts, "selection_report": deepcopy(report)}, benchmarks))
    monkeypatch.setattr(api, "client", lambda: pytest.fail("Unexpected paid API client"))
    output = tmp_path / "profile-pilot"
    result = pilot.prepare(directory, output, selection_profile=selection.PROFILE)
    contract = pilot.load(output)
    assert result["selection_report"] == report
    assert contract["config"]["selection_profile"] == selection.PROFILE
    assert contract["files"]["selection_report.json"] == file_hash(output / "selection_report.json")
    assert runner.read_rows(output / "selected.jsonl")[0]["selection_features"] == hints
    (output / "selection_report.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selection_report.json"):
        pilot.load(output)
