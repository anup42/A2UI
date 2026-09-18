"""Cross-check saved model attempts, effective prompt flags, and runtime evidence."""
import json

if not __debug__:
    raise RuntimeError("Prompt-study integrity checks require Python without -O/-OO optimization.")

SCAFFOLD_MARKER = "\nExpress scaffold (preserve all lines; replace table SELECT_DOMAIN and SELECT_PRESENTATION):\n"


def validate_attempt_evidence(run, meta, config, rows):
    planned = config["cases"]
    observed = [row["id"] for row in rows]
    if len(planned) != len(set(planned)) or len(observed) != len(set(observed)):
        raise ValueError(f"Duplicated planned/result case IDs: {run}")
    if set(planned) != set(meta["cases"]) or not set(observed) <= set(planned):
        raise ValueError(f"Unexpected or inconsistent case IDs: {run}")
    if set(observed) != set(planned) and not meta.get("earlyRejected"):
        raise ValueError(f"Missing completed case IDs: {run}")
    assert config["provider"] == "gemma", run
    assert config["accelerator"] == "GPU" and config["mtp"] and config["thinkingEnabled"], run
    assert int(config["thinkingBudgetOverride"]) == 1024, run
    assert config["maxRepairAttempts"] == meta["repairs"] and config["temperature"] == 0, run
    if "promptPath" in config or meta.get("packagedPrompt"):
        assert (config["promptPath"] == "aar_asset") == bool(meta.get("packagedPrompt")), run
        assert config["promptSha256"] == meta["promptSha256"], run
        assert config["corpusSha256"] == meta["corpusSha256"], run
        assert config["sourceBindings"] == meta["sourceBindings"] == True, run
        assert config["inputScaffold"] == bool(meta.get("inputScaffold")), run
    if meta.get("recordInputs"):
        assert config["recordInputs"] == True, run
    if not meta.get("earlyRejected"):
        completion = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        successes = sum(row["status"] == "success" for row in rows)
        assert completion["runId"] == run.name, run
        assert completion["total"] == len(config["cases"]) == len(rows), run
        assert completion["success"] == successes and completion["failure"] == len(rows) - successes, run
        if successes == len(rows):
            log = (run / "instrumentation.log").read_text(encoding="utf-8")
            assert "OK (1 test)" in log and "INSTRUMENTATION_CODE: -1" in log, (run, "Instrumentation did not finish successfully")
    expected = set()
    for row in rows:
        assert row["provider"] == "gemma4_e2b", (run, row["id"])
        assert 1 <= row["attempts"] <= meta["repairs"] + 1, (run, row["id"])
        expected.update(run / row["id"] / f"attempt_{attempt}_metrics.json"
                        for attempt in range(1, row["attempts"] + 1))
    found = {p for case_id in meta["cases"] for p in (run / case_id).glob("attempt_*_metrics.json")}
    assert expected <= found, (run, "Missing attempt metrics", sorted(str(p) for p in expected - found))
    orphaned = found - expected
    assert not orphaned or meta.get("earlyRejected"), (run, "Unexplained model attempts", orphaned)
    metrics = []
    scaffold_counts = []
    for path in sorted(found):
        metric = json.loads(path.read_text(encoding="utf-8"))
        assert metric["runtime"] == "LiteRT-LM/Gemma4/GPU+MTP", path
        raw = path.with_name(path.name.replace("_metrics.json", ".txt"))
        assert raw.is_file(), (path, "Missing raw model output")
        if meta.get("inputScaffold") or meta.get("recordInputs"):
            prompt = path.with_name(path.name.replace("_metrics.json", "_input.txt"))
            count = prompt.read_text(encoding="utf-8").count(SCAFFOLD_MARKER)
            expected_count = int(bool(meta.get("inputScaffold") or meta.get("sdkLayoutScaffold")))
            assert count == expected_count, (prompt, "Unexpected source-scaffold count", count, expected_count)
            if "inputScaffoldCount" in metric:
                assert metric["inputScaffoldCount"] == count, path
            scaffold_counts.append(count)
        metrics.append(metric)
    assert metrics, (run, "No captured model calls")
    return {"captured_model_calls": len(metrics), "reconciled_completed_case_calls": len(expected),
            "orphaned_inflight_metrics": [str(p.relative_to(run)) for p in sorted(orphaned)],
            "effective_scaffold_counts": scaffold_counts,
            "effective_prompt_provenance": "recorded_and_matched" if "promptPath" in config else "legacy_config_and_host_command"}
