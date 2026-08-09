from __future__ import annotations

import json

from ir_training.eval.compare_to_baseline import evaluate_predictions
from ir_training.eval.metrics import aggregate_scores, score_prediction


VALID_EXPRESS = '<a2ui>\nroot=Text("Hello")\n</a2ui>'


def test_repaired_candidate_is_not_counted_as_raw_native_valid() -> None:
    repaired = '<a2ui>\nroot=Text("Recovered")\n</a2ui>'
    raw = repaired.replace("</a2ui>", "")
    metrics = score_prediction("Recovered", None, raw, repaired)
    assert metrics["native_syntax_valid"] is False
    assert metrics["native_catalog_valid"] is False
    assert metrics["raw_standard_a2ui_valid"] is False
    assert metrics["repaired_syntax_valid"] is True
    assert metrics["repaired_catalog_valid"] is True
    assert metrics["repaired_standard_a2ui_valid"] is True
    assert metrics["repaired_canonical_semantic_valid"] is True
    assert metrics["canonical_semantic_valid"] is True


def test_v5_4_scoring_is_opt_in_and_exposes_official_headlines() -> None:
    legacy = score_prediction("Hello", None, VALID_EXPRESS)
    assert "generation_reward_v5_4" not in legacy

    metrics = score_prediction(
        "Hello",
        None,
        VALID_EXPRESS,
        metric_version="v5_4",
        intent="generic",
        assets=[],
    )

    assert 0.0 <= metrics["generation_reward_v5_4"] <= 100.0
    assert 0.0 <= metrics["render_artifact_quality_v5_4"] <= 100.0
    assert metrics["genui_metric_version"] == "5.4.0"
    assert metrics["metric_identity_v5_4"]["metric_version"] == "5.4.0"
    assert metrics["metric_identity_v5_4"]["metric_fingerprint"]
    assert metrics["metric_identity_v5_4"]["reward_pipeline_fingerprint"]

    aggregate = aggregate_scores([{"metrics": metrics}])
    assert aggregate["generation_reward_v5_4"] == metrics["generation_reward_v5_4"]
    assert aggregate["render_artifact_quality_v5_4"] == metrics["render_artifact_quality_v5_4"]
    assert aggregate["metric_identity_v5_4"] == metrics["metric_identity_v5_4"]

    dual_metrics = score_prediction(
        "Hello",
        None,
        VALID_EXPRESS,
        metric_version="dual",
    )
    assert dual_metrics["genui_metric_version"] == "5.4.0"
    assert "generation_reward_v5_4" in dual_metrics


def test_evaluate_predictions_forwards_v5_4_source_context(tmp_path, monkeypatch) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "response_text": "Hello [IMAGE_URL_1]",
                "generated_text": VALID_EXPRESS,
                "intent_bucket": "status",
                "assets": [{"url": "[IMAGE_URL_1]"}],
                "expected_ui_contract_v5_4": {"contract_version": "test"},
                "expected_ui_contract_v5_4_source": "persisted",
                "url_map": {"[IMAGE_URL_1]": {"url": "https://example.test/a.png"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_score_prediction(*args, **kwargs):
        captured.update(kwargs)
        captured["response_text"] = args[0]
        return {"generation_reward_v5_4": 75.0, "render_artifact_quality_v5_4": 70.0}

    monkeypatch.setattr(
        "ir_training.eval.compare_to_baseline.score_prediction",
        fake_score_prediction,
    )

    aggregate = evaluate_predictions(predictions, metric_version="dual")

    assert captured["metric_version"] == "dual"
    assert captured["intent"] == "status"
    assert captured["assets"] == [{"url": "https://example.test/a.png"}]
    assert captured["expected_ui_contract"] == {"contract_version": "test"}
    assert captured["expected_ui_contract_source"] == "persisted"
    assert captured["response_text"] == "Hello https://example.test/a.png"
    assert aggregate["generation_reward_v5_4"] == 75.0
    assert aggregate["render_artifact_quality_v5_4"] == 70.0
