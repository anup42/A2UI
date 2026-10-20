"""Real scorer integration: GRPO training and every eval share a fingerprint."""
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "src"))
sys.path.insert(0, str(ROOT / "dataset" / "src"))

from ir_training.eval.compare_to_baseline import evaluate_predictions
from ir_training.train.qat_grpo_reward import (
    make_qat_grpo_reward,
    reward_implementation_identity,
)
from pipeline.genui_quality.config_v5_4 import load_reward_config_v5_4

SOURCE = "- Hotel Aurora: Goa; Price: INR 5000; [Book now](https://example.test/aurora)"
COMPLETION = '''<a2ui>root=Card([city,price,book],title="Hotel Aurora")
city=Text("Goa")
price=Text("Price: INR 5000")
book=Button("Book now",onPress=openUrl("https://example.test/aurora"))</a2ui>'''


def test_real_grpo_eval_mapping_uses_same_config_and_identity_as_reward(tmp_path, monkeypatch):
    monkeypatch.delenv("A2UI_ANDROID_REPAIR_RUNTIME", raising=False)
    reward_mapping = yaml.safe_load((ROOT / "dataset/configs/genui_metric_v5_4.yaml").read_text())
    reward_mapping.setdefault("policies", {})["grpo_entity_extraction"] = True
    weights = yaml.safe_load((ROOT / "dataset/configs/run.yaml").read_text())
    weights["grpo_reward_config"] = reward_mapping
    reward_path, weights_path = tmp_path / "reward.yaml", tmp_path / "weights.yaml"
    reward_path.write_text(yaml.safe_dump(reward_mapping), encoding="utf-8")
    weights_path.write_text(yaml.safe_dump(weights), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"id": "synthetic", "response_text": SOURCE,
                                       "generated_text": COMPLETION}) + "\n", encoding="utf-8")

    canonical = evaluate_predictions(predictions, metric_version="v5_4")
    evaluated = evaluate_predictions(predictions, weights_config_path=weights_path, metric_version="v5_4")
    captured = {}
    values = make_qat_grpo_reward(load_reward_config_v5_4(reward_path))(
        [COMPLETION], SOURCE, log_extra=lambda key, value: captured.update({key: value}))
    expected = reward_implementation_identity(reward_path)
    identity = evaluated["metric_identity_v5_4"]
    assert identity["metric_fingerprint"] == expected["metric_fingerprint"]
    assert identity["metric_fingerprint"] == expected["evaluation_metric_fingerprint"]
    assert identity["reward_pipeline_fingerprint"] == expected["reward_pipeline_fingerprint"]
    assert identity["metric_fingerprint"] != canonical["metric_identity_v5_4"]["metric_fingerprint"]
    assert evaluated["generation_reward_v5_4_avg"] == pytest.approx(
        100.0 * captured["qat_reward"][0]["canonical_quality"])
    assert values[0] > 0.95
    assert evaluated["generation_reward_v5_4_avg"] > canonical["generation_reward_v5_4_avg"]
