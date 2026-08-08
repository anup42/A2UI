from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_gemma4_mtp_drafter_official_topology as topology
from ir_training.common.config import load_yaml
from ir_training.mtp.drafter_contract import (
    contract_summary,
    deployment_weight_specs,
    source_keys,
)
from ir_training.train.mtp_drafter import (
    MTPDrafterTrainingError,
    _anchor_positions,
    _deployment_trainable_parameters,
    _rollout_loss,
    build_drafter_training_plan,
)


def _config() -> dict:
    return load_yaml(ROOT / "configs" / "models" / "gemma4_e2b_mtp_drafter_qat.yaml")


def test_mobile_drafter_contract_is_exactly_23_mixed_precision_matrices():
    specs = deployment_weight_specs()

    assert len(specs) == 23
    assert [item.ordinal for item in specs] == list(range(23))
    assert contract_summary()["bit_histogram"] == {"8": 10, "4": 13}
    assert specs[0].source_key == "pre_projection.weight"
    assert specs[0].shape == (256, 3072)
    assert specs[21].source_key == "model.embed_tokens.weight"
    assert specs[21].shape == (262144, 256)
    assert specs[22].source_key == "post_projection.weight"


def test_drafter_training_plan_is_static_and_matches_mobile_qat_contract(tmp_path):
    config = _config()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text(
        json.dumps({"input_ids": [1, 2], "labels": [-100, 2]}) + "\n",
        encoding="utf-8",
    )
    target = tmp_path / "merged-target"
    target.mkdir()
    output = tmp_path / "run"
    config["run"]["dataset_dir"] = str(dataset)
    config["run"]["output_dir"] = str(output)
    config["target"]["model_id_or_path"] = str(target)
    config["assistant"]["best_checkpoint_dir"] = str(output / "best_checkpoint")

    plan = build_drafter_training_plan(config)

    assert plan["ready"] is True
    assert plan["training_executed"] is False
    assert plan["private_google_training_recipe_recovered"] is False
    assert plan["target"]["frozen"] is True
    assert plan["assistant"]["unmapped_parameters_frozen"] is True
    assert plan["assistant"]["trainable_source_keys"] == source_keys()
    assert plan["qat"]["exact_official_inventory_precision"] is True
    assert plan["mobile_contract"]["weight_count"] == 23


def test_drafter_plan_rejects_qat_precision_drift(tmp_path):
    config = _config()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text("{}\n", encoding="utf-8")
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "run"
    config["run"].update(dataset_dir=str(dataset), output_dir=str(output))
    config["target"]["model_id_or_path"] = str(target)
    config["assistant"]["best_checkpoint_dir"] = str(output / "best_checkpoint")
    config["qat"]["module_quant_configs"]["^pre_projection$"] = 4

    plan = build_drafter_training_plan(config)

    assert plan["ready"] is False
    assert plan["qat"]["exact_official_inventory_precision"] is False
    assert any(
        item["code"] == "qat_precision_does_not_match_official_mtp_inventory"
        for item in plan["issues"]
    )


def test_anchor_positions_only_select_supervised_next_tokens():
    positions = _anchor_positions(
        [-100, -100, 7, 8, -100, 9],
        positions_per_sequence=10,
        rng=random.Random(7),
    )

    assert positions == [1, 2, 4]


def test_rollout_uses_constant_position_and_fixed_target_kv():
    torch = pytest.importorskip("torch")

    class Target:
        def __init__(self):
            self.embedding = torch.nn.Embedding(32, 3)
            self.calls = []

        def get_input_embeddings(self):
            return self.embedding

        def __call__(self, *, input_ids, **kwargs):
            self.calls.append({"input_ids": input_ids.detach().clone(), **kwargs})
            sequence = input_ids.shape[1]
            hidden = torch.arange(
                sequence * 3, dtype=torch.float32, device=input_ids.device
            ).reshape(1, sequence, 3)
            shared = {
                "full_attention": (
                    torch.zeros(1, 1, sequence, 1, device=input_ids.device),
                    torch.zeros(1, 1, sequence, 1, device=input_ids.device),
                ),
                "sliding_attention": (
                    torch.zeros(1, 1, sequence, 1, device=input_ids.device),
                    torch.zeros(1, 1, sequence, 1, device=input_ids.device),
                ),
            }
            return SimpleNamespace(
                hidden_states=[hidden], shared_kv_states=shared
            )

    class Assistant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(6, 32, bias=False)
            self.calls = []

        def forward(
            self,
            *,
            inputs_embeds,
            attention_mask,
            position_ids,
            shared_kv_states,
            use_cache,
        ):
            self.calls.append(
                {
                    "position_ids": position_ids.detach().clone(),
                    "attention_mask": attention_mask.detach().clone(),
                    "shared_kv_states": shared_kv_states,
                    "use_cache": use_cache,
                }
            )
            return SimpleNamespace(
                logits=self.head(inputs_embeds),
                last_hidden_state=inputs_embeds[..., :3],
            )

    target = Target()
    assistant = Assistant()
    loss = _rollout_loss(
        target_model=target,
        assistant_model=assistant,
        input_ids=[1, 2, 3, 4, 5],
        labels=[-100, 2, 3, 4, 5],
        anchors=[1, 2],
        draft_steps=4,
        device=torch.device("cpu"),
    )

    loss.backward()
    assert len(target.calls) == 1
    assert target.calls[0]["input_ids"].shape == (1, 5)
    assert target.calls[0]["return_shared_kv_states"] is True
    assert target.calls[0]["output_hidden_states"] is True
    assert len(assistant.calls) == 5
    assert [item["position_ids"].item() for item in assistant.calls] == [
        1,
        1,
        1,
        2,
        2,
    ]
    assert len({id(item["shared_kv_states"]) for item in assistant.calls}) == 2
    assert [
        item["shared_kv_states"]["full_attention"][0].shape[2]
        for item in assistant.calls
    ] == [2, 2, 2, 3, 3]
    assert all(item["use_cache"] is False for item in assistant.calls)
    assert assistant.head.weight.grad is not None


def test_deployment_training_scope_requires_tied_vocabulary_matrix():
    torch = pytest.importorskip("torch")

    class Leaf(torch.nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(shape))

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = Leaf((2, 2))
            self.o_proj = Leaf((2, 2))

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = Leaf((2, 2))
            self.up_proj = Leaf((2, 2))
            self.down_proj = Leaf((2, 2))

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
            self.mlp = MLP()

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer() for _ in range(4)])
            self.embed_tokens = torch.nn.Embedding(8, 2)

    class Assistant(torch.nn.Module):
        def __init__(self, *, tied: bool):
            super().__init__()
            self.pre_projection = Leaf((2, 2))
            self.model = Backbone()
            self.post_projection = Leaf((2, 2))
            self.lm_head = Leaf((8, 2))
            if tied:
                self.lm_head.weight = self.model.embed_tokens.weight
            self.unmapped = Leaf((2, 2))

    tied = Assistant(tied=True)
    trainable = _deployment_trainable_parameters(tied)
    assert list(trainable) == source_keys()
    assert tied.unmapped.weight.requires_grad is False
    assert tied.lm_head.weight.requires_grad is True

    with pytest.raises(MTPDrafterTrainingError, match="must remain tied"):
        _deployment_trainable_parameters(Assistant(tied=False))


def test_drafter_export_scope_matches_checked_in_training_config():
    report = topology._drafter_training_scope_report(
        ROOT / "configs" / "models" / "gemma4_e2b_mtp_drafter_qat.yaml",
        official_base_model_id="google/gemma-4-E2B-it-assistant",
    )

    assert report["supported_projection_only_transplant"] is True
    assert all(report["checks"].values())
    assert len(report["precision_assignments"]) == 23


def test_drafter_export_provenance_hashes_checkpoint_and_config(tmp_path):
    config_path = ROOT / "configs" / "models" / "gemma4_e2b_mtp_drafter_qat.yaml"
    checkpoint = tmp_path / "best_checkpoint"
    checkpoint.mkdir()
    model_file = checkpoint / "model.safetensors"
    model_file.write_bytes(b"checkpoint")
    metadata = {
        "manifest_version": 1,
        "training_method": "target_conditioned_autoregressive_qat",
        "loss": "teacher_forced_completion_cross_entropy",
        "assistant_base_model_id_or_path": "google/gemma-4-E2B-it-assistant",
        "training_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "target_frozen": True,
        "draft_steps": 4,
        "teacher_forcing": True,
        "constant_position_ids": True,
        "fixed_target_kv_during_each_draft_rollout": True,
        "unmapped_parameters_frozen": True,
        "training_unmapped_constants_source": "public_assistant_checkpoint",
        "deployment_unmapped_constants_source": "official_litertlm_artifact",
        "private_google_training_recipe_recovered": False,
        "trainable_source_keys": source_keys(),
        "mobile_contract": contract_summary(),
        "checkpoint_files": [
            {
                "path": model_file.name,
                "size": model_file.stat().st_size,
                "sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
            }
        ],
    }
    (checkpoint / "mtp_drafter_training_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    report = topology._training_provenance_report(
        checkpoint,
        config_path,
        official_assistant_model_id="google/gemma-4-E2B-it-assistant",
    )
    assert report["verified"] is True

    model_file.write_bytes(b"tampered")
    tampered = topology._training_provenance_report(
        checkpoint,
        config_path,
        official_assistant_model_id="google/gemma-4-E2B-it-assistant",
    )
    assert tampered["verified"] is False
    assert tampered["checks"]["checkpoint_hashes_match"] is False


def test_drafter_export_plan_fails_closed_without_artifacts(tmp_path):
    output = tmp_path / "candidate.litertlm"
    plan = topology.build_plan(
        tmp_path / "official.litertlm",
        tmp_path / "target.litertlm",
        tmp_path / "checkpoint",
        assistant_training_config=(
            ROOT / "configs" / "models" / "gemma4_e2b_mtp_drafter_qat.yaml"
        ),
        official_artifact_sha256="0" * 64,
        output_dir=tmp_path / "export",
        package_output=output,
    )

    assert plan["ready"] is False
    assert output.exists() is False
    assert plan["training_executed"] is False
    assert {item["code"] for item in plan["issues"]} >= {
        "official_artifact_missing",
        "package_input_missing",
        "assistant_checkpoint_invalid",
        "unverified_drafter_training_provenance",
    }
