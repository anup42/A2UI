from __future__ import annotations

from typing import Any

from ir_training.qat.fake_quant import QATSpec, qat_numeric_contract
from ir_training.qat.mobile_training_seed import OFFICIAL_MOBILE_MODEL_ID
from ir_training.qat_mtp.workflow import WorkflowIssue


def validate_qat_config(config: dict[str, Any]) -> list[WorkflowIssue]:
    """Validate the opt-in true-QAT SFT profile without loading a model."""

    model = _section(config, "model")
    training = _section(config, "training")
    lora = _section(config, "lora")
    qat = _section(config, "qat")
    issues: list[WorkflowIssue] = []

    if qat.get("enabled") is not True:
        issues.append(
            WorkflowIssue(
                "error",
                "qat_not_enabled",
                "The true-QAT path requires qat.enabled: true; use qat_mtp for the separate QAT-derived workflow.",
            )
        )

    method = str(training.get("method") or "").strip().lower()
    if method not in {"qat_lora_sft", "lora_sft", "sft_lora"}:
        issues.append(
            WorkflowIssue(
                "error",
                "unsupported_qat_training_method",
                "True QAT currently supports LoRA SFT only (training.method=qat_lora_sft).",
            )
        )

    if bool(model.get("load_in_4bit", False)):
        issues.append(
            WorkflowIssue(
                "error",
                "qlora_is_not_qat",
                "model.load_in_4bit must be false; BitsAndBytes NF4/QLoRA is not fake-quantization-aware training.",
            )
        )

    if qat.get("effective_merged_weight", True) is not True:
        issues.append(
            WorkflowIssue(
                "error",
                "effective_merged_weight_qat_required",
                "QAT must fake-quantize base_weight + LoRA_delta; quantizing only "
                "the frozen base does not simulate the post-merge LiteRT weight.",
            )
        )
    try:
        lora_dropout = float(lora.get("dropout", 0.0) or 0.0)
    except (TypeError, ValueError):
        lora_dropout = -1.0
    if lora_dropout != 0.0:
        issues.append(
            WorkflowIssue(
                "error",
                "nonzero_lora_dropout_breaks_merged_qat",
                "Exact effective-weight QAT requires lora.dropout: 0.0 because "
                "a per-example adapter dropout mask cannot be represented by the "
                "final merged inference matrix.",
            )
        )

    quantizer = str(qat.get("quantizer", "ste_absmax")).strip().lower()
    if quantizer not in {"ste_absmax", "ste_ai_edge"}:
        issues.append(
            WorkflowIssue(
                "error",
                "unsupported_qat_quantizer",
                "Supported QAT quantizers are ste_absmax and ste_ai_edge.",
            )
        )

    try:
        numeric_contract = qat_numeric_contract(QATSpec.from_config(config))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        numeric_contract = {"public_ai_edge_numeric_contract": False}
        issues.append(
            WorkflowIssue(
                "error",
                "qat_numeric_contract_unreadable",
                f"Could not resolve the QAT numerical contract: {exc}",
            )
        )
    if (
        quantizer == "ste_ai_edge"
        and numeric_contract.get("public_ai_edge_numeric_contract") is not True
    ):
        issues.append(
            WorkflowIssue(
                "error",
                "ai_edge_numeric_contract_mismatch",
                "ste_ai_edge requires the public ai-edge-quantizer 0.8.0 "
                "FLOAT32 scale calculation and 1e-9 minimum scale contract.",
            )
        )

    weight_bits = _positive_int(qat.get("weight_bits", 8))
    activation_bits = _positive_int(qat.get("activation_bits", 8))
    if weight_bits not in {2, 4, 8}:
        issues.append(WorkflowIssue("error", "invalid_weight_bits", "qat.weight_bits must be 2, 4, or 8."))
    if activation_bits not in {8, 16, 32}:
        issues.append(
            WorkflowIssue(
                "error",
                "invalid_activation_bits",
                "qat.activation_bits must be 8, 16, or 32.",
            )
        )
    group_size = qat.get("group_size")
    if group_size not in (None, "") and _positive_int(group_size) <= 0:
        issues.append(WorkflowIssue("error", "invalid_group_size", "qat.group_size must be a positive integer or null."))

    model_id = str(model.get("model_id") or "").lower()
    if "gemma-4" in model_id or "gemma4" in model_id:
        if (
            model_id == OFFICIAL_MOBILE_MODEL_ID.lower()
            and model.get("architecture_preflight_required") is not True
        ):
            issues.append(
                WorkflowIssue(
                    "error",
                    "gemma4_architecture_preflight_required",
                    "Gemma 4 mobile training must compare all reconstructed "
                    "checkpoint keys/shapes with Gemma4ForCausalLM on the meta "
                    "device before allocating or training the model.",
                )
            )
        if (
            model_id == OFFICIAL_MOBILE_MODEL_ID.lower()
            and model.get("require_exact_checkpoint_keys") is not True
        ):
            issues.append(
                WorkflowIssue(
                    "error",
                    "gemma4_exact_checkpoint_keys_required",
                    "Gemma 4 mobile training must require zero missing, unexpected, "
                    "mismatched, or errored checkpoint keys during the real load.",
                )
            )
        try:
            gemma4_spec = QATSpec.from_config(config)
            observable_layout_matches = bool(
                quantizer == "ste_ai_edge"
                and qat.get("quantize_embeddings") is True
                and gemma4_spec.weight_bits_for_module(
                    "language_model.embed_tokens"
                )
                == 2
                and gemma4_spec.weight_bits_for_module(
                    "language_model.embed_tokens_per_layer"
                )
                == 4
                and gemma4_spec.group_size_for_module(
                    "language_model.embed_tokens_per_layer"
                )
                == 256
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            observable_layout_matches = False
        if not observable_layout_matches:
            issues.append(
                WorkflowIssue(
                    "error",
                    "gemma4_mobile_observable_layout_mismatch",
                    "Gemma 4 mobile QAT requires the public W2 token embedding, "
                    "W4 per-layer embedding, and its observed 256-column grouped "
                    "scales under the ste_ai_edge range convention.",
                )
            )
        if weight_bits != 8 or activation_bits != 8:
            issues.append(
                WorkflowIssue(
                    "warning",
                    "gemma4_mobile_profile_mismatch",
                    "The documented Gemma 4 mobile approximation uses W8A8 fake quantization; the final converter must still apply Google's exact mobile recipe.",
                )
            )
    elif "gemma-3-270m" in model_id:
        if (
            weight_bits != 8
            or activation_bits < 16
            or quantizer != "ste_ai_edge"
            or not bool(qat.get("quantize_embeddings", False))
        ):
            issues.append(
                WorkflowIssue(
                    "warning",
                    "gemma270m_int8_profile_mismatch",
                    "The audited Gemma 3 270M Q8 graph uses public AI Edge per-row "
                    "INT8 weight ranges, a quantized embedding table, and floating-point "
                    "activation edges.",
                )
            )
    elif "functiongemma" in model_id:
        if weight_bits != 8 or activation_bits != 8:
            issues.append(
                WorkflowIssue(
                    "warning",
                    "functiongemma270m_profile_mismatch",
                    "Keep the separately configured FunctionGemma W8A8 profile until "
                    "its target package is independently audited.",
                )
            )
    else:
        issues.append(
            WorkflowIssue(
                "warning",
                "unrecognized_qat_target",
                f"No target-specific profile was recognized for model_id={model_id or '<missing>'}; verify the export recipe manually.",
            )
        )

    if float(training.get("learning_rate", 0.0) or 0.0) > 1e-4:
        issues.append(
            WorkflowIssue(
                "warning",
                "aggressive_learning_rate",
                "QAT can be sensitive to large updates; compare 5e-5 and 1e-4 before promotion.",
            )
        )
    if qat.get("only_base_layers", True) is not True:
        issues.append(
            WorkflowIssue(
                "warning",
                "full_linear_scope",
                "Keep qat.only_base_layers=true so adapter matrices are represented "
                "only through the fake-quantized effective merged weight.",
            )
        )
    if not bool(qat.get("final_runtime_validation_required", False)):
        issues.append(
            WorkflowIssue(
                "error",
                "missing_runtime_gate",
                "A true-QAT checkpoint must require post-export target-only runtime validation before promotion.",
            )
        )
    if not str(qat.get("final_quantization") or "").strip():
        issues.append(
            WorkflowIssue(
                "warning",
                "missing_final_quantization",
                "Set qat.final_quantization to the exact LiteRT/LiteRT-LM or GGUF recipe used after adapter merge.",
            )
        )
    return issues


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def _positive_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1
