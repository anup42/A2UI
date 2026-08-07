from __future__ import annotations

from typing import Any

from ir_training.qat_mtp.workflow import WorkflowIssue


def validate_qat_config(config: dict[str, Any]) -> list[WorkflowIssue]:
    """Validate the opt-in true-QAT SFT profile without loading a model."""

    model = _section(config, "model")
    training = _section(config, "training")
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

    quantizer = str(qat.get("quantizer", "ste_absmax")).strip().lower()
    if quantizer not in {"ste_absmax", "ste_ai_edge"}:
        issues.append(
            WorkflowIssue(
                "error",
                "unsupported_qat_quantizer",
                "Supported QAT quantizers are ste_absmax and ste_ai_edge.",
            )
        )

    weight_bits = _positive_int(qat.get("weight_bits", 8))
    activation_bits = _positive_int(qat.get("activation_bits", 8))
    if weight_bits not in {2, 4, 8}:
        issues.append(WorkflowIssue("error", "invalid_weight_bits", "qat.weight_bits must be 2, 4, or 8."))
    if activation_bits not in {8, 16}:
        issues.append(WorkflowIssue("error", "invalid_activation_bits", "qat.activation_bits must be 8 or 16."))
    group_size = qat.get("group_size")
    if group_size not in (None, "") and _positive_int(group_size) <= 0:
        issues.append(WorkflowIssue("error", "invalid_group_size", "qat.group_size must be a positive integer or null."))

    model_id = str(model.get("model_id") or "").lower()
    if "gemma-4" in model_id or "gemma4" in model_id:
        if weight_bits != 8 or activation_bits != 8:
            issues.append(
                WorkflowIssue(
                    "warning",
                    "gemma4_mobile_profile_mismatch",
                    "The documented Gemma 4 mobile approximation uses W8A8 fake quantization; the final converter must still apply Google's exact mobile recipe.",
                )
            )
    elif "270m" in model_id or "functiongemma" in model_id:
        if weight_bits != 8 or activation_bits != 8:
            issues.append(
                WorkflowIssue(
                    "warning",
                    "gemma270m_int8_profile_mismatch",
                    "The Gemma 270M profiles use W8A8 fake quantization before dynamic INT8 export.",
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
                "Wrapping every Linear also fake-quantizes LoRA A/B layers; base-layer-only is the safer default.",
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
