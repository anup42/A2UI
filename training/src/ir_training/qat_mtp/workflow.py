from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

OFFICIAL_QAT_TARGET = "google/gemma-4-E2B-it-qat-q4_0-unquantized"
OFFICIAL_QAT_ASSISTANT = "google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant"

_PRECISION_RE = re.compile(r"-qat-([a-z0-9_]+?)-(?:unquantized|gguf|ct)(?:-|$)", re.IGNORECASE)


@dataclass(frozen=True)
class WorkflowIssue:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def checkpoint_precision_family(model_id: str) -> str | None:
    match = _PRECISION_RE.search(str(model_id))
    return match.group(1).lower() if match else None


def validate_training_config(config: dict[str, Any]) -> list[WorkflowIssue]:
    model = _section(config, "model")
    training = _section(config, "training")
    lora = _section(config, "lora")
    workflow = _section(config, "qat_mtp")
    issues: list[WorkflowIssue] = []

    model_id = str(model.get("model_id") or "")
    if model_id != OFFICIAL_QAT_TARGET:
        issues.append(
            WorkflowIssue(
                "warning",
                "non_recommended_target",
                f"Recommended Q4_0 seed is {OFFICIAL_QAT_TARGET}; configured target is {model_id or '<missing>'}.",
            )
        )
    if bool(model.get("load_in_4bit", False)):
        issues.append(
            WorkflowIssue(
                "error",
                "qlora_is_not_qat",
                "model.load_in_4bit must remain false for the recommended BF16 LoRA path; BitsAndBytes NF4 is not Q4_0 QAT.",
            )
        )
    dtype = str(model.get("dtype", "")).lower()
    if dtype not in {"bfloat16", "bf16"}:
        issues.append(
            WorkflowIssue(
                "warning",
                "bf16_preferred",
                f"BF16 is preferred when hardware supports it; configured dtype is {dtype or '<missing>'}.",
            )
        )
    method = str(training.get("method") or "").lower()
    if method not in {"lora_sft", "sft_lora"}:
        issues.append(
            WorkflowIssue(
                "error",
                "unsupported_training_method",
                "The recommended profile is target-only LoRA SFT; it must not claim joint QAT or assistant training.",
            )
        )
    if float(training.get("learning_rate", 0.0) or 0.0) > 1e-4:
        issues.append(
            WorkflowIssue(
                "warning",
                "aggressive_learning_rate",
                "Learning rates above 1e-4 can increase target drift; compare 5e-5 and 1e-4 before promoting a checkpoint.",
            )
        )
    target_modules = str(lora.get("target_modules") or "").strip().lower()
    if target_modules == "all-linear":
        issues.append(
            WorkflowIssue(
                "warning",
                "broad_lora_scope",
                "Blanket all-linear LoRA can touch non-language projections; prefer PEFT's Gemma 4 LM defaults.",
            )
        )
    modules_to_save = {str(item) for item in (lora.get("modules_to_save") or [])}
    if modules_to_save.intersection({"lm_head", "embed_tokens"}):
        issues.append(
            WorkflowIssue(
                "warning",
                "vocabulary_weights_modified",
                "Updating lm_head/embed_tokens can reduce compatibility with the frozen assistant unless tokenizer changes require it.",
            )
        )

    assistant_id = str(workflow.get("assistant_model_id") or "")
    if assistant_id != OFFICIAL_QAT_ASSISTANT:
        issues.append(
            WorkflowIssue(
                "error",
                "assistant_mismatch",
                f"Use the exact matching QAT assistant {OFFICIAL_QAT_ASSISTANT} for the Q4_0 reference workflow.",
            )
        )
    if bool(workflow.get("train_assistant", False)):
        issues.append(
            WorkflowIssue(
                "error",
                "assistant_training_unsupported",
                "Public Google tooling does not provide the Gemma 4 E2B assistant training objective/export recipe; train_assistant must be false.",
            )
        )
    if bool(workflow.get("continued_qat", False)):
        issues.append(
            WorkflowIssue(
                "error",
                "continued_qat_not_implemented",
                "This repository implements QAT-derived LoRA, not fake-quantization-aware weight updates; continued_qat must be false.",
            )
        )
    if str(workflow.get("final_quantization") or "").lower() != "q4_0":
        issues.append(
            WorkflowIssue(
                "warning",
                "deployment_format_not_q4_0",
                "The supplied checkpoint pair is Q4_0-derived; use a separate validated profile for mobile wNa8o8 or server W4A16.",
            )
        )
    if not bool(workflow.get("final_runtime_validation_required", False)):
        issues.append(
            WorkflowIssue(
                "error",
                "missing_runtime_gate",
                "Final packed-INT4 target-only and target+assistant validation must be required before promotion.",
            )
        )
    return issues


def validate_benchmark_config(config: dict[str, Any]) -> list[WorkflowIssue]:
    source = _section(config, "source")
    benchmark = _section(config, "benchmark")
    issues: list[WorkflowIssue] = []
    target_base = str(source.get("target_base_model_id") or source.get("target_model_id") or "")
    assistant_id = str(source.get("assistant_model_id") or "")

    target_precision = checkpoint_precision_family(target_base)
    assistant_precision = checkpoint_precision_family(assistant_id)
    if target_precision is None or assistant_precision is None or target_precision != assistant_precision:
        issues.append(
            WorkflowIssue(
                "error",
                "precision_family_mismatch",
                f"Target precision family {target_precision!r} and assistant precision family {assistant_precision!r} must match.",
            )
        )
    if assistant_id != OFFICIAL_QAT_ASSISTANT:
        issues.append(
            WorkflowIssue(
                "error",
                "assistant_mismatch",
                f"Configured assistant must be {OFFICIAL_QAT_ASSISTANT}.",
            )
        )
    if str(benchmark.get("mode") or "") != "transformers_reference":
        issues.append(
            WorkflowIssue(
                "error",
                "unsupported_benchmark_mode",
                "This script supports only a Transformers reference benchmark; final runtime benchmarking is separate.",
            )
        )
    if bool(benchmark.get("packed_int4", False)):
        issues.append(
            WorkflowIssue(
                "error",
                "false_int4_claim",
                "The Transformers reference benchmark loads unquantized QAT-derived weights and must not claim packed INT4.",
            )
        )
    if not bool(benchmark.get("final_runtime_validation_required", False)):
        issues.append(
            WorkflowIssue(
                "error",
                "missing_runtime_gate",
                "The benchmark config must explicitly require a separate packed-runtime device benchmark.",
            )
        )
    schedule = str(benchmark.get("num_assistant_tokens_schedule") or "heuristic")
    if schedule not in {"heuristic", "constant"}:
        issues.append(
            WorkflowIssue(
                "error",
                "invalid_assistant_schedule",
                "num_assistant_tokens_schedule must be 'heuristic' or 'constant'.",
            )
        )
    return issues


def summarize_issues(issues: list[WorkflowIssue]) -> dict[str, Any]:
    return {
        "ok": not any(issue.severity == "error" for issue in issues),
        "error_count": sum(issue.severity == "error" for issue in issues),
        "warning_count": sum(issue.severity == "warning" for issue in issues),
        "issues": [issue.to_dict() for issue in issues],
    }


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}
