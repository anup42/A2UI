from __future__ import annotations

from typing import Any

from ir_training.qat.fake_quant import QATSpec, qat_numeric_contract
from ir_training.qat.mobile_training_seed import OFFICIAL_MOBILE_MODEL_ID
from ir_training.qat.numeric_preflight import (
    RETAINED_MOBILE_POLICY,
    resolve_numeric_policy,
)
from ir_training.qat_mtp.workflow import WorkflowIssue


def validate_qat_config(config: dict[str, Any]) -> list[WorkflowIssue]:
    """Validate the opt-in true-QAT SFT profile without loading a model."""

    from ir_training.qat.full_model_contract import (
        is_full_qat,
        validate_full_qat_config,
    )

    if is_full_qat(config):
        try:
            validate_full_qat_config(config)
        except (TypeError, ValueError, KeyError) as exc:
            return [WorkflowIssue("error", "invalid_all_parameter_qat_contract", str(exc))]
        return []

    model = _section(config, "model")
    training = _section(config, "training")
    lora = _section(config, "lora")
    qat = _section(config, "qat")
    issues: list[WorkflowIssue] = []
    diagnostic_numeric_policy = False
    try:
        diagnostic_numeric_policy = resolve_numeric_policy(config) == RETAINED_MOBILE_POLICY
    except (TypeError, ValueError) as exc:
        issues.append(WorkflowIssue("error", "invalid_numeric_preflight_policy", str(exc)))

    if qat.get("enabled") is not True:
        issues.append(
            WorkflowIssue(
                "error",
                "qat_not_enabled",
                "The true-QAT path requires qat.enabled: true; use qat_mtp for the separate QAT-derived workflow.",
            )
        )

    method = str(training.get("method") or "").strip().lower()
    full_finetune = method == "full_finetune_qat"
    if method not in {"qat_lora_sft", "lora_sft", "sft_lora", "full_finetune_qat"}:
        issues.append(
            WorkflowIssue(
                "error",
                "unsupported_qat_training_method",
                "QAT supports qat_lora_sft or full_finetune_qat with the checked HF SFT backend.",
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

    if not full_finetune and qat.get("effective_merged_weight", True) is not True:
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

    scale_mode = str(qat.get("scale_mode", "dynamic")).strip().lower()
    if full_finetune and (lora or scale_mode == "retained_mobile" or qat.get("effective_lora_only")):
        issues.append(WorkflowIssue("error", "unsupported_full_qat_contract", "Full QAT requires dynamic scales, no LoRA settings, and full eligible weight coverage."))
    if scale_mode not in {"dynamic", "retained_mobile"}:
        issues.append(
            WorkflowIssue(
                "error",
                "unsupported_qat_scale_mode",
                "qat.scale_mode must be dynamic or retained_mobile.",
            )
        )
    ste_gradient = str(qat.get("ste_gradient", "identity")).strip().lower()
    activation_quantizer = str(qat.get("activation_quantizer", "legacy")).strip().lower()
    if activation_quantizer not in {"legacy", "gemma_mobile_srq"}:
        issues.append(WorkflowIssue("error", "unsupported_activation_quantizer", "Unknown activation quantizer."))
    if activation_quantizer == "gemma_mobile_srq" and (
        scale_mode != "retained_mobile" or qat.get("activation_bits", 8) != 8
        or qat.get("activation_symmetric", True) is not True
    ):
        issues.append(WorkflowIssue("error", "mobile_srq_contract_mismatch", "Mobile SRQ requires symmetric retained-mobile A8."))
    if qat.get("simulate_frozen_activations", False) and (
        activation_quantizer != "gemma_mobile_srq"
        or _positive_int(qat.get("expected_frozen_activation_modules")) != 70
        or qat.get("require_lora_trainable_scope") is not True
    ):
        issues.append(WorkflowIssue("error", "frozen_mobile_activation_contract_mismatch", "Frozen mobile SRQ requires all 70 inventoried W8 paths and strict LoRA-only trainables."))
    # A new pipeline run must not silently fall back to the historical A8 path.
    # Saved legacy configurations remain valid and retain their old semantics.
    if _section(config, "run").get("purpose") == "e2b_retained_mobile_golden_bixby_no_mtp_v2" and (
        activation_quantizer != "gemma_mobile_srq"
        or qat.get("simulate_frozen_activations") is not True
        or _positive_int(qat.get("expected_frozen_activation_modules")) != 70
        or qat.get("require_lora_trainable_scope") is not True
    ):
        issues.append(WorkflowIssue("error", "official_mobile_v2_activation_contract_required",
            "Official mobile v2 requires explicit SRQ, all 70 frozen A8 paths and strict LoRA-only trainables."))
    if ste_gradient not in {"identity", "clipped"}:
        issues.append(
            WorkflowIssue(
                "error",
                "unsupported_qat_ste_gradient",
                "qat.ste_gradient must be identity or clipped.",
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
        if model_id == OFFICIAL_MOBILE_MODEL_ID.lower():
            retained_contract = (
                qat.get("mobile_qparams_contract")
                or model.get("mobile_qparams_contract")
            )
            if scale_mode != "retained_mobile":
                issues.append(
                    WorkflowIssue(
                        "error",
                        "gemma4_mobile_retained_scale_mode_required",
                        "The reconstructed mobile seed must retain Google's "
                        "published scales; dynamic abs-max scale recomputation "
                        "corrupts W2/W4 cell centers.",
                    )
                )
            if not str(retained_contract or "").strip():
                issues.append(
                    WorkflowIssue(
                        "error",
                        "gemma4_mobile_qparams_contract_required",
                        "Configure the hash-bound mobile_qparams.json emitted "
                        "with the reconstructed seed.",
                    )
                )
            if qat.get("fixed_scale_required") is not True:
                issues.append(
                    WorkflowIssue(
                        "error",
                        "gemma4_mobile_fixed_scale_required",
                        "Every effective LoRA projection must bind an exact "
                        "published weight scale before training.",
                    )
                )
            if qat.get("fixed_activation_scale_required") is not True:
                issues.append(
                    WorkflowIssue(
                        "error",
                        "gemma4_mobile_fixed_activation_scale_required",
                        "Every effective LoRA projection must use its published "
                        "static A8 input/output scales.",
                    )
                )
            if qat.get("effective_lora_only") is not True:
                issues.append(
                    WorkflowIssue(
                        "error",
                        "gemma4_mobile_effective_lora_only_required",
                        "Frozen mobile cell centers must remain unchanged; fake "
                        "quantize only base+LoRA projections with retained qparams.",
                    )
                )
            try:
                expected_lora_modules = int(
                    qat.get("expected_effective_lora_modules", 0) or 0
                )
            except (TypeError, ValueError):
                expected_lora_modules = 0
            if expected_lora_modules != 205:
                issues.append(
                    WorkflowIssue(
                        "error",
                        "gemma4_mobile_exact_lora_scope_required",
                        "The retained-scale mobile profile must bind exactly "
                        "205 q/k/v/o and gate/up/down projections.",
                    )
                )
            if ste_gradient != "clipped":
                issues.append(
                    WorkflowIssue(
                        "error",
                        "gemma4_mobile_clipped_ste_required",
                        "Immutable W2/W4 scales require a saturation-aware clipped STE.",
                    )
                )
            preflight = (
                config.get("preflight")
                if isinstance(config.get("preflight"), dict)
                else {}
            )
            if preflight.get("require_zero_adapter_parity") is not True:
                issues.append(
                    WorkflowIssue(
                        "error",
                        "gemma4_mobile_zero_adapter_parity_required",
                        "Training must compare QAT-off and zero-adapter QAT-on "
                        "loss/logits before optimizer step 1.",
                    )
                )
            if preflight.get("require_initial_loss_gate") is not True:
                issues.append(
                    WorkflowIssue(
                        "error",
                        "gemma4_mobile_initial_loss_gate_required",
                        "Training must fail before step 1 on catastrophic or "
                        "non-finite completion loss.",
                    )
                )
            try:
                top1_floor = float(
                    preflight.get("min_top1_probe_match", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                top1_floor = 0.0
            if not diagnostic_numeric_policy and top1_floor < 0.90:
                issues.append(
                    WorkflowIssue(
                        "error",
                        "gemma4_mobile_top1_probe_gate_too_weak",
                        "Retained-scale QAT must keep at least 90% of fixed "
                        "teacher-forced top-1 probes before optimizer step 1.",
                    )
                )
            if (
                preflight.get("require_greedy_determinism") is not True
                or _positive_int(preflight.get("greedy_probe_rows")) < 1
                or _positive_int(preflight.get("greedy_probe_new_tokens")) < 8
                or _positive_int(preflight.get("min_greedy_tokens")) < 8
                or (
                    not diagnostic_numeric_policy
                    and _positive_int(preflight.get("min_baseline_qat_greedy_prefix_tokens")) < 8
                )
            ):
                issues.append(
                    WorkflowIssue(
                        "error",
                        "gemma4_mobile_greedy_preflight_required",
                        "Before Trainer construction, zero-adapter QAT must "
                        "repeat a deterministic nontrivial greedy-generation probe.",
                    )
                )
            try:
                eval_steps = int(training.get("eval_steps", 0) or 0)
                save_steps = int(training.get("save_steps", 0) or 0)
            except (TypeError, ValueError):
                eval_steps = save_steps = 0
            if eval_steps < 1 or save_steps != eval_steps:
                issues.append(
                    WorkflowIssue(
                        "error",
                        "gemma4_mobile_checkpoint_eval_cadence_mismatch",
                        "save_steps must equal positive eval_steps so every "
                        "Golden-best adapter receives immediate provenance.",
                    )
                )
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
                and (
                    qat.get("quantize_embeddings") is True
                    or (
                        scale_mode == "retained_mobile"
                        and qat.get("effective_lora_only") is True
                    )
                )
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
