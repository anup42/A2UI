"""True fake-quantization-aware training helpers.

The existing :mod:`ir_training.qat_mtp` package describes Google's
QAT-derived checkpoints and MTP deployment guards.  This package is the
separate, opt-in training path that applies differentiable fake quantization
to the base linear layers while LoRA is trained.
"""

from ir_training.qat.fake_quant import QATController, QATSpec, prepare_qat_model
from ir_training.qat.mobile_schema import (
    PUBLIC_GEMMA4_E2B_MOBILE_SCHEMA,
    MobileQuantSchema,
    audit_module_names,
    compare_to_public_schema,
    load_mobile_quant_schema,
)
from ir_training.qat.workflow import validate_qat_config

__all__ = [
    "QATController",
    "QATSpec",
    "prepare_qat_model",
    "PUBLIC_GEMMA4_E2B_MOBILE_SCHEMA",
    "MobileQuantSchema",
    "audit_module_names",
    "compare_to_public_schema",
    "load_mobile_quant_schema",
    "validate_qat_config",
]
