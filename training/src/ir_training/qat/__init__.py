"""True fake-quantization-aware training helpers.

The existing :mod:`ir_training.qat_mtp` package describes Google's
QAT-derived checkpoints and MTP deployment guards.  This package is the
separate, opt-in training path that applies differentiable fake quantization
to the base linear layers while LoRA is trained.
"""

from ir_training.qat.fake_quant import QATController, QATSpec, prepare_qat_model
from ir_training.qat.workflow import validate_qat_config

__all__ = [
    "QATController",
    "QATSpec",
    "prepare_qat_model",
    "validate_qat_config",
]
