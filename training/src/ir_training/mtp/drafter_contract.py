"""Observable Gemma 4 E2B mobile drafter weight contract.

The released ``tf_lite_mtp_drafter`` graph contains 23 independently packed
W4/W8 matrices.  Their traversal order and shapes were recovered from the
public LiteRT-LM artifact and matched to the public Transformers assistant
checkpoint.  This module is the shared mapping authority for audits, QAT, and
exact-topology export.

It does not encode Google's private training data, loss weighting, optimizer,
or observer schedule.  Those properties are not present in the artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

# Google explicitly requires a QAT target to use the matching QAT assistant at
# the same precision.  This is the public floating checkpoint used to seed the
# optional trained-drafter reconstruction; ``weight_source=official`` still
# preserves the assistant section compiled into the released LiteRT-LM package.
OFFICIAL_GEMMA4_E2B_ASSISTANT = (
    "google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant"
)
GEMMA4_E2B_MTP_MODEL_TYPE = "tf_lite_mtp_drafter"


@dataclass(frozen=True)
class DrafterWeightSpec:
    ordinal: int
    source_key: str
    module_name: str
    bits: int
    shape: tuple[int, int]


def deployment_weight_specs() -> tuple[DrafterWeightSpec, ...]:
    """Return the exact 23-weight order observed in the released E2B graph."""

    entries: list[tuple[str, str, int, tuple[int, int]]] = [
        ("pre_projection.weight", "pre_projection", 8, (256, 3072)),
    ]
    for layer in range(4):
        attention_width = 2048 if layer == 3 else 1024
        prefix = f"model.layers.{layer}"
        entries.extend(
            [
                (
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.q_proj",
                    8,
                    (attention_width, 256),
                ),
                (
                    f"{prefix}.self_attn.o_proj.weight",
                    f"{prefix}.self_attn.o_proj",
                    8,
                    (256, attention_width),
                ),
                (
                    f"{prefix}.mlp.gate_proj.weight",
                    f"{prefix}.mlp.gate_proj",
                    4,
                    (2048, 256),
                ),
                (
                    f"{prefix}.mlp.up_proj.weight",
                    f"{prefix}.mlp.up_proj",
                    4,
                    (2048, 256),
                ),
                (
                    f"{prefix}.mlp.down_proj.weight",
                    f"{prefix}.mlp.down_proj",
                    4,
                    (256, 2048),
                ),
            ]
        )
    entries.extend(
        [
            (
                "model.embed_tokens.weight",
                "model.embed_tokens",
                4,
                (262144, 256),
            ),
            ("post_projection.weight", "post_projection", 8, (1536, 256)),
        ]
    )
    return tuple(
        DrafterWeightSpec(
            ordinal=ordinal,
            source_key=source_key,
            module_name=module_name,
            bits=bits,
            shape=shape,
        )
        for ordinal, (source_key, module_name, bits, shape) in enumerate(entries)
    )


def source_keys() -> list[str]:
    return [item.source_key for item in deployment_weight_specs()]


def module_bits() -> dict[str, int]:
    """Map trainable module names to their released mobile weight widths."""

    return {item.module_name: item.bits for item in deployment_weight_specs()}


def contract_summary() -> dict[str, object]:
    specs = deployment_weight_specs()
    histogram: dict[str, int] = {}
    for item in specs:
        label = str(item.bits)
        histogram[label] = histogram.get(label, 0) + 1
    return {
        "model_type": GEMMA4_E2B_MTP_MODEL_TYPE,
        "weight_count": len(specs),
        "bit_histogram": histogram,
        "source_keys": [item.source_key for item in specs],
        "shapes": [list(item.shape) for item in specs],
    }
