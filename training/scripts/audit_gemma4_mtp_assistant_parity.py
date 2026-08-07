"""Audit whether a public Gemma 4 assistant checkpoint explains MTP constants.

The released LiteRT-LM ``tf_lite_mtp_drafter`` section contains packed W4/W8
weights.  Public Gemma 4 assistant checkpoints contain BF16 weights, so this
script applies the observable public symmetric per-output-channel min/max
candidate and compares the resulting packed bytes/scales with the official
MTP section.  It is an evidence tool, not a claim that the private assistant
trainer or exporter has been reconstructed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_gemma4_mobile_checkpoint_parity import (  # noqa: E402
    _records,
    _section_by_model_type,
)
from build_converter_topology_parity import _read_section, _unpack_model  # noqa: E402
from build_random_official_topology_parity import _pack_low_bit  # noqa: E402
from ir_training.export.litertlm_inspector import (  # noqa: E402
    LiteRTLMInspectionError,
    inspect_litertlm,
)


class MTPAssistantParityError(RuntimeError):
    """Raised when the assistant source cannot be audited."""


_LAYER_KEYS = (
    "self_attn.q_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def _source_keys() -> list[str]:
    keys = ["pre_projection.weight"]
    for layer in range(4):
        keys.extend(f"model.layers.{layer}.{suffix}" for suffix in _LAYER_KEYS)
    keys.extend(["model.embed_tokens.weight", "post_projection.weight"])
    return keys


def _pack_source(values: np.ndarray, bits: int) -> tuple[bytes, np.ndarray]:
    if values.ndim != 2:
        raise MTPAssistantParityError(f"Expected rank-2 source weight, got {values.shape}.")
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    scales = np.max(np.abs(values), axis=1).astype(np.float32) / float(qmax)
    scales = np.maximum(scales, np.finfo(np.float32).tiny)
    quantized = np.rint(values / scales[:, None])
    quantized = np.clip(quantized, qmin, qmax).astype(np.int8, copy=False)
    if bits == 8:
        return quantized.tobytes(order="C"), scales
    return _pack_low_bit(quantized.reshape(-1), bits).tobytes(), scales


def _load_source(source: Path) -> tuple[Any, set[str]]:
    try:
        from safetensors import safe_open
        import torch
    except ImportError as exc:  # pragma: no cover - optional audit dependency
        raise MTPAssistantParityError(
            "The assistant audit requires safetensors and torch."
        ) from exc
    try:
        handle = safe_open(str(source), framework="pt", device="cpu")
        keys = set(handle.keys())
    except Exception as exc:  # pragma: no cover - source/tool dependent
        raise MTPAssistantParityError(f"Could not open assistant safetensors: {exc}") from exc

    def get(key: str) -> np.ndarray:
        if key not in keys:
            raise KeyError(key)
        tensor = handle.get_tensor(key)
        return tensor.to(dtype=torch.float32).detach().cpu().numpy()

    return get, keys


def run(
    artifact: str | Path,
    assistant_source: str | Path,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    artifact_path = Path(artifact).expanduser().resolve()
    source_path = Path(assistant_source).expanduser().resolve()
    if not artifact_path.is_file():
        raise MTPAssistantParityError(f"Official artifact does not exist: {artifact_path}")
    if not source_path.is_file():
        raise MTPAssistantParityError(f"Assistant safetensors does not exist: {source_path}")
    try:
        package = inspect_litertlm(artifact_path, inspect_tflite=False)
        section = _section_by_model_type(package, "tf_lite_mtp_drafter")
    except (OSError, LiteRTLMInspectionError) as exc:
        raise MTPAssistantParityError(f"Could not inspect official artifact: {exc}") from exc
    section_bytes = _read_section(artifact_path, section)
    model, _, _ = _unpack_model(section_bytes)
    records = list(_records(model, section_bytes))
    source_get, source_keys = _load_source(source_path)
    expected_keys = _source_keys()

    # The official MTP traversal is stable for the released package.  Keep the
    # mapping explicit and fail loudly if a future package changes its order.
    if len(records) != len(expected_keys):
        raise MTPAssistantParityError(
            f"Expected {len(expected_keys)} unique MTP weights, found {len(records)}."
        )

    comparisons: list[dict[str, Any]] = []
    exact_raw = 0
    exact_scales = 0
    shape_matches = 0
    for ordinal, (record, source_key) in enumerate(zip(records, expected_keys)):
        source = None
        error = None
        try:
            source = source_get(source_key)
        except Exception as exc:  # pragma: no cover - malformed source dependent
            error = f"source_read:{exc}"
        candidate_raw = None
        candidate_scales = None
        shape_match = False
        raw_match = False
        scale_match = False
        if source is not None:
            shape_match = tuple(source.shape) == tuple(record["shape"])
            if shape_match:
                shape_matches += 1
                candidate_raw, candidate_scales = _pack_source(source, int(record["bits"]))
                raw_match = bytes(record["raw"]) == candidate_raw
                scale_match = bool(
                    np.array_equal(
                        np.asarray(record["scales"], dtype=np.float32), candidate_scales
                    )
                )
                exact_raw += int(raw_match)
                exact_scales += int(scale_match)
            else:
                error = f"shape:{tuple(source.shape)} != {tuple(record['shape'])}"
        comparisons.append(
            {
                "ordinal": ordinal,
                "source_key": source_key,
                "bits": int(record["bits"]),
                "shape": list(record["shape"]),
                "shape_match": shape_match,
                "raw_match": raw_match,
                "scale_match": scale_match,
                "official_buffer_bytes": len(record["raw"]),
                "source_error": error,
            }
        )

    result = {
        "artifact": str(artifact_path),
        "assistant_source": str(source_path),
        "model_type": "tf_lite_mtp_drafter",
        "training_executed": False,
        "private_mtp_recipe_recovered": False,
        "source_tensor_count": len(source_keys),
        "expected_mtp_source_tensor_count": len(expected_keys),
        "official_inventory_count": len(records),
        "shape_matches": shape_matches,
        "exact_raw_matches": exact_raw,
        "exact_scale_matches": exact_scales,
        "all_candidate_constants_exact": exact_raw == len(records)
        and exact_scales == len(records),
        "candidate_formula": "per-output-channel max(abs(row))/(2^(bits-1)-1), symmetric signed packing",
        "comparisons": comparisons,
        "interpretation": (
            "The public assistant source and the observable min/max candidate explain every "
            "official MTP constant."
            if exact_raw == len(records) and exact_scales == len(records)
            else "The public assistant source does not reproduce the official MTP constants "
            "under the public symmetric min/max candidate; the released drafter therefore "
            "still requires a private/other source checkpoint, QAT scales, or exporter rule."
        ),
    }
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["output"] = str(output_path)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="Official Gemma 4 .litertlm package.")
    parser.add_argument("assistant_source", help="Public assistant model.safetensors file.")
    parser.add_argument("--output")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run(args.artifact, args.assistant_source, output=args.output)
    except (OSError, MTPAssistantParityError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
