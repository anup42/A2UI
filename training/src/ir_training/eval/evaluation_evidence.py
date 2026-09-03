from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ir_training.eval.tensorboard_logging import artifact_identity


class EvaluationEvidenceError(ValueError):
    """Raised when saved evaluation evidence does not bind to current bytes."""


def verify_evaluation_evidence(
    *,
    aggregate_path: str | Path,
    artifact_path: str | Path,
    artifact_field: str,
    evaluation_name: str,
    run_id: str,
    required_rows: int,
    tensorboard_run_dir: str | Path,
    golden_split_path: str | Path,
    golden_split_sha256: str,
    metric_version: str,
    expected_step: int,
    package_inspection_path: str | Path | None = None,
    package_precision_contract: str | None = None,
) -> dict[str, Any]:
    """Verify aggregate, result, TensorBoard record, and artifact identities."""

    aggregate_file = Path(aggregate_path).expanduser().resolve()
    artifact = Path(artifact_path).expanduser().resolve()
    golden_split = Path(golden_split_path).expanduser().resolve()
    expected_tb_run = Path(tensorboard_run_dir).expanduser().resolve()
    aggregate = _load_json_object(aggregate_file, label="evaluation aggregate")
    if aggregate.get("count") != int(required_rows):
        raise EvaluationEvidenceError(
            f"Evaluation aggregate must contain count={required_rows}: {aggregate_file}"
        )

    result_path = aggregate_file.parent / "evaluation_result.json"
    result = _load_json_object(result_path, label="evaluation result")
    if result.get("row_count") != int(required_rows):
        raise EvaluationEvidenceError(
            f"Evaluation result must contain row_count={required_rows}: {result_path}"
        )
    if result.get("aggregate") != aggregate:
        raise EvaluationEvidenceError(
            f"Evaluation result aggregate differs from {aggregate_file}."
        )
    result_artifact_key = "checkpoint" if artifact_field == "checkpoint" else "model"
    result_artifact = result.get(result_artifact_key)
    if not result_artifact or Path(str(result_artifact)).expanduser().resolve() != artifact:
        raise EvaluationEvidenceError(
            f"Evaluation result does not identify the current {artifact_field} artifact."
        )

    record_value = result.get("tensorboard_record")
    if not record_value:
        raise EvaluationEvidenceError(
            f"Evaluation result has no TensorBoard record: {result_path}"
        )
    record_path = Path(str(record_value)).expanduser().resolve()
    try:
        record_path.relative_to(expected_tb_run)
    except ValueError as exc:
        raise EvaluationEvidenceError(
            f"TensorBoard record is outside run directory {expected_tb_run}: {record_path}"
        ) from exc
    record = _load_json_object(record_path, label="TensorBoard evaluation record")
    if record.get("run_id") != run_id or record.get("evaluation_name") != evaluation_name:
        raise EvaluationEvidenceError(
            "TensorBoard record run/evaluation identity does not match the scorecard lane."
        )
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    if metadata.get("rows") != int(required_rows):
        raise EvaluationEvidenceError(
            f"TensorBoard record must contain rows={required_rows}: {record_path}"
        )
    if str(metadata.get("metric_version") or "") != str(metric_version):
        raise EvaluationEvidenceError(
            "TensorBoard metric version does not match the scorecard contract."
        )
    if record.get("step") != int(expected_step):
        raise EvaluationEvidenceError(
            f"TensorBoard step does not match selected checkpoint step {expected_step}."
        )
    if record.get("metrics") != aggregate:
        raise EvaluationEvidenceError(
            f"TensorBoard record metrics differ from {aggregate_file}."
        )
    source_aggregate = (
        record.get("source_aggregate")
        if isinstance(record.get("source_aggregate"), dict)
        else {}
    )
    aggregate_sha256 = _sha256_file(aggregate_file)
    if source_aggregate.get("sha256") != aggregate_sha256:
        raise EvaluationEvidenceError(
            "TensorBoard record does not bind the current aggregate bytes."
        )

    current_identity = artifact_identity(artifact)
    recorded_artifacts = (
        record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
    )
    recorded_identity = recorded_artifacts.get(artifact_field)
    if not isinstance(recorded_identity, dict):
        raise EvaluationEvidenceError(
            f"TensorBoard record has no {artifact_field!r} artifact identity."
        )
    for key in ("path", "exists", "kind", "size_bytes", "sha256"):
        if recorded_identity.get(key) != current_identity.get(key):
            raise EvaluationEvidenceError(
                f"TensorBoard {artifact_field} identity mismatch for {key}."
            )
    if current_identity.get("exists") is not True or not current_identity.get("sha256"):
        raise EvaluationEvidenceError(
            f"Evaluation artifact is absent or unhashable: {artifact}"
        )

    current_golden_identity = artifact_identity(golden_split)
    recorded_golden_identity = recorded_artifacts.get("golden_split")
    if not isinstance(recorded_golden_identity, dict):
        raise EvaluationEvidenceError(
            "TensorBoard record has no pinned Golden-split artifact identity."
        )
    expected_golden_sha256 = str(golden_split_sha256 or "").strip().lower()
    if (
        not _is_sha256(expected_golden_sha256)
        or current_golden_identity.get("kind") != "file"
        or current_golden_identity.get("sha256") != expected_golden_sha256
    ):
        raise EvaluationEvidenceError(
            f"Current Golden split does not match the pinned digest: {golden_split}"
        )
    for key in ("path", "exists", "kind", "size_bytes", "sha256"):
        if recorded_golden_identity.get(key) != current_golden_identity.get(key):
            raise EvaluationEvidenceError(
                f"TensorBoard Golden-split identity mismatch for {key}."
            )
    if Path(str(metadata.get("split") or "")).expanduser().resolve() != golden_split:
        raise EvaluationEvidenceError(
            "TensorBoard metadata does not identify the pinned Golden split."
        )

    package_evidence = None
    if package_inspection_path is not None:
        package_path = Path(package_inspection_path).expanduser().resolve()
        package = _load_json_object(package_path, label="LiteRT-LM package inspection")
        inspected_path = package.get("path")
        sections = package.get("sections") if isinstance(package.get("sections"), list) else []
        weights = package.get("weights") if isinstance(package.get("weights"), dict) else {}
        if (
            not inspected_path
            or Path(str(inspected_path)).expanduser().resolve() != artifact
            or package.get("file_size") != artifact.stat().st_size
            or package.get("sha256") != current_identity.get("sha256")
            or weights.get("hashed") is not True
            or not sections
            or any(not _is_sha256(item.get("sha256")) for item in sections if isinstance(item, dict))
            or any(not isinstance(item, dict) for item in sections)
        ):
            raise EvaluationEvidenceError(
                f"Package inspection does not bind every section of {artifact}."
            )
        package_evidence = {
            "path": str(package_path),
            "sha256": _sha256_file(package_path),
            "section_count": len(sections),
        }
        if package_precision_contract is not None:
            package_evidence["precision_contract"] = (
                validate_litertlm_precision_contract(
                    package,
                    expected_format=package_precision_contract,
                )
            )

    return {
        "aggregate": str(aggregate_file),
        "aggregate_sha256": aggregate_sha256,
        "evaluation_result": str(result_path),
        "evaluation_result_sha256": _sha256_file(result_path),
        "tensorboard_record": str(record_path),
        "tensorboard_record_sha256": _sha256_file(record_path),
        "tensorboard_step": record.get("step"),
        "artifact": current_identity,
        "golden_split": current_golden_identity,
        "package_inspection": package_evidence,
    }


def validate_litertlm_precision_contract(
    package: dict[str, Any],
    *,
    expected_format: str,
) -> dict[str, Any]:
    """Fail closed when an inspected package does not match its score lane.

    File hashes prove which bytes were evaluated, but a hash alone cannot show
    that a package placed in the W4 directory is actually W4.  The inspector's
    embedded-TFLite summaries provide an independent tensor-type and
    quantization-layout observation.  This gate intentionally checks stored
    constant tensors in the observable graph rather than trusting filenames,
    activation/scratch tensor types, or converter commands.
    """

    contract = str(expected_format or "").strip().lower()
    supported = {"w32", "w16", "w8", "w4", "mixed_w4_w8"}
    if contract not in supported:
        raise EvaluationEvidenceError(
            f"Unsupported LiteRT-LM precision contract: {expected_format!r}."
        )

    graphs = package.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise EvaluationEvidenceError(
            "Package precision cannot be proven without embedded TFLite graph inspection."
        )

    type_counts: dict[str, int] = {}
    layout_type_counts: dict[str, int] = {}
    quantized_tensor_count = 0
    for index, graph in enumerate(graphs):
        if not isinstance(graph, dict) or graph.get("available") is not True:
            raise EvaluationEvidenceError(
                f"Embedded TFLite graph {index} was not inspectable."
            )
        summary = graph.get("summary")
        if not isinstance(summary, dict):
            raise EvaluationEvidenceError(
                f"Embedded TFLite graph {index} has no precision summary."
            )
        histogram = summary.get("constant_tensor_type_histogram")
        layouts = summary.get("constant_quantization_layout_histogram")
        if not isinstance(histogram, dict) or not isinstance(layouts, dict):
            raise EvaluationEvidenceError(
                f"Embedded TFLite graph {index} has incomplete constant-weight "
                "precision histograms."
            )
        for tensor_type, count in histogram.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise EvaluationEvidenceError(
                    f"Invalid tensor-type count in embedded graph {index}."
                )
            name = str(tensor_type).upper()
            type_counts[name] = type_counts.get(name, 0) + count
        graph_quantized = summary.get("constant_quantized_tensor_count")
        if (
            isinstance(graph_quantized, bool)
            or not isinstance(graph_quantized, int)
            or graph_quantized < 0
        ):
            raise EvaluationEvidenceError(
                f"Invalid constant quantized-tensor count in embedded graph {index}."
            )
        quantized_tensor_count += graph_quantized
        for layout, count in layouts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise EvaluationEvidenceError(
                    f"Invalid quantization-layout count in embedded graph {index}."
                )
            tensor_type = str(layout).split("|", 1)[0].upper()
            layout_type_counts[tensor_type] = (
                layout_type_counts.get(tensor_type, 0) + count
            )

    present = {name for name, count in type_counts.items() if count > 0}
    layout_present = {
        name for name, count in layout_type_counts.items() if count > 0
    }
    low4 = {"INT4", "UINT4"}
    low_bits = low4 | {"INT2", "UINT2", "INT8", "UINT8"}

    def require_types(required: set[str], *, layouts: bool = False) -> None:
        observed = layout_present if layouts else present
        if not required.issubset(observed):
            location = "quantization layouts" if layouts else "tensor types"
            missing = ", ".join(sorted(required - observed))
            raise EvaluationEvidenceError(
                f"LiteRT-LM {contract} precision mismatch: missing {missing} in {location}."
            )

    def require_any(required: set[str], *, layouts: bool = False) -> None:
        observed = layout_present if layouts else present
        if not observed.intersection(required):
            location = "quantization layouts" if layouts else "tensor types"
            raise EvaluationEvidenceError(
                f"LiteRT-LM {contract} precision mismatch: expected one of "
                f"{', '.join(sorted(required))} in {location}."
            )

    if contract == "w32":
        require_types({"FLOAT32"})
        forbidden = layout_present.intersection(low_bits)
        if "FLOAT16" in present:
            forbidden.add("FLOAT16")
    elif contract == "w16":
        require_types({"FLOAT16"})
        forbidden = layout_present.intersection(low_bits)
    elif contract == "w8":
        require_types({"INT8"})
        require_types({"INT8"}, layouts=True)
        forbidden = layout_present.intersection(low4 | {"INT2", "UINT2"})
    elif contract == "w4":
        require_any(low4)
        require_any(low4, layouts=True)
        forbidden = layout_present.intersection({"INT2", "UINT2"})
    else:
        require_any(low4)
        require_any(low4, layouts=True)
        require_types({"INT8"})
        require_types({"INT8"}, layouts=True)
        forbidden = layout_present.intersection({"INT2", "UINT2"})

    if forbidden:
        raise EvaluationEvidenceError(
            f"LiteRT-LM {contract} precision mismatch: forbidden precision types "
            f"{', '.join(sorted(forbidden))}."
        )
    if contract in {"w8", "w4", "mixed_w4_w8"} and quantized_tensor_count < 1:
        raise EvaluationEvidenceError(
            f"LiteRT-LM {contract} precision mismatch: no quantized tensors observed."
        )

    return {
        "expected_format": contract,
        "graph_count": len(graphs),
        "constant_tensor_type_histogram": dict(sorted(type_counts.items())),
        "constant_quantization_layout_type_histogram": dict(
            sorted(layout_type_counts.items())
        ),
        "constant_quantized_tensor_count": quantized_tensor_count,
        "verified": True,
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationEvidenceError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationEvidenceError(f"Invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationEvidenceError(f"{label.capitalize()} must be a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)
