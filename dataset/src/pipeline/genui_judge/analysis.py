"""Statistical analysis and calibration for the single-Codex benchmark."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean, median, pstdev, stdev
from typing import Any, Callable, Iterable, Mapping, Sequence


ANALYSIS_SCHEMA_VERSION = "genui_single_codex_analysis.v2"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return (
        ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    )


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if math.isfinite(value)]
    return {
        "count": len(finite),
        "mean": mean(finite) if finite else 0.0,
        "median": median(finite) if finite else 0.0,
        "sd_population": pstdev(finite) if len(finite) > 1 else 0.0,
        "minimum": min(finite) if finite else 0.0,
        "p05": _quantile(finite, 0.05),
        "p25": _quantile(finite, 0.25),
        "p75": _quantile(finite, 0.75),
        "p95": _quantile(finite, 0.95),
        "maximum": max(finite) if finite else 0.0,
    }


def _metric_vector_stats(
    predictions: Sequence[float],
    targets: Sequence[float],
) -> dict[str, float | int]:
    import numpy as np
    x = np.asarray(predictions, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ValueError("paired vectors must have equal length >= 2")
    differences = x - y
    pearson = _safe_pearson(x, y)
    spearman = _safe_spearman(x, y)
    kendall = _safe_kendall(x, y)
    return {
        "count": int(len(x)),
        "pearson_r": float(pearson),
        "spearman_rho": float(spearman),
        "kendall_tau": float(kendall),
        "mae": float(np.mean(np.abs(differences))),
        "rmse": float(np.sqrt(np.mean(differences**2))),
        "signed_bias_prediction_minus_target": float(
            np.mean(differences)
        ),
    }


def _safe_pearson(left: Sequence[float], right: Sequence[float]) -> float:
    import numpy as np
    from scipy import stats

    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) < 2 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return 0.0
    value = float(stats.pearsonr(x, y).statistic)
    return value if math.isfinite(value) else 0.0


def _safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    import numpy as np
    from scipy import stats

    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) < 2 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return 0.0
    value = float(stats.spearmanr(x, y).statistic)
    return value if math.isfinite(value) else 0.0


def _safe_kendall(left: Sequence[float], right: Sequence[float]) -> float:
    import numpy as np
    from scipy import stats

    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) < 2 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return 0.0
    value = float(stats.kendalltau(x, y).statistic)
    return value if math.isfinite(value) else 0.0


def _bootstrap_interval(
    predictions: Sequence[float],
    targets: Sequence[float],
    statistic: Callable[[Sequence[float], Sequence[float]], float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    import numpy as np

    x = np.asarray(predictions, dtype=float)
    y = np.asarray(targets, dtype=float)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(x), size=len(x))
        value = float(statistic(x[indices], y[indices]))
        if math.isfinite(value):
            estimates.append(value)
    if not estimates:
        return {
            "iterations_requested": iterations,
            "iterations_finite": 0,
            "lower_95": 0.0,
            "upper_95": 0.0,
        }
    return {
        "iterations_requested": iterations,
        "iterations_finite": len(estimates),
        "lower_95": _quantile(estimates, 0.025),
        "upper_95": _quantile(estimates, 0.975),
    }


def _paired_bootstrap(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np
    statistics: dict[str, Callable[[Sequence[float], Sequence[float]], float]] = {
        "pearson_r": _safe_pearson,
        "spearman_rho": _safe_spearman,
        "kendall_tau": _safe_kendall,
        "mae": lambda x, y: float(
            np.mean(np.abs(np.asarray(x) - np.asarray(y)))
        ),
        "rmse": lambda x, y: float(
            np.sqrt(
                np.mean(
                    (np.asarray(x, dtype=float) - np.asarray(y, dtype=float))
                    ** 2
                )
            )
        ),
        "signed_bias_prediction_minus_target": lambda x, y: float(
            np.mean(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))
        ),
    }
    return {
        name: _bootstrap_interval(
            predictions,
            targets,
            function,
            iterations=iterations,
            seed=seed + index * 997,
        )
        for index, (name, function) in enumerate(statistics.items())
    }


def _icc_absolute_agreement(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """ICC(A,1): two-way mixed, single-measure absolute agreement."""

    import numpy as np

    ratings = np.column_stack(
        [np.asarray(first, dtype=float), np.asarray(second, dtype=float)]
    )
    n, k = ratings.shape
    if n < 2:
        return 0.0
    row_means = ratings.mean(axis=1)
    column_means = ratings.mean(axis=0)
    grand = ratings.mean()
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_columns = n * np.sum((column_means - grand) ** 2) / (k - 1)
    residual = ratings - row_means[:, None] - column_means[None, :] + grand
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    denominator = (
        ms_rows
        + (k - 1) * ms_error
        + (k * (ms_columns - ms_error) / n)
    )
    if abs(float(denominator)) < 1e-12:
        return 1.0 if np.allclose(ratings[:, 0], ratings[:, 1]) else 0.0
    return float((ms_rows - ms_error) / denominator)


def _repeat_statistics(
    repeat_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    original = [
        float(row["original_composite_0_100"]) for row in repeat_rows
    ]
    repeated = [
        float(row["repeat_composite_0_100"]) for row in repeat_rows
    ]
    if len(original) < 2:
        return {"count": len(original), "complete": False}
    base = _metric_vector_stats(repeated, original)
    differences = [
        repeated[index] - original[index]
        for index in range(len(original))
    ]
    bias = mean(differences)
    difference_sd = stdev(differences) if len(differences) > 1 else 0.0
    return {
        **base,
        "complete": len(original) == 96,
        "icc_a_1_absolute_agreement": _icc_absolute_agreement(
            original, repeated
        ),
        "bland_altman": {
            "signed_bias": bias,
            "difference_sd_sample": difference_sd,
            "lower_limit_95": bias - 1.96 * difference_sd,
            "upper_limit_95": bias + 1.96 * difference_sd,
        },
    }


def _fit_calibrators(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    target_key: str,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if _finite_float(row.get(metric_key)) is not None:
            by_split[str(row["split"])].append(row)
    for split in ("calibration", "validation", "holdout"):
        if not by_split[split]:
            raise ValueError(
                f"no {split} rows available for calibration of {metric_key}"
            )

    def vectors(split: str) -> tuple[Any, Any]:
        x = np.asarray(
            [float(row[metric_key]) for row in by_split[split]],
            dtype=float,
        )
        y = np.asarray(
            [float(row[target_key]) for row in by_split[split]],
            dtype=float,
        )
        return x, y

    x_cal, y_cal = vectors("calibration")
    x_validation, y_validation = vectors("validation")
    x_holdout, y_holdout = vectors("holdout")
    slope, intercept = np.polyfit(x_cal, y_cal, deg=1)
    isotonic = IsotonicRegression(
        y_min=0.0,
        y_max=100.0,
        out_of_bounds="clip",
    )
    isotonic.fit(x_cal, y_cal)

    def identity(values: Any) -> Any:
        return np.clip(values, 0.0, 100.0)

    def affine(values: Any) -> Any:
        return np.clip(slope * values + intercept, 0.0, 100.0)

    def isotonic_predict(values: Any) -> Any:
        return isotonic.predict(values)

    predictors = {
        "identity": identity,
        "affine": affine,
        "isotonic": isotonic_predict,
    }

    def evaluation(
        values: Any,
        targets: Any,
    ) -> dict[str, dict[str, float | int]]:
        return {
            name: _metric_vector_stats(
                predictor(values).tolist(), targets.tolist()
            )
            for name, predictor in predictors.items()
        }

    validation = evaluation(x_validation, y_validation)
    affine_mae = float(validation["affine"]["mae"])
    isotonic_mae = float(validation["isotonic"]["mae"])
    selected = (
        "isotonic"
        if affine_mae - isotonic_mae >= 0.5 - 1e-12
        else "affine"
    )
    selected_holdout_predictions = predictors[selected](x_holdout)
    holdout = _metric_vector_stats(
        selected_holdout_predictions.tolist(),
        y_holdout.tolist(),
    )
    return {
        "metric": metric_key,
        "target": target_key,
        "selection_rule": (
            "Select isotonic only when validation MAE improves by at "
            "least 0.5 points over affine; otherwise select affine."
        ),
        "selected_model": selected,
        "calibration_count": len(x_cal),
        "validation_count": len(x_validation),
        "holdout_count": len(x_holdout),
        "validation": validation,
        "holdout_selected_model_once": holdout,
        "models": {
            "identity": {"kind": "identity", "clip": [0.0, 100.0]},
            "affine": {
                "kind": "affine",
                "slope": float(slope),
                "intercept": float(intercept),
                "clip": [0.0, 100.0],
            },
            "isotonic": {
                "kind": "isotonic",
                "x_thresholds": [
                    float(value) for value in isotonic.X_thresholds_
                ],
                "y_thresholds": [
                    float(value) for value in isotonic.y_thresholds_
                ],
                "out_of_bounds": "clip",
            },
        },
    }


def _milestone_statistics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    intents = sorted({str(row["intent_bucket"]) for row in rows})
    strata = sorted({str(row["selection_stratum"]) for row in rows})
    design: list[list[float]] = []
    values: list[float] = []
    for row in rows:
        design.append(
            [1.0]
            + [
                1.0 if row["intent_bucket"] == value else 0.0
                for value in intents[1:]
            ]
            + [
                1.0 if row["selection_stratum"] == value else 0.0
                for value in strata[1:]
            ]
        )
        values.append(float(row["composite_0_100"]))
    matrix = np.asarray(design, dtype=float)
    target = np.asarray(values, dtype=float)
    coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
    residuals = target - matrix @ coefficients
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    grouped_blocks: dict[int, list[tuple[float, float]]] = defaultdict(list)
    positions: list[float] = []
    for index, row in enumerate(rows):
        grouped[int(row["milestone_index"])].append(
            (float(target[index]), float(residuals[index]))
        )
        grouped_blocks[int(row["block_index"])].append(
            (float(target[index]), float(residuals[index]))
        )
        positions.append(float(row["schedule_position"]))
    milestones = []
    for milestone, pairs in sorted(grouped.items()):
        raw = [value for value, _ in pairs]
        adjusted = [value for _, value in pairs]
        milestones.append(
            {
                "milestone_index": milestone,
                "count": len(pairs),
                "raw": _distribution(raw),
                "composition_adjusted_residual_mean": mean(adjusted),
            }
        )
    adjusted_means = [
        float(row["composition_adjusted_residual_mean"])
        for row in milestones
    ]
    blocks = [
        {
            "block_index": block,
            "count": len(pairs),
            "raw": _distribution([value for value, _ in pairs]),
            "composition_adjusted_residual_mean": mean(
                value for _, value in pairs
            ),
        }
        for block, pairs in sorted(grouped_blocks.items())
    ]
    position_matrix = np.column_stack(
        [np.ones(len(positions)), np.asarray(positions, dtype=float)]
    )
    position_coefficients = np.linalg.lstsq(
        position_matrix,
        residuals,
        rcond=None,
    )[0]
    return {
        "milestones": milestones,
        "blocks": blocks,
        "composition_adjusted_position_slope_per_100_packets": float(
            position_coefficients[1] * 100.0
        ),
        "maximum_absolute_adjusted_drift": (
            max(abs(value) for value in adjusted_means)
            if adjusted_means
            else 0.0
        ),
        "adjusted_drift_range": (
            max(adjusted_means) - min(adjusted_means)
            if adjusted_means
            else 0.0
        ),
        "adjustment": "OLS intent and population/stress fixed effects",
    }


def _judge_task_provenance(
    raw_judgments: Sequence[Mapping[str, Any]],
    schedule_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_packet: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_judgments:
        by_packet[str(row["packet_id"])].append(row)
    milestone_tasks: dict[int, set[str]] = defaultdict(set)
    scheduled_task_ids: set[str] = set()
    adjudication_task_ids: set[str] = set()
    packet_task_mismatches: list[str] = []
    timestamp_order_violations: list[str] = []
    model_identifiers: set[str] = set()
    for packet_id, rows in by_packet.items():
        task_ids = {str(row["judge_task_id"]) for row in rows}
        model_identifiers.update(
            str(row["judge_model_identifier"]) for row in rows
        )
        if len(task_ids) != 1:
            packet_task_mismatches.append(packet_id)
        screen = next(
            (
                row
                for row in rows
                if row.get("pass_type") == "screenshot_only"
            ),
            None,
        )
        source = next(
            (
                row
                for row in rows
                if row.get("pass_type") == "source_conditioned"
            ),
            None,
        )
        if (
            screen is not None
            and source is not None
            and str(screen["judged_at"]) > str(source["judged_at"])
        ):
            timestamp_order_violations.append(packet_id)
        schedule = schedule_map.get(packet_id)
        if schedule is None:
            adjudication_task_ids.update(task_ids)
            continue
        milestone = int(schedule["schedule_position"]) // 80
        milestone_tasks[milestone].update(task_ids)
        scheduled_task_ids.update(task_ids)
    task_to_milestones: dict[str, set[int]] = defaultdict(set)
    for milestone, task_ids in milestone_tasks.items():
        for task_id in task_ids:
            task_to_milestones[task_id].add(milestone)
    reused = {
        task_id: sorted(milestones)
        for task_id, milestones in task_to_milestones.items()
        if len(milestones) > 1
    }
    milestones = [
        {
            "milestone_index": milestone,
            "task_ids": sorted(task_ids),
            "single_task": len(task_ids) == 1,
        }
        for milestone, task_ids in sorted(milestone_tasks.items())
    ]
    return {
        "model_identifiers": sorted(model_identifiers),
        "stable_single_model": len(model_identifiers) == 1,
        "milestones": milestones,
        "single_task_per_milestone": bool(milestones)
        and all(row["single_task"] for row in milestones),
        "task_ids_reused_across_milestones": reused,
        "fresh_task_after_each_80": bool(milestones)
        and all(row["single_task"] for row in milestones)
        and not reused,
        "packet_pass_task_mismatches": sorted(packet_task_mismatches),
        "pass_timestamp_order_violations": sorted(
            timestamp_order_violations
        ),
        "scheduled_task_ids": sorted(scheduled_task_ids),
        "adjudication_task_ids": sorted(adjudication_task_ids),
        "adjudication_uses_fresh_task": not (
            scheduled_task_ids & adjudication_task_ids
        ),
    }


def _make_plots(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    repeat_rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    metric_pairs = [
        ("legacy_score", "Legacy metric"),
        ("v5_4_score", "GenUI metric v5.4"),
    ]
    targets = [
        ("source_representation_0_100", "R"),
        ("rendered_ux_0_100", "U"),
        ("composite_0_100", "J"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for row_index, (metric_key, metric_label) in enumerate(metric_pairs):
        for column_index, (target_key, target_label) in enumerate(targets):
            axis = axes[row_index][column_index]
            paired = [
                (float(row[metric_key]), float(row[target_key]))
                for row in rows
                if _finite_float(row.get(metric_key)) is not None
            ]
            axis.scatter(
                [item[0] for item in paired],
                [item[1] for item in paired],
                alpha=0.35,
                s=14,
            )
            axis.plot([0, 100], [0, 100], "--", color="grey", linewidth=1)
            axis.set(
                xlabel=metric_label,
                ylabel=f"Codex {target_label}",
                xlim=(0, 100),
                ylim=(0, 100),
                title=f"{metric_label} vs {target_label}",
            )
            axis.grid(alpha=0.2)
    correlation_path = output_dir / "metric_correlations.png"
    figure.savefig(correlation_path, dpi=160)
    plt.close(figure)
    paths.append(str(correlation_path))

    if repeat_rows:
        original = [
            float(row["original_composite_0_100"]) for row in repeat_rows
        ]
        repeated = [
            float(row["repeat_composite_0_100"]) for row in repeat_rows
        ]
        averages = [
            (left + right) / 2.0
            for left, right in zip(original, repeated)
        ]
        differences = [
            right - left for left, right in zip(original, repeated)
        ]
        bias = mean(differences)
        sd = stdev(differences) if len(differences) > 1 else 0.0
        figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
        axis.scatter(averages, differences, alpha=0.65)
        axis.axhline(bias, color="black", label=f"Bias {bias:.2f}")
        axis.axhline(
            bias + 1.96 * sd,
            color="red",
            linestyle="--",
            label="95% limits",
        )
        axis.axhline(bias - 1.96 * sd, color="red", linestyle="--")
        axis.set(
            xlabel="Mean of original and repeat J",
            ylabel="Repeat minus original J",
            title="Blind-repeat Bland-Altman",
        )
        axis.legend()
        axis.grid(alpha=0.2)
        repeat_path = output_dir / "repeat_bland_altman.png"
        figure.savefig(repeat_path, dpi=160)
        plt.close(figure)
        paths.append(str(repeat_path))

    milestone_group: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        milestone_group[int(row["milestone_index"])].append(
            float(row["composite_0_100"])
        )
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    keys = sorted(milestone_group)
    axis.plot(
        keys,
        [mean(milestone_group[key]) for key in keys],
        marker="o",
    )
    axis.set(
        xlabel="80-position milestone",
        ylabel="Mean Codex composite J",
        title="Judgment score by milestone",
    )
    axis.grid(alpha=0.2)
    milestone_path = output_dir / "milestone_drift.png"
    figure.savefig(milestone_path, dpi=160)
    plt.close(figure)
    paths.append(str(milestone_path))

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for axis, (metric_key, metric_label) in zip(axes, metric_pairs):
        paired = sorted(
            (
                float(row[metric_key]),
                float(row["composite_0_100"]),
            )
            for row in rows
            if _finite_float(row.get(metric_key)) is not None
        )
        bin_count = min(10, max(1, len(paired)))
        bins: list[list[tuple[float, float]]] = [
            paired[
                index * len(paired) // bin_count :
                (index + 1) * len(paired) // bin_count
            ]
            for index in range(bin_count)
        ]
        bins = [values for values in bins if values]
        raw_means = [
            mean(value[0] for value in values) for values in bins
        ]
        target_means = [
            mean(value[1] for value in values) for values in bins
        ]
        axis.plot(
            raw_means,
            target_means,
            marker="o",
            label="Empirical decile means",
        )
        axis.plot([0, 100], [0, 100], "--", color="grey", label="Identity")
        model = calibration[metric_key]
        selected = str(model["selected_model"])
        if selected == "affine":
            parameters = model["models"]["affine"]
            x_values = [float(value) for value in range(101)]
            y_values = [
                min(
                    100.0,
                    max(
                        0.0,
                        float(parameters["slope"]) * value
                        + float(parameters["intercept"]),
                    ),
                )
                for value in x_values
            ]
        else:
            parameters = model["models"]["isotonic"]
            x_values = [
                float(value) for value in parameters["x_thresholds"]
            ]
            y_values = [
                float(value) for value in parameters["y_thresholds"]
            ]
        axis.plot(
            x_values,
            y_values,
            color="tab:red",
            label=f"Selected {selected}",
        )
        axis.set(
            xlabel=metric_label,
            ylabel="Mean Codex composite J",
            xlim=(0, 100),
            ylim=(0, 100),
            title=f"{metric_label} calibration",
        )
        axis.grid(alpha=0.2)
        axis.legend()
    calibration_path = output_dir / "calibration_curves.png"
    figure.savefig(calibration_path, dpi=160)
    plt.close(figure)
    paths.append(str(calibration_path))
    return paths


def analyze_benchmark(
    benchmark_dir: str | Path,
    *,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 20260729,
) -> dict[str, Any]:
    root = Path(benchmark_dir).resolve()
    output_dir = root / "analysis"
    if output_dir.exists():
        raise FileExistsError(
            f"immutable analysis directory already exists: {output_dir}"
        )
    output_dir.mkdir()
    finalized = _read_jsonl(root / "finalized_groundtruth.jsonl")
    selection = {
        row["ui_id"]: row
        for row in _read_jsonl(root / "selection_manifest.jsonl")
    }
    packet_map = {
        row["packet_id"]: row
        for row in _read_jsonl(
            root / "sealed" / "packet_identity_map.jsonl"
        )
    }
    schedule_map = {
        row["packet_id"]: row
        for row in _read_jsonl(root / "judge_schedule.jsonl")
    }
    v5_rows = {
        row["ui_id"]: row
        for row in _read_jsonl(root / "metric_v5_4_native_scores.jsonl")
    }
    repeat_rows = _read_jsonl(root / "repeat_analysis.jsonl")
    joined: list[dict[str, Any]] = []
    for row in finalized:
        ui_id = str(row["ui_id"])
        selected = selection[ui_id]
        packet = packet_map[str(row["original_packet_id"])]
        schedule_row = schedule_map[str(row["original_packet_id"])]
        v5 = v5_rows.get(ui_id)
        joined.append(
            {
                **row,
                "selection_stratum": selected["selection_stratum"],
                "split": selected["split"],
                "migration_anchor": selected["migration_anchor"],
                "legacy_score": selected.get("legacy_score"),
                "v5_4_score": (
                    v5.get("quality_0_100") if v5 is not None else None
                ),
                "schedule_position": packet["schedule_position"],
                "block_index": schedule_row["block_index"],
                "milestone_index": int(packet["schedule_position"]) // 80,
            }
        )
    _atomic_write_jsonl(output_dir / "analysis_rows.jsonl", joined)

    distributions: dict[str, Any] = {}
    for stratum in ("population", "stress", "combined"):
        subset = (
            joined
            if stratum == "combined"
            else [
                row
                for row in joined
                if row["selection_stratum"] == stratum
            ]
        )
        distributions[stratum] = {
            target: _distribution(
                [float(row[target]) for row in subset]
            )
            for target in (
                "source_representation_0_100",
                "rendered_ux_0_100",
                "composite_0_100",
            )
        }
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_intent[str(row["intent_bucket"])].append(row)
    intent_results = {
        intent: {
            target: _distribution(
                [float(row[target]) for row in rows]
            )
            for target in (
                "source_representation_0_100",
                "rendered_ux_0_100",
                "composite_0_100",
            )
        }
        for intent, rows in sorted(by_intent.items())
    }
    macro_intent = {
        target: mean(
            float(result[target]["mean"])
            for result in intent_results.values()
        )
        for target in (
            "source_representation_0_100",
            "rendered_ux_0_100",
            "composite_0_100",
        )
    }
    micro_intent = {
        target: mean(float(row[target]) for row in joined)
        for target in macro_intent
    }

    correlations: dict[str, Any] = {}
    for metric_key in ("legacy_score", "v5_4_score"):
        correlations[metric_key] = {}
        for target in (
            "source_representation_0_100",
            "rendered_ux_0_100",
            "composite_0_100",
        ):
            paired = [
                (float(row[metric_key]), float(row[target]))
                for row in joined
                if _finite_float(row.get(metric_key)) is not None
            ]
            predictions = [value for value, _ in paired]
            targets = [value for _, value in paired]
            correlations[metric_key][target] = {
                **_metric_vector_stats(predictions, targets),
                "paired_bootstrap_95": _paired_bootstrap(
                    predictions,
                    targets,
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed,
                ),
            }

    calibration = {
        metric: _fit_calibrators(
            joined,
            metric_key=metric,
            target_key="composite_0_100",
        )
        for metric in ("legacy_score", "v5_4_score")
    }
    repeats = _repeat_statistics(repeat_rows)
    milestone = _milestone_statistics(joined)
    task_provenance = _judge_task_provenance(
        _read_jsonl(root / "raw_judgments.jsonl"),
        schedule_map,
    )
    _atomic_write_json(
        output_dir / "judge_task_provenance.json",
        task_provenance,
    )
    _atomic_write_json(
        output_dir / "correlations.json",
        correlations,
    )
    _atomic_write_json(
        output_dir / "distributions.json",
        {
            "population_stress_combined": distributions,
            "intent_results": intent_results,
            "macro_intent": macro_intent,
            "micro_intent": micro_intent,
        },
    )
    _atomic_write_json(
        output_dir / "repeat_reliability.json",
        repeats,
    )
    _atomic_write_json(
        output_dir / "milestone_drift.json",
        milestone,
    )
    plots = _make_plots(output_dir, joined, repeat_rows, calibration)

    score_band_examples: list[dict[str, Any]] = []
    score_band_asset_dir = output_dir / "score_band_examples"
    score_band_asset_dir.mkdir()
    for lower in range(0, 100, 10):
        candidates = [
            row
            for row in joined
            if lower <= float(row["composite_0_100"]) < lower + 10
            or (
                lower == 90
                and float(row["composite_0_100"]) == 100.0
            )
        ]
        chosen = sorted(
            candidates,
            key=lambda row: (
                abs(float(row["composite_0_100"]) - (lower + 5)),
                str(row["original_packet_id"]),
            ),
        )[:3]
        for row in chosen:
            screenshot_packet_path = (
                root
                / "packets"
                / "screenshot_only"
                / f"{row['original_packet_id']}.json"
            )
            screenshot_packet = json.loads(
                screenshot_packet_path.read_text(encoding="utf-8")
            )
            overview_source = (
                screenshot_packet_path.parent
                / str(screenshot_packet["overview_asset"])
            ).resolve()
            overview_destination = (
                score_band_asset_dir
                / f"{lower:02d}_{row['original_packet_id']}.jpg"
            )
            shutil.copy2(overview_source, overview_destination)
            score_band_examples.append(
                {
                    "band": f"{lower:02d}-{lower + 9:02d}",
                    "packet_id": row["original_packet_id"],
                    "intent_bucket": row["intent_bucket"],
                    "source_representation_0_100": row[
                        "source_representation_0_100"
                    ],
                    "rendered_ux_0_100": row["rendered_ux_0_100"],
                    "composite_0_100": row["composite_0_100"],
                    "screenshot_packet": str(screenshot_packet_path),
                    "source_packet": str(
                        root
                        / "packets"
                        / "source_conditioned"
                        / f"{row['original_packet_id']}.json"
                    ),
                    "overview_asset": str(overview_destination),
                    "overview_sha256": _hash_file(
                        overview_destination
                    ),
                }
            )
    _atomic_write_jsonl(
        output_dir / "score_band_examples.jsonl",
        score_band_examples,
    )

    source_manifest = json.loads(
        (root / "selection_manifest.json").read_text(encoding="utf-8")
    )
    current_source_hash = _hash_file(
        Path(source_manifest["source_genui"])
    )
    native_provenance_path = root / "native_provenance.json"
    native_provenance = (
        json.loads(native_provenance_path.read_text(encoding="utf-8"))
        if native_provenance_path.exists()
        else {}
    )
    captures = _read_jsonl(root / "native_capture_manifest.jsonl")
    capture_summary_path = root / "native_capture_summary.json"
    capture_summary = (
        json.loads(capture_summary_path.read_text(encoding="utf-8"))
        if capture_summary_path.exists()
        else {}
    )
    required_failed = [
        row
        for row in captures
        if bool(row.get("required", True))
        and not bool(row.get("image_captured", row.get("ok")))
    ]
    intent_counts = Counter(row["intent_bucket"] for row in joined)
    acceptance = {
        "960_unique_finalized": (
            len(joined) == 960
            and len({row["ui_id"] for row in joined}) == 960
        ),
        "exactly_30_per_intent": (
            len(intent_counts) == 32
            and set(intent_counts.values()) == {30}
        ),
        "screenshot_completeness": (
            not required_failed
            and int(capture_summary.get("missing_capture_count", -1)) == 0
            and int(
                capture_summary.get("duplicate_capture_key_count", -1)
            )
            == 0
        ),
        "judgment_completeness": len(finalized) == 960,
        "apk_checkout_renderer_parity": bool(
            native_provenance.get("parity_established")
        ),
        "repeat_icc_at_least_0_85": (
            float(repeats.get("icc_a_1_absolute_agreement", 0.0))
            >= 0.85
        ),
        "repeat_mae_at_most_5": (
            float(repeats.get("mae", math.inf)) <= 5.0
        ),
        "repeat_absolute_bias_below_2": (
            abs(
                float(
                    repeats.get(
                        "signed_bias_prediction_minus_target",
                        math.inf,
                    )
                )
            )
            < 2.0
        ),
        "no_unexplained_milestone_drift_above_3": (
            float(milestone["maximum_absolute_adjusted_drift"]) <= 3.0
        ),
        "source_hash_unchanged": (
            current_source_hash == source_manifest["source_genui_sha256"]
        ),
        "all_formulas_finite_bounded": all(
            math.isfinite(float(row[target]))
            and 0.0 <= float(row[target]) <= 100.0
            for row in joined
            for target in (
                "source_representation_0_100",
                "rendered_ux_0_100",
                "composite_0_100",
            )
        ),
        "stable_single_codex_model": bool(
            task_provenance["stable_single_model"]
        ),
        "fresh_codex_task_after_each_80": bool(
            task_provenance["fresh_task_after_each_80"]
        ),
        "same_task_for_both_packet_passes": not bool(
            task_provenance["packet_pass_task_mismatches"]
        ),
        "screenshot_pass_precedes_source_pass": not bool(
            task_provenance["pass_timestamp_order_violations"]
        ),
        "adjudication_uses_fresh_task": bool(
            task_provenance["adjudication_uses_fresh_task"]
        ),
    }
    acceptance["publishable_as_provisional_groundtruth"] = all(
        acceptance.values()
    )
    analysis = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "authority": "provisional_single_codex_groundtruth",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(joined),
        "distributions": distributions,
        "intent_results": intent_results,
        "macro_intent": macro_intent,
        "micro_intent": micro_intent,
        "repeat_reliability": repeats,
        "milestone_drift": milestone,
        "judge_task_provenance": task_provenance,
        "correlations": correlations,
        "calibration": calibration,
        "acceptance": acceptance,
        "plots": plots,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "calibration_status": (
            "provisional single-Codex calibration; not human calibration"
        ),
    }
    _atomic_write_json(output_dir / "analysis.json", analysis)
    _atomic_write_json(
        output_dir / "calibration_models.json",
        {
            "schema_version": "genui_single_codex_calibration.v2",
            "authority": "provisional_single_codex_groundtruth",
            "calibration": calibration,
        },
    )
    report = _report_markdown(analysis)
    (output_dir / "REPORT.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    root_report = root / "REPORT.md"
    if root_report.exists():
        raise FileExistsError(
            f"immutable benchmark report already exists: {root_report}"
        )
    root_report.write_text(report, encoding="utf-8", newline="\n")
    _atomic_write_json(
        root / "judge_aggregates.json",
        {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "authority": "provisional_single_codex_groundtruth",
            "distributions": distributions,
            "macro_intent": macro_intent,
            "micro_intent": micro_intent,
            "repeat_reliability": repeats,
            "milestone_drift": milestone,
            "judge_task_provenance": task_provenance,
            "acceptance": acceptance,
        },
    )
    return analysis


def _report_markdown(analysis: Mapping[str, Any]) -> str:
    combined = analysis["distributions"]["combined"]
    repeats = analysis["repeat_reliability"]
    task_provenance = analysis["judge_task_provenance"]
    acceptance = analysis["acceptance"]
    lines = [
        "# Single-Codex GenUI Judge Benchmark v2",
        "",
        "> Label authority: `provisional_single_codex_groundtruth`. "
        "This is neither human ground truth nor human calibration.",
        "",
        "## Scores",
        "",
        "| Surface | Mean | Median | SD |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("source_representation_0_100", "Source representation R"),
        ("rendered_ux_0_100", "Rendered UX U"),
        ("composite_0_100", "Composite J"),
    ):
        value = combined[key]
        lines.append(
            f"| {label} | {value['mean']:.2f} | "
            f"{value['median']:.2f} | {value['sd_population']:.2f} |"
        )
    lines += [
        "",
        "## Blind-repeat reliability",
        "",
        f"- ICC(A,1): {float(repeats.get('icc_a_1_absolute_agreement', 0.0)):.3f}",
        f"- MAE: {float(repeats.get('mae', 0.0)):.2f}",
        "- Signed bias, repeat minus original: "
        f"{float(repeats.get('signed_bias_prediction_minus_target', 0.0)):.2f}",
        "",
        "## Judge provenance",
        "",
        "- Model identifier: "
        + ", ".join(task_provenance["model_identifiers"]),
        "- Fresh task after each 80 scheduled packets: "
        + (
            "yes"
            if task_provenance["fresh_task_after_each_80"]
            else "no"
        ),
        "",
        "## Metric agreement",
        "",
        "| Metric | Judge target | Pearson | Spearman | Kendall | MAE | RMSE | Bias |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    target_labels = {
        "source_representation_0_100": "R",
        "rendered_ux_0_100": "U",
        "composite_0_100": "J",
    }
    for metric, targets in analysis["correlations"].items():
        for target, values in targets.items():
            lines.append(
                f"| `{metric}` | {target_labels[target]} | "
                f"{float(values['pearson_r']):.3f} | "
                f"{float(values['spearman_rho']):.3f} | "
                f"{float(values['kendall_tau']):.3f} | "
                f"{float(values['mae']):.2f} | "
                f"{float(values['rmse']):.2f} | "
                f"{float(values['signed_bias_prediction_minus_target']):.2f} |"
            )
    lines += [
        "",
        "Paired-bootstrap 95% intervals are stored in "
        "`analysis/correlations.json`.",
        "",
        "## Acceptance",
        "",
    ]
    for key, value in acceptance.items():
        lines.append(f"- [{'x' if value else ' '}] `{key}`")
    lines += [
        "",
        "## Calibration",
        "",
    ]
    for metric, calibration in analysis["calibration"].items():
        holdout = calibration["holdout_selected_model_once"]
        lines.append(
            f"- `{metric}` selected `{calibration['selected_model']}`; "
            f"one-shot holdout MAE {holdout['mae']:.2f}, "
            f"Spearman {holdout['spearman_rho']:.3f}."
        )
    lines += [
        "",
        "Initial mappings are expert-prior metric calibration against one "
        "Codex judge. Metric weights were not optimized from these labels.",
        "",
        "Score-band example overviews and their packet index are stored under "
        "`analysis/score_band_examples/` and "
        "`analysis/score_band_examples.jsonl`.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "analyze_benchmark",
]
