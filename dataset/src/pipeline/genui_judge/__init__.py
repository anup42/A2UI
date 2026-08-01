"""Single-Codex visual judging support for GenUI benchmark artifacts."""

from .protocol import (
    COMPOSITE_WEIGHTS,
    DIMENSIONS,
    JUDGE_AUTHORITY,
    JUDGE_PROTOCOL_VERSION,
    JUDGE_SCHEMA_VERSION,
    JudgedScores,
    compute_judged_scores,
    protocol_fingerprint,
    validate_judgment_pass,
)
from .selection import (
    DEFAULT_SELECTION_SEED,
    build_benchmark_selection,
    write_selection_coverage_audit,
)
from .judgments import (
    append_judgment_pass,
    build_repeat_analysis,
    finalize_groundtruth,
)
from .packets import build_adjudication_packets, build_blinded_packets
from .metric_audit import (
    precompute_selection_v5_4,
    rescore_selected_v5_4,
)
from .analysis import analyze_benchmark
from .blind_workspace import (
    blind_workspace_status,
    import_blind_adjudication_batch,
    import_blind_batch,
    initialize_blind_workspace,
    prepare_blind_adjudication_batch,
    prepare_blind_batch,
    sync_frozen_protocol_to_blind_workspace,
)
from .provenance import (
    implementation_fingerprint,
    seal_implementation_provenance,
    verify_implementation_provenance,
)

__all__ = [
    "COMPOSITE_WEIGHTS",
    "DEFAULT_SELECTION_SEED",
    "DIMENSIONS",
    "JUDGE_AUTHORITY",
    "JUDGE_PROTOCOL_VERSION",
    "JUDGE_SCHEMA_VERSION",
    "JudgedScores",
    "build_benchmark_selection",
    "write_selection_coverage_audit",
    "build_adjudication_packets",
    "build_blinded_packets",
    "append_judgment_pass",
    "analyze_benchmark",
    "blind_workspace_status",
    "build_repeat_analysis",
    "compute_judged_scores",
    "finalize_groundtruth",
    "import_blind_batch",
    "import_blind_adjudication_batch",
    "implementation_fingerprint",
    "initialize_blind_workspace",
    "protocol_fingerprint",
    "precompute_selection_v5_4",
    "prepare_blind_adjudication_batch",
    "prepare_blind_batch",
    "rescore_selected_v5_4",
    "seal_implementation_provenance",
    "sync_frozen_protocol_to_blind_workspace",
    "verify_implementation_provenance",
    "validate_judgment_pass",
]
# Versioned criterion-referenced judge.  V2 exports above remain unchanged.
from .criterion_protocol_v3 import (
    JUDGE_NAME as CRITERION_JUDGE_NAME,
    JUDGE_PROTOCOL_VERSION as CRITERION_JUDGE_PROTOCOL_VERSION,
    compute_criterion_judged_scores,
    compute_dimension_score as compute_criterion_dimension_score,
    protocol_fingerprint as criterion_protocol_fingerprint,
    validate_criterion_judgment_pass,
)
from .criterion_judgments_v3 import (
    append_criterion_judgment_pass,
    combine_criterion_passes,
    rebuild_criterion_packet_judgments,
)
from .criterion_packets_v3 import build_criterion_review_packets
from .criterion_analysis_v3 import (
    analyze_fresh_criterion_repeats,
    backtest_v2_repeat_bundle,
)

__all__ += [
    "CRITERION_JUDGE_NAME",
    "CRITERION_JUDGE_PROTOCOL_VERSION",
    "analyze_fresh_criterion_repeats",
    "append_criterion_judgment_pass",
    "backtest_v2_repeat_bundle",
    "build_criterion_review_packets",
    "combine_criterion_passes",
    "compute_criterion_dimension_score",
    "compute_criterion_judged_scores",
    "criterion_protocol_fingerprint",
    "rebuild_criterion_packet_judgments",
    "validate_criterion_judgment_pass",
]
