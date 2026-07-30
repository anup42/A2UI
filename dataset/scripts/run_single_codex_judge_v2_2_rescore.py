"""Host CLI for the immutable Single-Codex judge v2.2 reliability rescore."""

from __future__ import annotations

import run_single_codex_judge_v2_1_rescore as _cli

from pipeline.genui_judge.reliability_rescore_v2_2 import (
    analyze_rescore_repeats,
    finalize_rescore_overlay,
    import_rescore_batch,
    initialize_reliability_rescore,
    prepare_rescore_batch,
    rescore_status,
)


# Reuse the thoroughly exercised host state machine while replacing every
# profile-sensitive operation with the v2.2 sidecar implementation.
_cli.__doc__ = __doc__
_cli.analyze_rescore_repeats = analyze_rescore_repeats
_cli.finalize_rescore_overlay = finalize_rescore_overlay
_cli.import_rescore_batch = import_rescore_batch
_cli.initialize_reliability_rescore = initialize_reliability_rescore
_cli.prepare_rescore_batch = prepare_rescore_batch
_cli.rescore_status = rescore_status


def main() -> int:
    return _cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
