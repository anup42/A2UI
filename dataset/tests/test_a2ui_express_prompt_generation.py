from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_a2ui_express_prompt.py"
PROMPT_PATHS = [
    ROOT / "prompts" / "genui_gen_mobile_a2ui_express_v1.md",
    ROOT / ".." / "android" / "app" / "src" / "main" / "assets" / "pipeline_prompts" / "genui_gen.md",
    ROOT / ".." / "android" / "app" / "src" / "main" / "assets" / "pipeline_prompts" / "genui_gen_a2ui_express_v1.md",
]


def test_generated_prompt_copies_are_drift_free() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    contents = [path.resolve().read_bytes() for path in PROMPT_PATHS]
    assert len({content for content in contents}) == 1


def test_generated_prompt_is_catalog_and_quality_complete() -> None:
    text = PROMPT_PATHS[0].read_text(encoding="utf-8")
    folded = text.lower()
    for token in (
        "grammar=",
        "catalog=",
        "profile=",
        "<a2ui>",
        "Table(columns, statePath, rows, title, domain, preferredPresentation)",
        "emitEvent(name, context, wantResponse, responsePath)",
        "preserve every requested fact",
        "_props",
    ):
        assert token.lower() in folded
