"""Static guardrails for the single active A2UI Express boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_active_python_paths_do_not_import_compact_or_legacy_fallbacks() -> None:
    active_paths = (
        ROOT / "src/pipeline/stage3_genui.py",
        ROOT / "src/pipeline/stage4_render.py",
        ROOT / "src/pipeline/metrics.py",
        ROOT / "src/pipeline/ir_formats/active.py",
        ROOT / "src/pipeline/ir_formats/codec.py",
        ROOT / "src/pipeline/ir_formats/express.py",
        ROOT / "src/pipeline/ir_formats/a2ui_wire.py",
    )
    forbidden_imports = ("migration.compact_ir_v2", "dual_format_pilot", "compact_ir.py")
    for path in active_paths:
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden_imports), path


def test_training_config_and_target_boundary_are_express_only() -> None:
    targets = (ROOT.parent / "training/src/ir_training/data/ir_targets.py").read_text(encoding="utf-8")
    assert "SUPPORTED_TARGETS = (A2UI_EXPRESS_V1,)" in targets
    for path in (ROOT.parent / "training/configs").glob("*_a2ui_express_v1.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "target_format: a2ui_express_v1" in text


def test_android_production_boundary_has_no_compact_decoder_or_legacy_fallback() -> None:
    android = ROOT.parent / "android/app/src/main/java/com/samsung/genuicraft"
    codec = (android / "pipeline/GenUiIrCodec.kt").read_text(encoding="utf-8")
    stage_pipeline = (android / "pipeline/GenUiStagePipeline.kt").read_text(encoding="utf-8")
    settings = (android / "pipeline/IrPromptVersionSettings.kt").read_text(encoding="utf-8")
    assert "CompactIrCodec" not in codec
    assert "FlatSpecIngestor" not in stage_pipeline
    assert settings.count('id = "a2ui_express_v1"') == 1
    assert "A2UI_EXPRESS_V1" in codec and "A2UI_V1_WIRE" in codec


def test_token_estimate_is_explicitly_diagnostic() -> None:
    metrics = (ROOT / "src/pipeline/metrics.py").read_text(encoding="utf-8")
    assert "def lexical_token_estimate" in metrics
    assert "def count_tokens" not in metrics
