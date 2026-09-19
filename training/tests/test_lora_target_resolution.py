from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training" / "src"))
from ir_training.common.config import load_yaml
from ir_training.qat.mobile_qparams import MobileQParams
from ir_training.qat.mobile_training_seed import OFFICIAL_MOBILE_MODEL_ID
from ir_training.train.lora_config import (
    load_retained_mobile_lora_qparams,
    resolve_lora_config_targets,
)
from ir_training.train.lora_targets import bind_retained_mobile_peft_targets


def _review_selector():
    path = Path(__file__).resolve().parents[1] / "configs/models/gemma4_e2b_a2ui_express_review_sft.yaml"
    return load_yaml(path)["lora"]["target_modules"]


def _model():
    nn = pytest.importorskip("torch.nn")

    class WrappedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.q_proj = WrappedLinear()
            self.language_model.v_proj = nn.Linear(2, 2, bias=False)
            self.language_model.o_proj = nn.Linear(2, 2, bias=False)
            self.language_model.norm = nn.LayerNorm(2)
            self.vision_model = nn.Module()
            self.vision_model.q_proj = nn.Linear(2, 2, bias=False)
            self.embeddings = nn.Embedding(4, 2)
            self.lm_head = nn.Linear(2, 4, bias=False)
            self.config = SimpleNamespace(model_type="mock_model")

        def get_output_embeddings(self):
            return self.lm_head

    return Model()


def test_language_qv_regex_resolves_actual_inner_linear_without_modality_widening():
    config = SimpleNamespace(target_modules=r"language_model\.(q_proj|v_proj)")
    names = resolve_lora_config_targets(config, _model())
    assert names == {"language_model.q_proj.linear", "language_model.v_proj"}
    assert config.target_modules == names


def test_suffix_list_resolves_wrapper_and_direct_linear():
    config = SimpleNamespace(target_modules=["q_proj", "v_proj"])
    assert resolve_lora_config_targets(config, _model()) == {
        "language_model.q_proj.linear", "language_model.v_proj", "vision_model.q_proj"}


def test_all_linear_excludes_output_head_and_embedding_and_deduplicates_inner():
    config = SimpleNamespace(target_modules="all-linear")
    names = resolve_lora_config_targets(config, _model())
    assert names == {"language_model.q_proj.linear", "language_model.v_proj",
                     "language_model.o_proj", "vision_model.q_proj"}
    assert "lm_head" not in names
    assert "embeddings" not in names


def test_exact_materialized_names_resolve_idempotently_for_resume():
    config = SimpleNamespace(target_modules=r"language_model\.(q_proj|v_proj)(\.linear)?")
    model = _model()
    first = resolve_lora_config_targets(config, model)
    assert resolve_lora_config_targets(config, model) == first
    assert first == {"language_model.q_proj.linear", "language_model.v_proj"}


def test_default_mapping_is_materialized_before_adapter_construction(monkeypatch):
    constants = ModuleType("peft.utils.constants")
    constants.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {"mock_model": ["q_proj", "v_proj"]}
    monkeypatch.setitem(sys.modules, "peft.utils.constants", constants)
    config = SimpleNamespace(target_modules=None)
    assert resolve_lora_config_targets(config, _model()) == {
        "language_model.q_proj.linear", "language_model.v_proj", "vision_model.q_proj"}


def test_zero_unsupported_and_invalid_regex_fail_without_model_forward():
    for selector in (["missing"], ["embeddings"], ["norm"]):
        with pytest.raises(ValueError, match="no supported"):
            resolve_lora_config_targets(SimpleNamespace(target_modules=selector), _model())
    with pytest.raises(Exception, match="unterminated"):
        resolve_lora_config_targets(SimpleNamespace(target_modules="["), _model())


def test_explicit_exclusions_apply_at_wrapper_and_inner_names():
    config = SimpleNamespace(target_modules="all-linear", exclude_modules=["q_proj"])
    names = resolve_lora_config_targets(config, _model())
    assert "language_model.q_proj.linear" not in names
    assert "vision_model.q_proj" not in names


def test_all_linear_excludes_output_embedding_with_nonstandard_name():
    model = _model()
    output = model.lm_head
    del model.lm_head
    model.custom_output = output
    model.get_output_embeddings = lambda: output
    names = resolve_lora_config_targets(SimpleNamespace(target_modules="all-linear"), model)
    assert "custom_output" not in names


def test_explicit_output_head_selection_is_not_silently_overridden():
    config = SimpleNamespace(target_modules=["lm_head"])
    assert resolve_lora_config_targets(config, _model()) == {"lm_head"}


def test_unknown_peft_default_fails_with_actionable_message(monkeypatch):
    constants = ModuleType("peft.utils.constants")
    constants.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {}
    monkeypatch.setitem(sys.modules, "peft.utils.constants", constants)
    with pytest.raises(ValueError, match="choose explicit language targets"):
        resolve_lora_config_targets(SimpleNamespace(target_modules=None), _model())


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_e2b_review_selector_supports_both_language_roots_without_modality_widening(multimodal, wrapped):
    nn = pytest.importorskip("torch.nn")
    model = nn.Module()
    model.model = nn.Module()
    decoder = nn.Module()
    decoder.layers = nn.ModuleList([nn.Module()])
    layer = decoder.layers[0]
    expected = set()
    prefix = "model.language_model.layers.0" if multimodal else "model.layers.0"
    for block, projections in (("self_attn", ("q", "k", "v", "o")), ("mlp", ("gate", "up", "down"))):
        branch = nn.Module()
        setattr(layer, block, branch)
        for projection in projections:
            module = nn.Linear(2, 2, bias=False)
            if wrapped:
                wrapper = nn.Module()
                wrapper.linear = module
                module = wrapper
            setattr(branch, f"{projection}_proj", module)
            expected.add(f"{prefix}.{block}.{projection}_proj" + (".linear" if wrapped else ""))
    if multimodal:
        model.model.language_model = decoder
    else:
        model.model.layers = decoder.layers
    # Deliberately give modality towers the SAME nested language-style paths.
    # A leading .* would accidentally select these.
    import copy
    for tower in ("vision_tower", "audio_tower"):
        branch = nn.Module()
        branch.language_model = copy.deepcopy(decoder)
        branch.layers = copy.deepcopy(decoder.layers)
        setattr(model.model, tower, branch)
    model.lm_head = nn.Linear(2, 4, bias=False)
    config = SimpleNamespace(target_modules=_review_selector())
    assert resolve_lora_config_targets(config, model) == expected
    assert resolve_lora_config_targets(config, model) == expected


def test_no_match_reports_bounded_real_inventory_and_does_not_mutate_selector():
    model = _model()
    config = SimpleNamespace(target_modules="missing")
    with pytest.raises(ValueError) as error:
        resolve_lora_config_targets(config, model)
    message = str(error.value)
    assert "model_class=Model" in message
    assert "model_type='mock_model'" in message
    assert "linear_module_count=5" in message
    assert "language_model.q_proj.linear" in message
    assert "No fallback" in message
    assert config.target_modules == "missing"
    for index in range(20):
        model.add_module(f"extra_{index:02}", pytest.importorskip("torch.nn").Linear(2, 2))
    with pytest.raises(ValueError) as error:
        resolve_lora_config_targets(config, model)
    assert "linear_module_count=25" in str(error.value)
    assert "extra_11" in str(error.value)
    assert "extra_12" not in str(error.value)


def _retained_mobile_fixture(*, wrapped=False):
    """Tiny modules, real qparams key selection; no model files or inference."""
    nn = pytest.importorskip("torch.nn")
    model = nn.Module()
    model.config = SimpleNamespace(model_type="gemma4_text", to_dict=lambda: {"model_type": "gemma4_text"})
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module() for _ in range(35)])
    model.lm_head = nn.Linear(2, 4, bias=False)
    model.model.extra_linear = nn.Linear(2, 2, bias=False)
    model.model.vision_tower = nn.Module()
    model.model.vision_tower.q_proj = nn.Linear(2, 2, bias=False)
    model.get_output_embeddings = lambda: model.lm_head
    inventory, expected = {}, set()
    for index, layer in enumerate(model.model.layers):
        layer.self_attn, layer.mlp = nn.Module(), nn.Module()
        families = {"self_attn": ("q", "k", "v", "o") if index < 15 else ("q", "o"),
                    "mlp": ("gate", "up", "down")}
        for block, projections in families.items():
            for projection in projections:
                name = f"model.layers.{index}.{block}.{projection}_proj"
                linear = nn.Linear(2, 2, bias=False)
                if wrapped:
                    wrapper = nn.Module()
                    wrapper.linear = linear
                    linear = wrapper
                setattr(getattr(layer, block), f"{projection}_proj", linear)
                inventory[name + ".weight"] = {"bits": 4,
                    "input_activation_scale_f32_le_hex": "0000803f",
                    "output_activation_scale_f32_le_hex": "0000803f"}
                expected.add(name + (".linear" if wrapped else ""))
        for frozen in ("per_layer_input_gate", "per_layer_projection"):
            setattr(layer, frozen, nn.Linear(2, 2, bias=False))
            inventory[f"model.layers.{index}.{frozen}.weight"] = {"bits": 8,
                "input_activation_scale_f32_le_hex": "0000803f",
                "output_activation_scale_f32_le_hex": "0000803f"}
    inventory["lm_head.weight"] = {"bits": 8,
        "input_activation_scale_f32_le_hex": "00000000",
        "output_activation_scale_f32_le_hex": "00000000"}
    # Skip on-disk provenance setup only; exercise the actual canonical key
    # method shared with live QAT. Loader provenance is tested separately below.
    qparams = MobileQParams.__new__(MobileQParams)
    qparams.inventory = inventory
    qparams.path = Path("synthetic-mobile-qparams.json")
    qparams.storage_path = Path("synthetic-mobile-qparams.safetensors")
    qparams.report = {"verified": True, "contract_sha256": "a" * 64,
                      "scale_storage_sha256": "b" * 64, "inventory_sha256": "c" * 64}
    return model, qparams, expected


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("text_mapping_present", [False, True])
def test_retained_mobile_default_uses_exact_contract_not_peft_qv(monkeypatch, wrapped, text_mapping_present):
    constants = ModuleType("peft.utils.constants")
    mapping = {"gemma4": r".*language_model\..*\.(q_proj|v_proj)"}
    if text_mapping_present:
        mapping["gemma4_text"] = ["q_proj", "v_proj"]
    constants.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = mapping
    monkeypatch.setitem(sys.modules, "peft.utils.constants", constants)
    model, qparams, expected = _retained_mobile_fixture(wrapped=wrapped)
    config = SimpleNamespace(target_modules=None)  # build_lora_config's peft-default
    resolved = resolve_lora_config_targets(config, model, retained_mobile_qparams=qparams)
    assert resolved == expected and len(resolved) == 205
    assert resolve_lora_config_targets(config, model, retained_mobile_qparams=qparams) == resolved
    for index in range(35):
        prefix = f"model.layers.{index}."
        assert len([name for name in resolved if name.startswith(prefix)]) == (7 if index < 15 else 5)
        for projection in ("k", "v"):
            name = f"{prefix}self_attn.{projection}_proj" + (".linear" if wrapped else "")
            assert (name in resolved) == (index < 15)
    assert all("lm_head" not in name and "per_layer" not in name and "vision" not in name for name in resolved)


@pytest.mark.parametrize("change", ["missing", "extra_shared_kv", "duplicate_module", "tied_weight", "nonlinear", "ambiguous"])
def test_retained_mobile_malformed_model_fails_closed(change):
    nn = pytest.importorskip("torch.nn")
    model, qparams, _ = _retained_mobile_fixture()
    attention = model.model.layers[0].self_attn
    if change == "missing":
        del attention.q_proj
    elif change == "extra_shared_kv":
        model.model.layers[15].self_attn.k_proj = nn.Linear(2, 2, bias=False)
    elif change == "duplicate_module":
        attention.k_proj = attention.q_proj
    elif change == "tied_weight":
        attention.k_proj.weight = attention.q_proj.weight
    elif change == "nonlinear":
        attention.q_proj = nn.Identity()
    else:
        attention.q_proj.linear = nn.Linear(2, 2, bias=False)
    config = SimpleNamespace(target_modules=None)
    with pytest.raises(ValueError, match="[Rr]etained mobile"):
        resolve_lora_config_targets(config, model, retained_mobile_qparams=qparams)
    assert config.target_modules is None


@pytest.mark.parametrize("change", ["unverified", "missing_key", "duplicate_key", "wrong_model_type"])
def test_retained_mobile_contract_must_be_verified_complete_and_text_only(change):
    model, qparams, _ = _retained_mobile_fixture()
    if change == "unverified":
        qparams.report["verified"] = False
    elif change == "missing_key":
        del qparams.inventory["model.layers.0.self_attn.q_proj.weight"]
    elif change == "duplicate_key":
        keys = qparams.trainable_projection_weight_keys()
        qparams.trainable_projection_weight_keys = lambda: (keys[1], *keys[1:])
    else:
        model.config.model_type = "gemma4"
    with pytest.raises(ValueError, match="Retained mobile"):
        resolve_lora_config_targets(SimpleNamespace(target_modules=None), model, retained_mobile_qparams=qparams)


@pytest.mark.parametrize("selector", [["q_proj", "v_proj"], "all-linear", ["lm_head"]])
def test_retained_mobile_explicit_selector_cannot_change_contract(selector):
    model, qparams, _ = _retained_mobile_fixture()
    config = SimpleNamespace(target_modules=selector)
    with pytest.raises(ValueError, match="exact retained mobile projection contract"):
        resolve_lora_config_targets(config, model, retained_mobile_qparams=qparams)
    assert config.target_modules == selector


def test_retained_mobile_exclusions_cannot_drop_a_projection():
    model, qparams, _ = _retained_mobile_fixture()
    with pytest.raises(ValueError, match="exact retained mobile projection contract"):
        resolve_lora_config_targets(SimpleNamespace(target_modules=None, exclude_modules=["q_proj"]),
                                    model, retained_mobile_qparams=qparams)


def test_ordinary_gemma4_peft_default_remains_qv_and_text_requires_opt_in(monkeypatch):
    constants = ModuleType("peft.utils.constants")
    constants.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {"gemma4": r"language_model\.(q_proj|v_proj)"}
    monkeypatch.setitem(sys.modules, "peft.utils.constants", constants)
    model = _model()
    model.config.model_type = "gemma4"
    assert resolve_lora_config_targets(SimpleNamespace(target_modules=None), model) == {
        "language_model.q_proj.linear", "language_model.v_proj"}
    model.config.model_type = "gemma4_text"
    with pytest.raises(ValueError, match="Cannot resolve PEFT default"):
        resolve_lora_config_targets(SimpleNamespace(target_modules=None), model)


@pytest.mark.parametrize("change", [None, "seed_unverified", "missing_contract", "contract_sha256", "scale_storage_sha256", "inventory_sha256"])
def test_retained_mobile_loader_binds_to_verified_seed(monkeypatch, change):
    import copy

    from ir_training.qat import mobile_qparams

    _, qparams, _ = _retained_mobile_fixture()
    model_cfg = {"model_id": OFFICIAL_MOBILE_MODEL_ID, "mobile_qparams_contract": "seed/mobile_qparams.json"}
    qat_cfg = {"enabled": True, "scale_mode": "retained_mobile"}
    seed = {"required": True, "verified": True, "mobile_qparams": copy.deepcopy(qparams.report)}
    if change == "seed_unverified":
        seed["verified"] = False
    elif change == "missing_contract":
        del model_cfg["mobile_qparams_contract"]
    elif change:
        seed["mobile_qparams"][change] = "d" * 64
    monkeypatch.setattr(mobile_qparams, "MobileQParams", lambda path: qparams)
    if change:
        with pytest.raises(ValueError, match="[Mm]obile|qparams"):
            load_retained_mobile_lora_qparams(model_cfg, qat_cfg, verified_seed=seed)
    else:
        assert load_retained_mobile_lora_qparams(model_cfg, qat_cfg, verified_seed=seed) is qparams


@pytest.mark.parametrize("model_id,enabled,scale_mode", [
    ("google/gemma-4-E2B-it", True, "retained_mobile"),
    (OFFICIAL_MOBILE_MODEL_ID, False, "retained_mobile"),
    (OFFICIAL_MOBILE_MODEL_ID, True, "dynamic"),
])
def test_ordinary_training_does_not_load_retained_target_contract(monkeypatch, model_id, enabled, scale_mode):
    from ir_training.qat import mobile_qparams

    def unexpected(*_args, **_kwargs):
        pytest.fail("Ordinary LoRA must not load a retained-mobile contract")

    monkeypatch.setattr(mobile_qparams, "MobileQParams", unexpected)
    assert load_retained_mobile_lora_qparams({"model_id": model_id},
        {"enabled": enabled, "scale_mode": scale_mode}, verified_seed={}) is None


def test_sft_resolves_retained_targets_before_fresh_and_resumed_peft():
    import inspect

    from ir_training.train.sft import train_sft

    source = inspect.getsource(train_sft)
    binding = source.index("load_retained_mobile_lora_qparams(model_cfg, qat_cfg, verified_seed=mobile_training_seed)")
    resolution = source.index("retained_mobile_qparams=retained_lora_qparams")
    assert binding < resolution < source.index("PeftModel.from_pretrained(")
    assert resolution < source.index("get_peft_model(model, configured_lora)")


def test_tiny_real_gemma4_text_peft_and_live_qat_share_exact_205_scope(monkeypatch, tmp_path):
    """Real HF/PEFT attachment and QAT binding only, no forward or optimizer."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers", minversion="5.10.1")
    peft = pytest.importorskip("peft", minversion="0.20.0")
    from ir_training.qat import mobile_qparams
    from ir_training.qat.fake_quant import prepare_qat_model
    from ir_training.train.sft import _validate_resumed_lora_model

    _, qparams, expected = _retained_mobile_fixture()
    config = transformers.Gemma4TextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=35, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, global_head_dim=16, max_position_embeddings=64,
        vocab_size_per_layer_input=32, hidden_size_per_layer_input=4,
        layer_types=["full_attention"] * 35, num_kv_shared_layers=20,
    )
    model = transformers.Gemma4ForCausalLM(config)
    lora = peft.LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0, task_type="CAUSAL_LM")
    assert model.config.model_type == "gemma4_text" and lora.target_modules is None
    assert resolve_lora_config_targets(lora, model, retained_mobile_qparams=qparams) == expected
    adapted = peft.get_peft_model(model, lora)
    realized = {name.removeprefix("base_model.model.")
                for name, module in adapted.named_modules() if hasattr(module, "lora_A")}
    assert realized == expected and len(realized) == 205
    bind_retained_mobile_peft_targets(adapted, expected)
    assert set(adapted.peft_config["default"].target_modules) == expected
    assert all("lora_" in name for name, parameter in adapted.named_parameters() if parameter.requires_grad)
    # Scalar values/storage are synthetic; actual canonical key selection and
    # real live effective-LoRA coverage validation remain active.
    monkeypatch.setattr(qparams, "load_scale", lambda key, *, weight_shape, **_kwargs:
                        torch.ones((weight_shape[0], 1), dtype=torch.float32))
    monkeypatch.setattr(mobile_qparams, "MobileQParams", lambda _path: qparams)
    controller = prepare_qat_model(adapted, {"qat": {
        "enabled": True, "scale_mode": "retained_mobile", "mobile_qparams_contract": "synthetic",
        "expected_effective_lora_modules": 205, "effective_merged_weight": True,
        "effective_lora_only": True, "require_lora_trainable_scope": True,
        "fixed_scale_required": True, "fixed_activation_scale_required": True,
        "weight_bits": 4, "activation_bits": 8, "quantizer": "ste_ai_edge",
        "weight_per_channel": True, "weight_axis": 0, "group_size": None,
        "exclude_modules": [], "quantize_embeddings": False,
    }})
    try:
        assert controller.wrapped_effective_lora_count == 205
        bindings = controller.summary()["retained_qparams_bindings"]
        assert len(bindings) == 205
        assert {binding["weight_key"] for binding in bindings.values()} == set(qparams.trainable_projection_weight_keys())
    finally:
        controller.restore()
    # No optimizer step: use nonzero synthetic adapter values solely to exercise
    # the existing strict resume validation and checkpoint serialization.
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.01)
    adapted.save_pretrained(tmp_path, safe_serialization=True, save_embedding_layers=False)
    reloaded = peft.PeftModel.from_pretrained(transformers.Gemma4ForCausalLM(config), tmp_path, is_trainable=True)
    bind_retained_mobile_peft_targets(reloaded, expected)
    report = _validate_resumed_lora_model(reloaded, expected_config=lora, checkpoint=tmp_path)
    assert report["adapter_pair_count"] == 205
    assert set(report["target_modules"]) == expected
    reloaded.peft_config["default"].lora_alpha += 1
    with pytest.raises(RuntimeError, match="lora_alpha"):
        _validate_resumed_lora_model(reloaded, expected_config=lora, checkpoint=tmp_path)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_post_peft_binding_rejects_different_scope_without_changing_config(change):
    nn = pytest.importorskip("torch.nn")
    _, _, expected = _retained_mobile_fixture()
    config = SimpleNamespace(target_modules={"q_proj", "v_proj"})
    paths = sorted(expected)
    if change == "missing":
        paths.pop()
    elif change == "extra":
        paths.append("lm_head")
    else:
        paths.append(paths[0])
    modules = [("base_model.model." + name, SimpleNamespace(lora_A={"default": nn.Identity()},
                                                          lora_B={"default": nn.Identity()})) for name in paths]
    model = SimpleNamespace(peft_config={"default": config}, named_modules=lambda **_kwargs: iter(modules))
    with pytest.raises(ValueError, match="Attached PEFT scope differs"):
        bind_retained_mobile_peft_targets(model, expected)
    assert config.target_modules == {"q_proj", "v_proj"}
