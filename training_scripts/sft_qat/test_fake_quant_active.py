"""
Behavioral test: does get_peft_model(..., qat_scheme=...) actually inject
fake-quantization noise into the forward pass, or is it a silent no-op?

Static checkpoint inspection cannot answer this (see convert_qat_to_quantized.py
discussion) -- fake quant recomputes its quantize/dequantize round-trip fresh
from the current bf16 weight on every forward pass, so it leaves nothing extra
in the saved state regardless of whether it's active.

This test sidesteps needing to know Unsloth's internal fake-quant naming
entirely. Trick: a freshly created get_peft_model() has lora_B initialized to
all zeros (standard PEFT init), so the LoRA delta (lora_B @ lora_A) is exactly
zero at this point -- the model's forward pass is IDENTICAL to the plain base
model's forward pass, UNLESS qat_scheme itself alters the base layer's
computation (i.e. fake-quant noise). So:

  - logits(qat_scheme model) == logits(plain base model)  -> qat_scheme did
    NOT change anything: fake quantization is NOT actually active.
  - logits differ                                          -> fake-quant noise
    IS being injected into the forward pass: qat_scheme is doing real work.

Usage:
    python test_fake_quant_active.py \
        --model_name /path/to/base/model \
        --lora_r 16 --lora_alpha 32 --qat_scheme int8-int4
    (use the exact lora_r/lora_alpha/qat_scheme the run in question was
    trained with -- check train_config.json in its output_dir if unsure)
"""

import os
import sys

if "--gpu_ids" in sys.argv:
    gpu_idx = sys.argv.index("--gpu_ids")
    if gpu_idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[gpu_idx + 1]
os.environ["FORCE_CUDA"] = "1"

# unsloth must be imported before transformers/peft/trl (including via
# qat_utils, which imports trl/transformers) -- otherwise its patches don't
# apply to those modules and it warns + may silently skip optimizations.
import unsloth  # noqa: F401
import argparse
import torch
from transformers import set_seed
from unsloth import FastLanguageModel, FastModel

from qat_utils import QAT_TARGET_MODULES, resolve_target_module_suffixes


def parse_args():
    p = argparse.ArgumentParser(description="Behavioral fake-quant activity test")
    p.add_argument("--model_name", type=str, required=True,
                   help="Base model path (same as training's --model_name)")
    p.add_argument("--gpu_ids", type=str, default="0")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--qat_scheme", type=str, default="int8-int4",
                   choices=["int4", "int8-int4", "fp8-int4", "fp8-fp8"])
    p.add_argument("--max_length", type=int, default=6144)
    p.add_argument("--test_prompt", type=str, default="The quick brown fox")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_legacy_model", action="store_true",
                   help="Force UNSLOTH_USE_NEW_MODEL=0 before get_peft_model(), "
                        "to test whether the legacy code path's qat_scheme "
                        "handling actually works (the new-model default path "
                        "appears to skip it entirely)")
    p.add_argument("--manual_qat", action="store_true",
                   help="Build LoRA without qat_scheme, then call Unsloth's own "
                        "FastLlamaModel._prepare_for_qat() directly -- bypasses "
                        "the new-model dispatch that silently drops qat_scheme, "
                        "without hitting the legacy path's patch_peft_model crash")
    p.add_argument("--use_fastmodel", action="store_true",
                   help="RECOMMENDED/STANDARD: use FastModel instead of "
                        "FastLanguageModel. FastModel routes to "
                        "FastBaseModel.get_peft_model (vision.py), which natively "
                        "accepts qat_scheme and applies it in Unsloth's intended "
                        "order (_get_peft_model -> _prepare_model_for_qat -> "
                        "fix_lora_auto_mapping -> post_patch_model). "
                        "FastLanguageModel.get_peft_model drops qat_scheme when "
                        "forwarding on the new-model path.")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    set_seed(args.seed)

    print("=" * 70)
    print("Fake-quantization behavioral test")
    print("=" * 70)
    print(f"Model:      {args.model_name}")
    print(f"qat_scheme: {args.qat_scheme}")
    print(f"lora_r:     {args.lora_r}, lora_alpha: {args.lora_alpha}")

    # ---- 1. Load base model, compute reference logits ----
    Loader = FastModel if args.use_fastmodel else FastLanguageModel
    print(f"\n[1/3] Loading base model via "
          f"{'FastModel' if args.use_fastmodel else 'FastLanguageModel'} "
          f"and computing reference logits...")
    model, tokenizer = Loader.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_length,
        load_in_4bit=False, load_in_8bit=False, load_in_16bit=True,
    )
    model.eval()

    try:
        encoded = tokenizer(args.test_prompt, return_tensors="pt").to(model.device)
    except TypeError as te:
        if "NoneType" in str(te) or "subscriptable" in str(te):
            # Gemma-4's multi-modal processor needs text= as a keyword
            encoded = tokenizer(text=args.test_prompt, return_tensors="pt").to(model.device)
        else:
            raise

    with torch.no_grad():
        base_logits = model(**encoded).logits.float().clone()
    print(f"  ✓ Base model logits computed, shape={tuple(base_logits.shape)}")
    print(f"    sample values (last token, first 5 logits): "
          f"{base_logits[0, -1, :5].tolist()}")

    # ---- 2. Wrap with get_peft_model + qat_scheme ----
    if args.force_legacy_model:
        os.environ["UNSLOTH_USE_NEW_MODEL"] = "0"
        print(f"\n  [override] Forcing UNSLOTH_USE_NEW_MODEL=0 (legacy path) "
              f"before get_peft_model()")
    print(f"\n[2/3] Applying get_peft_model(qat_scheme='{args.qat_scheme}')...")
    print(f"  [env check] UNSLOTH_USE_NEW_MODEL="
          f"{os.environ.get('UNSLOTH_USE_NEW_MODEL', '<unset>')} "
          f"right before get_peft_model() -- if '1', a different code path is "
          f"taken that may not reach the qat_scheme block at all")
    print(f"  [module check] {Loader.__name__}.get_peft_model resolves to: "
          f"{Loader.get_peft_model.__module__}."
          f"{Loader.get_peft_model.__qualname__}")
    target_modules = resolve_target_module_suffixes(model, QAT_TARGET_MODULES)

    if args.use_fastmodel:
        # STANDARD PATH: FastModel routes to FastBaseModel.get_peft_model
        # (vision.py), which natively accepts qat_scheme and applies it in
        # Unsloth's own intended order. No custom sequencing on our side.
        print(f"  [use_fastmodel] calling FastModel.get_peft_model with "
              f"qat_scheme='{args.qat_scheme}' (Unsloth's standard supported path)")
        qat_model = FastModel.get_peft_model(
            model, r=args.lora_r, target_modules=target_modules,
            lora_alpha=args.lora_alpha, qat_scheme=args.qat_scheme,
        )
    elif args.manual_qat:
        # Unsloth's new-model dispatch path silently drops qat_scheme (proven by
        # behavioral test), while the legacy path that honors it crashes in
        # patch_peft_model ("gemma3_text/gemma4 is not yet implemented"). So:
        # build LoRA WITHOUT qat_scheme (staying on the working path), then call
        # Unsloth's own QAT prep function directly. Same underlying
        # implementation, just invoked manually instead of via broken plumbing.
        print(f"  [manual_qat] calling get_peft_model WITHOUT qat_scheme, then "
              f"FastLlamaModel._prepare_for_qat(model, '{args.qat_scheme}') directly")
        qat_model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r, target_modules=target_modules,
            lora_alpha=args.lora_alpha,
        )
        from unsloth.models.llama import FastLlamaModel
        qat_model = FastLlamaModel._prepare_for_qat(qat_model, args.qat_scheme)
        print(f"  [manual_qat] _prepare_for_qat returned "
              f"type={type(qat_model).__name__}")
    else:
        qat_model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r, target_modules=target_modules,
            lora_alpha=args.lora_alpha, qat_scheme=args.qat_scheme,
        )

    print(f"  [env check after] UNSLOTH_USE_NEW_MODEL="
          f"{os.environ.get('UNSLOTH_USE_NEW_MODEL', '<unset>')} "
          f"after get_peft_model() returned")
    qat_model.eval()

    # Sanity check: confirm the "LoRA contributes nothing yet" assumption holds.
    lora_b_max = max(
        (p.abs().max().item() for n, p in qat_model.named_parameters()
         if "lora_b" in n.lower()),
        default=None,
    )
    print(f"  [sanity] max |lora_B| across all layers: {lora_b_max:.2e} "
          f"(should be ~0.0 -- confirms LoRA contributes nothing yet, so any "
          f"logit difference below can ONLY come from qat_scheme itself)")

    with torch.no_grad():
        qat_logits = qat_model(**encoded).logits.float().clone()
    print(f"  ✓ QAT-wrapped model logits computed")
    print(f"    sample values (last token, first 5 logits): "
          f"{qat_logits[0, -1, :5].tolist()}")

    # ---- 3. Compare ----
    print(f"\n[3/3] Comparing base vs qat_scheme-wrapped logits...")
    diff = (base_logits - qat_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    identical = torch.allclose(base_logits, qat_logits, atol=1e-4, rtol=1e-3)

    print(f"  max |diff|  = {max_diff:.6e}")
    print(f"  mean |diff| = {mean_diff:.6e}")
    print(f"  torch.allclose(atol=1e-4, rtol=1e-3): {identical}")

    print("\n" + "=" * 70)
    if identical:
        print("RESULT: IDENTICAL logits.")
        print("  qat_scheme is NOT injecting any fake-quant noise into the")
        print("  forward pass for this model/scheme combination. Fake")
        print("  quantization was NOT actually active -- training ran in")
        print("  plain bf16 the whole time.")
    else:
        print("RESULT: DIFFERENT logits.")
        print("  qat_scheme IS altering the forward pass -- fake-quant noise")
        print("  is genuinely being injected. Fake quantization was active.")
    print("=" * 70)


if __name__ == "__main__":
    main()
