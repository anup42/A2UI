"""
Diagnose QAT checkpoint and conversion issues.
"""

import argparse
import json
import os
import torch
from pathlib import Path

def diagnose_checkpoint(checkpoint_path):
    """Check if checkpoint has QAT enabled."""
    print("\n" + "="*70)
    print("CHECKPOINT DIAGNOSIS")
    print("="*70)

    # Check for adapter files (LoRA)
    adapter_config = os.path.join(checkpoint_path, "adapter_config.json")
    adapter_model = os.path.join(checkpoint_path, "adapter_model.bin")

    print(f"\n[1] LoRA Adapters:")
    if os.path.exists(adapter_config):
        with open(adapter_config) as f:
            cfg = json.load(f)
        print(f"  ✓ adapter_config.json found")
        print(f"    - target_modules: {cfg.get('target_modules')}")
        print(f"    - r (rank): {cfg.get('r')}")
        print(f"    - lora_alpha: {cfg.get('lora_alpha')}")
    else:
        print(f"  ✗ No adapter_config.json")

    # Check both .bin and .safetensors formats
    adapter_safetensors = os.path.join(checkpoint_path, "adapter_model.safetensors")

    if os.path.exists(adapter_model):
        state = torch.load(adapter_model, map_location="cpu")
        print(f"  ✓ adapter_model.bin found ({len(state)} keys)")
        for k in list(state.keys())[:3]:
            print(f"    - {k}: {state[k].shape}")
    elif os.path.exists(adapter_safetensors):
        try:
            from safetensors.torch import load_file
            state = load_file(adapter_safetensors)
            print(f"  ✓ adapter_model.safetensors found ({len(state)} keys, {os.path.getsize(adapter_safetensors)/1e6:.1f}MB)")
            for k in list(state.keys())[:3]:
                print(f"    - {k}: {state[k].shape}")
        except Exception as e:
            print(f"  ✗ Could not load safetensors: {e}")
    else:
        print(f"  ✗ No adapter_model.bin or .safetensors")

    # Check for QAT scales
    print(f"\n[2] Quantization Scales (QAT):")
    adapter_state = torch.load(adapter_model, map_location="cpu") if os.path.exists(adapter_model) else {}

    qat_scales = [k for k in adapter_state.keys() if "scale" in k.lower() or "quant" in k.lower()]
    if qat_scales:
        print(f"  ✓ Found {len(qat_scales)} quantization-related keys:")
        for k in qat_scales[:5]:
            v = adapter_state[k]
            print(f"    - {k}: {v.shape}, min={v.min():.6f}, max={v.max():.6f}, mean={v.mean():.6f}")
    else:
        print(f"  ✗ No quantization scales found in adapter!")
        print(f"    WARNING: Model may not be QAT-trained!")
        print(f"    Keys in adapter: {list(adapter_state.keys())[:5]}")

    # Check training_args
    training_args_file = os.path.join(checkpoint_path, "training_args.bin")
    if os.path.exists(training_args_file):
        print(f"\n[3] Training Args:")
        try:
            args = torch.load(training_args_file, map_location="cpu")
            if hasattr(args, 'qat_scheme'):
                print(f"  ✓ qat_scheme: {args.qat_scheme}")
            if hasattr(args, 'num_train_epochs'):
                print(f"  - num_train_epochs: {args.num_train_epochs}")
            if hasattr(args, 'learning_rate'):
                print(f"  - learning_rate: {args.learning_rate}")
        except Exception as e:
            print(f"  ✗ Could not read training_args: {e}")

    print("\n" + "="*70)


def check_fake_vs_real_quantized(checkpoint_path, output_dir, test_data_path, max_samples=5):
    """Test: load fake-quantized, compare vs real-quantized."""
    print("\n" + "="*70)
    print("FAKE vs REAL QUANTIZATION TEST")
    print("="*70)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["FORCE_CUDA"] = "1"

    try:
        from unsloth import FastLanguageModel
        from transformers import set_seed
        from dataloader import load_test_data

        set_seed(42)

        print(f"\n[1] Loading fake-quantized checkpoint...")
        try:
            model_fake, tokenizer = FastLanguageModel.from_pretrained(
                model_name=checkpoint_path,
                max_seq_length=6144,
                load_in_4bit=False,
                load_in_8bit=False,
                load_in_16bit=True,
            )
            print(f"  ✓ Loaded fake-quantized model")

            # Check if it's actually quantized
            print(f"\n[2] Checking model quantization status:")
            has_quant = False
            for name, module in model_fake.named_modules():
                if "quant" in name.lower() or hasattr(module, 'fake_quant_enabled'):
                    has_quant = True
                    print(f"  ✓ Found quantization module: {name}")
                    break

            if not has_quant:
                print(f"  ✗ WARNING: No quantization modules found!")
                print(f"    Model may be loaded in bf16 without QAT!")

            # Generate on sample
            print(f"\n[3] Test generation on 1 sample (fake-quantized):")
            test_data = load_test_data(test_data_path, num_samples=1)
            if test_data:
                example = test_data[0]
                inp = example.get("response_text", "")

                messages = [{"role": "user", "content": inp}]
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except:
                    prompt = f"<|im_start|>user\n{inp}\n<|im_start|>assistant\n"

                try:
                    encoded = tokenizer(
                        prompt, return_tensors="pt", truncation=True, max_length=6144
                    ).to(model_fake.device)
                except TypeError:
                    encoded = tokenizer(
                        text=prompt, return_tensors="pt", truncation=True, max_length=6144
                    ).to(model_fake.device)

                model_fake.eval()
                with torch.no_grad():
                    output = model_fake.generate(
                        **encoded,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                input_len = encoded["input_ids"].shape[1]
                pred_fake = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
                print(f"  Generated ({len(pred_fake)} chars):")
                print(f"    {pred_fake[:200]}...")

        except Exception as e:
            print(f"  ✗ Failed to load fake-quantized: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n[4] Loading real-quantized model...")
        quantized_path = os.path.join(output_dir, "model_quantized")
        if os.path.exists(quantized_path):
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model_real = AutoModelForCausalLM.from_pretrained(
                    quantized_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                print(f"  ✓ Loaded real-quantized model")

                # Compare outputs
                print(f"\n[5] Test generation on same sample (real-quantized):")
                model_real.eval()
                with torch.no_grad():
                    output_real = model_real.generate(
                        **encoded,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                pred_real = tokenizer.decode(output_real[0][input_len:], skip_special_tokens=True).strip()
                print(f"  Generated ({len(pred_real)} chars):")
                print(f"    {pred_real[:200]}...")

                # Compare
                print(f"\n[6] Comparison:")
                if pred_fake == pred_real:
                    print(f"  ✓ Outputs are IDENTICAL (good sign)")
                else:
                    print(f"  ✗ Outputs DIFFER")
                    print(f"    Fake length: {len(pred_fake)}")
                    print(f"    Real length: {len(pred_real)}")

            except Exception as e:
                print(f"  ✗ Failed to load real-quantized: {e}")
        else:
            print(f"  ✗ Quantized model not found at {quantized_path}")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose QAT checkpoint and conversion")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to QAT checkpoint")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output dir with converted model (optional)")
    parser.add_argument("--test_data_path", type=str, default=None,
                       help="Test data for generation test (optional)")

    args = parser.parse_args()

    diagnose_checkpoint(args.checkpoint_path)

    if args.output_dir and args.test_data_path:
        check_fake_vs_real_quantized(
            args.checkpoint_path,
            args.output_dir,
            args.test_data_path,
        )
