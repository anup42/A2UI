# Re-export the existing rank-64 checkpoint with QAT-compatible weights

The explicit `--qat-compatible-weights` option addresses the BF16 rounding
ordering identified in the September 27 rank-64 metadata audit. It uses the
existing selected adapter and seed, without training or rewriting the source
checkpoint. The original `merged_best_hf` remains required for its existing
hash/provenance and ordinary-merge checks. The new package's adapted weight
buffers come from the QAT-compatible reconstruction instead.

The old exporter default is unchanged. The new mode is
`retained_scale_qat_compatible_weights_v1`, with arithmetic contract
`qat_bf16_delta_before_add_v1`:

1. Multiply the FP32 LoRA factors and apply alpha/rank with CPU autocast disabled.
2. Cast the delta to BF16 before adding it to the BF16 base. Check this weight
   calculation against the QAT helper.
3. Encode W2/W4 codes using the original retained FP32 scales. Compare the encoded
   codes and simulated BF16 dequantized weights with the CPU QAT weight forward.
4. Patch only the 205 adapted projections in the official package. Keep the graph,
   tokenizer, frozen weights, weight/activation scales, and MTP section intact.

The added gates test weight arithmetic and serialization. They do not establish
CUDA matmul/autocast equivalence or native CPU/GPU inference parity. The output is
a candidate for the small BXP-001/BXP-003 comparison with MTP disabled first.
Do not treat export success as evidence that the generation regression is fixed.

## Command for the supplied Linux training run

Run from the A2UI repository root after updating to the source containing this
option. Activate the same working CPU export environment used for the original
export; do not replace packages in the training environment. `python` below means
that environment's interpreter. The recorded `/group-volume/...` paths are Linux
training-host paths, not paths on the Windows PC containing only the metadata.

```bash
RUN_ID=e2b_v10_official_mobile_qat_lora_runD_r64_6144_8gpu_20260926_v1
WORK=/group-volume/k.anup/working_dir
RUN="$WORK/$RUN_ID"
TRAIN="$RUN/training/$RUN_ID"
SEED="$WORK/e2b_mobile_qat_seed_20260919"
OUT="$RUN/retained_scale_export_qat_compatible_$(date -u +%Y%m%d_%H%M%S)"

python -u training/scripts/build_gemma4_retained_scale_litertlm.py \
  --official-litertlm "$WORK/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm" \
  --official-artifact-sha256 181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c \
  --checkpoint "$RUN/merged_best_hf" \
  --adapter-checkpoint "$TRAIN/best_golden_checkpoint" \
  --training-config "$TRAIN/launch/resolved_training_config.yaml" \
  --mobile-training-seed-manifest "$SEED/mobile_training_seed_manifest.json" \
  --mobile-qparams-contract "$SEED/mobile_qparams.json" \
  --zero-adapter-checkpoint "$SEED" \
  --output-dir "$OUT" \
  --output-litertlm "$OUT/gemma4_e2b_a2ui_mobile.litertlm" \
  --qat-compatible-weights \
  --execute
```

Outputs in `$OUT`:

- `gemma4_e2b_a2ui_mobile.litertlm`
- `gemma4_retained_scale_code_only_report.json`

Leave out `--execute` for the provenance/input plan. The execute phase additionally
validates tensor values/dtypes and package contents. A fresh output directory is
required; the exporter refuses to overwrite an earlier package. Keep the original
adapter, seed, resolved config, and source merged checkpoint together on the
training host, because their hashes and paths are checked.

In the new report, inspect:

- `mode` and `weight_arithmetic` for the selected calculation.
- `gates.qat_weight_code_parity_205` and
  `gates.qat_dequantized_weight_forward_parity_205`.
- `trained_quantization.qat_weight_reconstruction.changed_code_count_from_merged_checkpoint`
  and its per-projection telemetry to measure the actual arithmetic impact.
- `output_sha256` to identify the new file when copying it to Android.
- `native_inference_parity_verified`, which remains `false` until separately tested.

The small counterexample and mocked-buffer integration tests do not export the
user's model on this PC. The actual weights remain on the training host.
