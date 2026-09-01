"""Export a LiteRT-LM model with phone-GPU-compatible primitive math.

Some Android GPU delegates accept LiteRT Torch StableHLO composite nodes but
produce numerically invalid results for this model family. The official
Gemma 3 270M mobile graph expands RMSNorm and attention softmax into primitive
LiteRT operators, so the custom export follows that topology for Adreno.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


_LITERTLM_MODEL_TYPE = b"tf_lite_prefill_decode"
_RUNTIME_MODEL_TYPE = b"TF_LITE_PREFILL_DECODE"


def _normalize_litertlm_model_type(output_dir: str) -> Path:
    """Make LiteRT-LM's model-type metadata compatible with the Android runtime.

    Recent LiteRT-Torch exporters emit the prefill/decode model-type value in
    lowercase, while the LiteRT-LM Android runtime performs an exact lookup of
    the historical uppercase value. Both strings have the same length, so the
    package can be normalized without changing section offsets or checksums.
    """

    root = Path(output_dir)
    candidates = [root] if root.is_file() else sorted(root.rglob("*.litertlm"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one LiteRT-LM package under {output_dir}, "
            f"found {len(candidates)}."
        )
    package = candidates[0]
    payload = package.read_bytes()
    lower_count = payload.count(_LITERTLM_MODEL_TYPE)
    upper_count = payload.count(_RUNTIME_MODEL_TYPE)
    if lower_count != 1:
        if upper_count == 1:
            return package
        raise RuntimeError(
            "Could not identify exactly one LiteRT-LM prefill/decode model-type "
            f"value in {package}: lowercase={lower_count}, uppercase={upper_count}."
        )
    package.write_bytes(payload.replace(_LITERTLM_MODEL_TYPE, _RUNTIME_MODEL_TYPE, 1))
    return package


def _normalize_gpu_operator_versions(package: Path) -> int:
    """Lower unsupported LiteRT GPU operator versions in the TFLite section.

    The current LiteRT quantizer emits weight-only INT8 graphs with
    ``DEQUANTIZE`` version 5.  LiteRT-LM 0.16.1's Android OpenCL delegate on
    SM8850 accepts the same INT8-per-channel tensors but only implements
    ``DEQUANTIZE`` through version 3.  The version is a scalar in the
    existing FlatBuffer OperatorCode table, so lowering only that field keeps
    all tensors, quantization parameters, offsets, and package sections
    unchanged.
    """

    payload = bytearray(package.read_bytes())
    identifier_offsets = []
    cursor = 0
    while True:
        offset = payload.find(b"TFL3", cursor)
        if offset < 0:
            break
        identifier_offsets.append(offset)
        cursor = offset + 1
    if not identifier_offsets:
        raise RuntimeError(f"No embedded TFLite model identifier found in {package}.")

    def u16(offset: int) -> int:
        return struct.unpack_from("<H", payload, offset)[0]

    def u32(offset: int) -> int:
        return struct.unpack_from("<I", payload, offset)[0]

    def i32(offset: int) -> int:
        return struct.unpack_from("<i", payload, offset)[0]

    def table_field(table_offset: int, field_index: int) -> int | None:
        vtable = table_offset - i32(table_offset)
        vtable_size = u16(vtable)
        entry = vtable + 4 + field_index * 2
        if entry + 2 > vtable + vtable_size:
            return None
        distance = u16(entry)
        return table_offset + distance if distance else None

    def follow(field_offset: int | None) -> int | None:
        if field_offset is None:
            return None
        relative = u32(field_offset)
        return field_offset + relative if relative else None

    changed = 0
    for identifier_offset in identifier_offsets:
        model_start = identifier_offset - 4
        root = model_start + u32(model_start)
        operator_codes_vector = follow(table_field(root, 1))
        if operator_codes_vector is None:
            continue
        operator_code_count = u32(operator_codes_vector)
        for index in range(operator_code_count):
            element = operator_codes_vector + 4 + index * 4
            operator_code = follow(element)
            if operator_code is None:
                continue
            # OperatorCode fields are deprecated_builtin_code, custom_code,
            # version, builtin_code in the current TFLite schema.
            version_field = table_field(operator_code, 2)
            builtin_field = table_field(operator_code, 3)
            if builtin_field is None or version_field is None:
                continue
            # BuiltinOperator.DEQUANTIZE is stable at enum value 6.
            builtin_code = i32(builtin_field)
            version = i32(version_field)
            if builtin_code == 6 and version > 3:
                struct.pack_into("<i", payload, version_field, 3)
                changed += 1

    if changed:
        package.write_bytes(payload)
    return changed


def _normalize_external_embedder_input_types(package: Path) -> int:
    """Restore FP32 types on decoder inputs backed by the external embedder.

    LiteRT-Torch's FP16 pass changes function argument tensors to FLOAT16 even
    when the external-embedder wrapper contains an explicit FP32-to-FP16 CAST.
    LiteRT-LM 0.16 requires the auxiliary embedder output and decoder input to
    be FLOAT32, so patch only the two named decoder boundary tensors back to
    TensorType.FLOAT32. The existing CAST then converts them to the FP16
    internal graph without changing any weights or operator topology.
    """

    payload = bytearray(package.read_bytes())
    identifier_offsets = []
    cursor = 0
    while True:
        offset = payload.find(b"TFL3", cursor)
        if offset < 0:
            break
        identifier_offsets.append(offset)
        cursor = offset + 1
    if not identifier_offsets:
        raise RuntimeError(f"No embedded TFLite model identifier found in {package}.")

    def u16(offset: int) -> int:
        return struct.unpack_from("<H", payload, offset)[0]

    def u32(offset: int) -> int:
        return struct.unpack_from("<I", payload, offset)[0]

    def i32(offset: int) -> int:
        return struct.unpack_from("<i", payload, offset)[0]

    def table_field(table_offset: int, field_index: int) -> int | None:
        vtable = table_offset - i32(table_offset)
        vtable_size = u16(vtable)
        entry = vtable + 4 + field_index * 2
        if entry + 2 > vtable + vtable_size:
            return None
        distance = u16(entry)
        return table_offset + distance if distance else None

    def follow(field_offset: int | None) -> int | None:
        if field_offset is None:
            return None
        relative = u32(field_offset)
        return field_offset + relative if relative else None

    def string_value(field_offset: int | None) -> str:
        string_offset = follow(field_offset)
        if string_offset is None:
            return ""
        length = u32(string_offset)
        return bytes(payload[string_offset + 4 : string_offset + 4 + length]).decode(
            "utf-8", errors="ignore"
        )

    changed = 0
    target_names = {"decode_embeddings"}
    for identifier_offset in identifier_offsets:
        model_start = identifier_offset - 4
        root = model_start + u32(model_start)
        subgraphs_vector = follow(table_field(root, 2))
        if subgraphs_vector is None:
            continue
        for subgraph_index in range(u32(subgraphs_vector)):
            subgraph_element = subgraphs_vector + 4 + subgraph_index * 4
            subgraph = follow(subgraph_element)
            if subgraph is None:
                continue
            tensors_vector = follow(table_field(subgraph, 0))
            if tensors_vector is None:
                continue
            for tensor_index in range(u32(tensors_vector)):
                tensor_element = tensors_vector + 4 + tensor_index * 4
                tensor = follow(tensor_element)
                if tensor is None:
                    continue
                name = string_value(table_field(tensor, 3))
                if not (
                    name in target_names
                    or (name.startswith("prefill_") and name.endswith("_embeddings"))
                ):
                    continue
                type_field = table_field(tensor, 1)
                if type_field is not None and payload[type_field] == 1:
                    # TensorType.FLOAT32 is enum value 0; FLOAT16 is 1.
                    payload[type_field] = 0
                    changed += 1

    if changed:
        package.write_bytes(payload)
    return changed


def _install_gpu_compatible_primitives(*, mixed_precision: bool = False) -> None:
    import importlib

    from litert_torch.backend import composite

    original = composite.StableHLOCompositeBuilder

    def _make_selective_builder(disabled_names):
        """Remove only the GPU-broken composites; preserve all other lowering."""

        class _SelectiveBuilder:
            def __init__(self, name, attr=None):
                self._delegate = (
                    None
                    if name in disabled_names
                    else original(name=name, attr=attr)
                )

            def mark_inputs(self, *values):
                if self._delegate is None:
                    return values[0] if len(values) == 1 else values
                return self._delegate.mark_inputs(*values)

            def mark_outputs(self, *values):
                if self._delegate is None:
                    return values[0] if len(values) == 1 else values
                return self._delegate.mark_outputs(*values)

        return _SelectiveBuilder

    # ``odml.rms_norm`` is the numerical failure seen in the custom model's
    # Android GPU run. ``odml.softmax`` and the fused SDPA boundary are also
    # expanded to primitive math so the exported topology matches the working
    # official mobile graph. Keep cache/update and other unrelated composites
    # intact: disabling every builder changes the embedding lowering into a
    # large float STABLEHLO_SCATTER graph and duplicates ~671 MB of weights.
    disabled = {
        "odml.rms_norm",
        "odml.softmax",
        "odml.scaled_dot_product_attention",
    }
    selective_builder = _make_selective_builder(disabled)
    composite.StableHLOCompositeBuilder = selective_builder

    # normalization.py and scaled_dot_product_attention.py import this class
    # by value from litert_torch.hlfb before the exporter is called. Patching
    # only backend.composite therefore removes the attention marker but leaves
    # every RMSNorm marker in the graph. Update those already-imported aliases
    # explicitly, while keeping the patch scoped to this exporter process.
    hlfb = importlib.import_module("litert_torch.hlfb")
    hlfb.StableHLOCompositeBuilder = selective_builder
    for module_name in (
        "litert_torch.generative.layers.normalization",
        "litert_torch.generative.layers.scaled_dot_product_attention",
    ):
        module = importlib.import_module(module_name)
        if hasattr(module, "StableHLOCompositeBuilder"):
            module.StableHLOCompositeBuilder = selective_builder

    # The official Gemma mobile graph expresses RMSNorm as SUM -> MUL(1/d) ->
    # RSQRT.  LiteRT's Adreno path is sensitive to the MEAN lowering emitted by
    # the generic RMSNorm implementation even when every node is delegated.
    # Keep the same arithmetic but spell the reduction like the known-good
    # mobile graph.  This patch is local to the exporter process.
    import torch

    normalization = importlib.import_module(
        "litert_torch.generative.layers.normalization"
    )

    def sum_based_rms_norm(self, x):
        dim = int(x.shape[-1])
        squared_sum = torch.sum(x.pow(2), dim=-1, keepdim=True)
        # Keep the reciprocal as a runtime-shaped scale tensor. LiteRT-Torch's
        # HF optimization pass rewrites a literal SUM * (1/d) back to MEAN;
        # that is precisely the reduction form avoided for Adreno.
        scale = torch.ones_like(squared_sum) * (1.0 / float(dim))
        mean_square = squared_sum * scale
        return x * torch.rsqrt(mean_square + self.eps)

    def sum_based_rms_norm_with_hlfb(x, w, eps, final_scale):
        builder = normalization.StableHLOCompositeBuilder(
            name="odml.rms_norm", attr={"epsilon": eps}
        )
        x, w = builder.mark_inputs(x, w)
        dim = int(x.shape[-1])
        squared_sum = torch.sum(x.float().pow(2), dim=-1, keepdim=True)
        scale = torch.ones_like(squared_sum) * (1.0 / float(dim))
        mean_square = squared_sum * scale
        output = x.float() * torch.rsqrt(mean_square + eps)
        out = output.type_as(x) * final_scale * w
        return builder.mark_outputs(out)

    normalization.RMSNorm._norm = sum_based_rms_norm
    normalization.rms_norm_with_hlfb = sum_based_rms_norm_with_hlfb

    # The exporter invokes this optional ModelUtils pass after lowering. Its
    # ``fuse_mean`` rewrite recognizes every literal reciprocal scale and turns
    # the SUM form above back into TFLite MEAN. Skip that optimization pass for
    # this mobile export; the remaining LiteRT converter passes are still used.
    mu_pass_lib = importlib.import_module(
        "litert_torch.generative.export_hf.core.mu.mu_pass_lib"
    )
    mu_pass_lib.update_model = lambda model: model

    if mixed_precision:
        # LiteRT's generic mixed-precision defaults preserve reductions and
        # attention in FP32. That is numerically conservative on CPU, but the
        # Android Adreno delegate used by LiteRT-LM requires one fully
        # delegated FP16 graph for this package. Keep the converter graph in
        # FP32 (see the export call below), then lower every floating-point
        # operation to FP16 in this post-conversion pass. In particular, do
        # not preserve BATCH_MATMUL or ADD: doing so creates a mixed graph
        # where the GPU delegate accepts only a small prefix of the nodes.
        mixed_precision_lib = importlib.import_module(
            "litert_torch.generative.export_hf.core.mu.mixed_precision"
        )

        def fp32_predicate(op):
            return False

        mixed_precision_lib.default_fp32_predicate = fp32_predicate
        mixed_precision_lib.convert_to_fp16.__defaults__ = (fp32_predicate,)

    # The Android GPU delegate accepts the primitive arithmetic below but its
    # ordinary SOFTMAX lowering is numerically invalid for this graph. Keep
    # the calculation in FP32 and return the dtype expected by PyTorch's
    # functional softmax API.
    import torch.nn.functional as functional

    def stable_primitive_softmax(input, dim=None, _stacklevel=3, dtype=None):
        axis = -1 if dim is None else dim
        work = input.to(torch.float32)
        # LiteRT's NHWC layout rewriter has a rule for ``aten.max.dim`` but
        # not for ``aten.amax``.  Keep this reduction in the supported form;
        # ``values`` is the same max-reduction result needed by softmax.
        maximum = torch.max(work, dim=axis, keepdim=True).values
        numerator = torch.exp(work - maximum)
        result = numerator / torch.sum(numerator, dim=axis, keepdim=True)
        return result.to(dtype if dtype is not None else input.dtype)

    functional.softmax = stable_primitive_softmax


def _install_post_quantization_mixed_precision() -> None:
    """Apply the GPU FP16 pass after quantization has seen FP32 weights.

    LiteRT's quantizer does not select the embedding/linear weights once the
    graph has already been converted to FP16.  Applying mixed precision in
    the normal exporter hook therefore produces an all-FP16 graph with an
    unsupported unquantized ``EMBEDDING_LOOKUP``.  Wrap the quantization step
    instead: quantize the original FP32 flatbuffer first, then lower its
    remaining floating-point compute to FP16 while preserving INT8 weights.
    """

    import importlib

    export_lib = importlib.import_module(
        "litert_torch.generative.export_hf.core.export_lib"
    )
    mixed_precision_lib = importlib.import_module(
        "litert_torch.generative.export_hf.core.mu.mixed_precision"
    )
    original_maybe_quantize_model = export_lib.maybe_quantize_model

    def quantize_then_convert_to_fp16(model_path, quantization_recipe=None):
        result = original_maybe_quantize_model(model_path, quantization_recipe)
        if quantization_recipe:
            converted = mixed_precision_lib.convert_model_to_fp16(
                result,
                fp32_op_predicate=lambda op: False,
            )
            Path(result).write_bytes(converted)
        return result

    export_lib.maybe_quantize_model = quantize_then_convert_to_fp16


def _install_embedding_quantization_recipe() -> None:
    """Expose direct INT4/INT8 ``EMBEDDING_LOOKUP`` recipes to LiteRT-Torch.

    The stock weight-only recipes quantize the embedding table and then insert
    a FLOAT32 ``DEQUANTIZE`` before ``EMBEDDING_LOOKUP``.  LiteRT's Adreno
    delegate rejects that form because the lookup sees an unquantized float
    table.  The dynamic recipe uses the quantized lookup kernel directly and
    leaves the rest of the graph untouched; the post-quantization FP16 pass
    then lowers the remaining floating-point compute.

    LiteRT-Torch resolves recipe names with ``ai_edge_quantizer.recipe.__dict__``
    inside its exporter, so registering this small project-owned recipe keeps
    the command line reproducible without modifying the installed package.
    """

    from ai_edge_quantizer import recipe, recipe_manager

    def dynamic_embedding(bits):
        manager = recipe_manager.RecipeManager()
        manager.add_dynamic_config(
            regex=".*",
            operation_name=recipe.TFLOperationName.EMBEDDING_LOOKUP,
            num_bits=bits,
        )
        return manager.get_quantization_recipe()

    recipe.dynamic_embedding_wi8_afp32 = lambda: dynamic_embedding(8)
    recipe.dynamic_embedding_wi4_afp32 = lambda: dynamic_embedding(4)


def _install_external_embedder_fp32_bridge() -> None:
    """Keep the embedder FP32 and cast its decoder input to FP16.

    LiteRT-LM 0.16 requires an external embedder to publish FLOAT32 output,
    but the Adreno decoder graph must execute in FP16.  The bridge therefore
    keeps the package's auxiliary embedder output at the runtime contract's
    FLOAT32 type and inserts a single FP32-to-FP16 CAST at the decoder boundary.
    The cast is represented in the decoder graph, so the GPU delegate can
    consume the same FP16 graph used by the non-external export.
    """

    import importlib
    import torch

    external_module = importlib.import_module(
        "litert_torch.generative.export_hf.core.external_emb.exportable_module"
    )

    def bridge_forward(original_forward):
        def forward(self, embeddings, input_pos, kv_cache, mask, **kwargs):
            return original_forward(
                self,
                embeddings.to(torch.float16),
                input_pos,
                kv_cache,
                mask,
                **kwargs,
            )

        return forward

    for class_name in (
        "LiteRTExportableModuleForDecoderOnlyLMPrefillExternalEmbedder",
        "LiteRTExportableModuleForDecoderOnlyLMGenerateExternalEmbedder",
    ):
        exportable_class = getattr(external_module, class_name)
        exportable_class.forward = bridge_forward(exportable_class.forward)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export LiteRT-LM with primitive RMSNorm/attention math for Android GPU."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--prefill-length", type=int, default=2048)
    parser.add_argument(
        "--prefill-lengths",
        default="",
        help=(
            "Comma-separated prefill graph lengths. When supplied, export all "
            "listed lengths so LiteRT-LM can select the closest graph at runtime."
        ),
    )
    parser.add_argument(
        "--quantization-recipe", default="dynamic_wi8_afp32"
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Keep exported weights in their source floating-point representation for diagnosis.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help=(
            "Convert the exported graph to LiteRT mixed FP16 precision while "
            "retaining numerically sensitive reductions in FP32."
        ),
    )
    parser.add_argument(
        "--sampler-top-k",
        type=int,
        default=1,
        help=(
            "Bake the runtime sampler policy into the LiteRT-LM metadata. "
            "The Android test app uses greedy decoding (top-k=1), so keeping "
            "the export metadata identical avoids selecting the optional GPU "
            "Top-K sampler at runtime."
        ),
    )
    parser.add_argument(
        "--static-cache-graph",
        action="store_true",
        help="Keep fixed prefill/cache shapes instead of LiteRT's dynamic GPU graph path.",
    )
    parser.add_argument(
        "--split-cache",
        action="store_true",
        help="Use LiteRT-Torch's split-cache topology for compact mobile packages.",
    )
    parser.add_argument(
        "--externalize-embedder",
        action="store_true",
        help="Put the token embedder in LiteRT-LM's auxiliary embedder section.",
    )
    args = parser.parse_args()

    prefill_lengths = (
        [int(value.strip()) for value in args.prefill_lengths.split(",") if value.strip()]
        if args.prefill_lengths.strip()
        else [args.prefill_length]
    )
    if not prefill_lengths or any(value <= 0 for value in prefill_lengths):
        raise ValueError(f"Invalid prefill lengths: {prefill_lengths}")

    _install_gpu_compatible_primitives(mixed_precision=args.mixed_precision)
    if args.quantization_recipe in (
        "dynamic_embedding_wi8_afp32",
        "dynamic_embedding_wi4_afp32",
    ):
        _install_embedding_quantization_recipe()
    if args.externalize_embedder and args.mixed_precision and args.no_quantization:
        _install_external_embedder_fp32_bridge()
    post_quantization_mixed_precision = (
        args.mixed_precision and not args.no_quantization
    )
    if post_quantization_mixed_precision:
        _install_post_quantization_mixed_precision()
    from litert_torch.generative.export_hf.export import export

    export(
        model=args.model,
        output_dir=args.output_dir,
        # The LiteRT-Torch config has a non-null default recipe. Pass an
        # explicit empty value to select its documented floating-point path.
        quantization_recipe="" if args.no_quantization else args.quantization_recipe,
        cache_length=args.cache_length,
        prefill_lengths=prefill_lengths,
        use_jinja_template=True,
        enable_gpu_dynamic_prefill=not args.static_cache_graph,
        enable_gpu_dynamic_cache=not args.static_cache_graph,
        split_cache=args.split_cache,
        externalize_embedder=args.externalize_embedder,
        bundle_litert_lm=True,
        sampler_top_k=args.sampler_top_k,
        # For quantized exports, mixed precision is installed around the
        # quantizer above so the embedding table is quantized before FP16
        # lowering.  The no-quantization diagnostic keeps the direct hook.
        experimental_use_mixed_precision=(
            args.mixed_precision and args.no_quantization
        ),
        # Keep the converter's source graph in FP32.  Setting
        # ``experimental_use_fp16`` here changes the PyTorch export itself and
        # can leave attention BATCH_MATMUL with an FP32 activation and an FP16
        # weight before LiteRT's mixed-precision pass runs.  The latter is the
        # supported place to lower the already-converted LiteRT graph.
        experimental_use_fp16=False,
    )
    package = _normalize_litertlm_model_type(args.output_dir)
    patched_operator_versions = _normalize_gpu_operator_versions(package)
    patched_external_input_types = (
        _normalize_external_embedder_input_types(package)
        if args.externalize_embedder
        else 0
    )
    print(
        "Normalized LiteRT GPU operator versions for Android runtime: "
        f"DEQUANTIZE patched={patched_operator_versions}"
    )
    print(
        "Normalized external embedder decoder input types for Android runtime: "
        f"FLOAT16->FLOAT32 patched={patched_external_input_types}"
    )
    print(f"Normalized LiteRT-LM model type for Android runtime: {package}")


if __name__ == "__main__":
    main()
