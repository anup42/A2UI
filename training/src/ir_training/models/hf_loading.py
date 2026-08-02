from __future__ import annotations

from typing import Any


def load_hf_model(model_id: str, config: dict[str, Any]):
    """Load a Hugging Face model without hard-coding one AutoModel class.

    Gemma 4 text-only/MTP examples use ``AutoModelForCausalLM`` while the
    multimodal model cards use ``AutoModelForMultimodalLM``. Keeping the
    loader selectable lets a pinned Transformers release follow either route.
    """

    try:
        import torch  # type: ignore
        import transformers  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError("Install the configured Hugging Face training environment before loading a model.") from exc

    loader_name = str(config.get("model_loader", "auto_causal_lm")).strip().lower()
    loader_class_name = {
        "auto_causal_lm": "AutoModelForCausalLM",
        "causal_lm": "AutoModelForCausalLM",
        "auto_multimodal_lm": "AutoModelForMultimodalLM",
        "multimodal_lm": "AutoModelForMultimodalLM",
    }.get(loader_name)
    if loader_class_name is None:
        raise ValueError(
            f"Unsupported model.model_loader {loader_name!r}. "
            "Use 'auto_causal_lm' or 'auto_multimodal_lm'."
        )
    loader_class = getattr(transformers, loader_class_name, None)
    if loader_class is None:
        raise RuntimeError(
            f"The installed Transformers release does not provide {loader_class_name}. "
            "Install training/requirements-gemma4-qat.txt or select a supported loader."
        )

    dtype_name = str(config.get("dtype", "bfloat16")).strip().lower()
    dtype = (
        torch.bfloat16
        if dtype_name in {"bf16", "bfloat16"}
        else torch.float16
        if dtype_name in {"fp16", "float16", "half"}
        else torch.float32
    )
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }
    # Transformers 5 renamed torch_dtype to dtype. Select by major version so
    # a TypeError raised for another reason is not swallowed and retried.
    version_text = str(getattr(transformers, "__version__", "4"))
    try:
        transformers_major = int(version_text.split(".", 1)[0])
    except ValueError:
        transformers_major = 4
    kwargs["dtype" if transformers_major >= 5 else "torch_dtype"] = dtype

    device_map = config.get("device_map", "auto")
    if not device_map_disabled(device_map):
        kwargs["device_map"] = device_map
    attn_implementation = str(config.get("attn_implementation", "")).strip()
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if bool(config.get("load_in_4bit", False)):
        bitsandbytes_config = getattr(transformers, "BitsAndBytesConfig", None)
        if bitsandbytes_config is None:
            raise RuntimeError("The installed Transformers release does not provide BitsAndBytesConfig.")
        kwargs["quantization_config"] = bitsandbytes_config(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    return loader_class.from_pretrained(model_id, **kwargs)


def device_map_disabled(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "none", "null", "false", "off", "ddp"}
