import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel

from transformers import AutoModelForCausalLM, Qwen3VLConfig, Qwen3VLForConditionalGeneration

def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )


def build_model(model_name: str):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=build_quant_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False
    return model


def build_lora_config(args):
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def load_tuned_model(model_name: str, adapter_dir: str):
    base_model = build_model(model_name)
    tuned_model = PeftModel.from_pretrained(base_model, adapter_dir)
    tuned_model.eval()
    return tuned_model