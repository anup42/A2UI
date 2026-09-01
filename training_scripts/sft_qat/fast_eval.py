"""Fast generation-based evaluation callback.

Drop-in replacement for `qat_utils.EvalPredictionCallback` that is dramatically
faster and scores two sources (validation + test) per evaluation.

Why the original is slow, in order of impact:

  1. The KV cache is never enabled. `config.use_cache = False` is set at load
     (qat_utils.py:1121) and `gradient_checkpointing=True` keeps it off; nothing
     re-enables it for generation. Per-sample token-work goes from `P + N` to
     `N*P + N**2/2` -- orders of magnitude at P~1500, N~6144.
  2. Generation is a serial one-sample-at-a-time loop (qat_utils.py:671-722).
  3. `max_new_tokens` has no stopping criteria beyond the model's own default
     EOS, so a model that never emits it burns the full budget every sample.

This module fixes all three. It is additive: `qat_utils.py` and
`train_qat_unsloth.py` are imported read-only and never modified.
"""

import contextlib
import gc
import json
import os
import time
import traceback

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback

from metrics import MetricsAggregator
from qat_utils import _dist_info, _save_checkpoint, _unwrap

# Sentinel distinguishing "attribute was absent" from "attribute was None".
_MISSING = object()

# Turn terminators across common chat templates. Gemma trains with
# add_generation_prompt=False (dataloader.py:87), so the assistant turn ends with
# <end_of_turn> -- that, not <eos>, is what a fine-tuned Gemma actually emits.
DEFAULT_STOP_STRINGS = ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|end|>")

# Must match dataloader.py:67 verbatim -- used to make prefixing idempotent.
IR_PREFIX = ("Given an agent response you have to generate a structured "
             "intermediate representation. ")


# ---------------------------------------------------------------------------
# Token / device helpers
# ---------------------------------------------------------------------------

def get_model_device(model):
    """Device to place generation inputs and gather tensors on.

    `qat_utils._score_and_save` uses `model.device` directly (qat_utils.py:754),
    which a DDP wrapper does not expose. Unwrap first, then fall back to the
    first parameter.
    """
    base = _unwrap(model)
    dev = getattr(base, "device", None)
    if isinstance(dev, torch.device):
        return dev
    for p in base.parameters():
        return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_configs(model, max_depth: int = 8):
    """Every distinct config object on the wrapper chain, outermost first.

    Setting `use_cache` on one config is not enough. PEFT's `__getattr__`
    forwarding is version-dependent, so the config the decoder's forward
    actually reads may be several wrappers down (.module -> .base_model ->
    .model). Collect them all and set the flag on each.
    """
    found, seen = [], set()
    node, depth = model, 0
    while node is not None and depth < max_depth:
        for attr in ("config", "generation_config"):
            cfg = getattr(node, attr, None)
            if cfg is not None and id(cfg) not in seen:
                seen.add(id(cfg))
                found.append(cfg)
        nxt = None
        for attr in ("module", "base_model", "model"):
            cand = getattr(node, attr, None)
            if cand is not None and cand is not node and isinstance(cand, torch.nn.Module):
                nxt = cand
                break
        node, depth = nxt, depth + 1
    return found


def resolve_pad_token_id(tokenizer) -> int:
    """Pad id for generate(), WITHOUT mutating the tokenizer.

    Deliberately does not do `tokenizer.pad_token = tokenizer.eos_token`: this
    tokenizer is the trainer's live `processing_class`, and changing pad_token
    mid-run would change the SFT collator's label masking for every subsequent
    training step.
    """
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, (list, tuple)) and eos:
        return int(eos[0])
    if eos is not None:
        return int(eos)
    raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id; "
                     "cannot pad a generation batch.")


def resolve_eos_token_ids(tokenizer, extra_stop_strings=DEFAULT_STOP_STRINGS):
    """Stop-token id list for generate(eos_token_id=...).

    Without <end_of_turn> a fine-tuned Gemma generates its answer, emits the
    turn terminator, and then keeps going until max_new_tokens -- which is pure
    wasted compute and can also corrupt the prediction with trailing text.
    """
    ids = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        ids.extend(int(e) for e in eos if e is not None)
    elif eos is not None:
        ids.append(int(eos))

    unk = getattr(tokenizer, "unk_token_id", None)
    added = {}
    try:
        added = tokenizer.get_added_vocab() or {}
    except Exception:
        pass

    for name in (extra_stop_strings or ()):
        tid = added.get(name)
        if tid is None:
            try:
                tid = tokenizer.convert_tokens_to_ids(name)
            except Exception:
                tid = None
        # Fast tokenizers return None for unknown tokens; slow ones return unk.
        if tid is None or int(tid) < 0 or (unk is not None and int(tid) == int(unk)):
            continue
        ids.append(int(tid))

    return list(dict.fromkeys(ids))  # dedupe, preserve order


def suggest_max_new_tokens(tokenizer, rows, output_field, percentile: float = 99.0,
                           slack: float = 1.25, hard_cap: int | None = None) -> int:
    """Data-driven generation cap from the reference lengths.

    Guessing this wrong in either direction is costly: too low silently truncates
    real outputs and corrupts metrics, too high lets one runaway sample charge
    its whole batch the full budget.
    """
    lengths = []
    for r in rows or ():
        ref = r.get(output_field) or ""
        if isinstance(ref, dict):
            ref = json.dumps(ref, ensure_ascii=False)
        if not ref:
            continue
        try:
            lengths.append(len(tokenizer(text=ref, add_special_tokens=False)["input_ids"]))
        except Exception:
            continue
    if not lengths:
        return hard_cap or 2048
    lengths.sort()
    idx = min(len(lengths) - 1, max(0, int(round((percentile / 100.0) * (len(lengths) - 1)))))
    suggested = int(lengths[idx] * slack) + 16
    if hard_cap:
        suggested = min(suggested, hard_cap)
    return max(16, suggested)


def left_pad_batch(list_of_ids, pad_id: int, device):
    """Left-pad a list of token-id lists into {input_ids, attention_mask}.

    Left padding (not right) is required for batched decoder-only generation:
    every row's last real token must sit at the same position so the first
    generated token is at a common index. Padding is done manually rather than
    via `tokenizer.padding_side` so no tokenizer state is touched.
    """
    maxlen = max((len(ids) for ids in list_of_ids), default=0)
    input_ids, attn = [], []
    for ids in list_of_ids:
        pad_n = maxlen - len(ids)
        input_ids.append([pad_id] * pad_n + list(ids))
        attn.append([0] * pad_n + [1] * len(ids))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attn, dtype=torch.long, device=device),
    }


def build_prompt(tokenizer, input_text: str) -> str:
    """Chat-templated generation prompt. Same fallback as qat_utils.py:693."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return f"<|im_start|>user\n{input_text}\n<|im_start|>assistant\n"


def encode_prompt_ids(tokenizer, prompt: str, max_prompt_tokens: int,
                      truncation_side: str = "left"):
    """Tokenize and truncate manually, mutating no tokenizer state.

    Default truncation_side is "left", which differs from the original callback.
    qat_utils.py:696 truncates with the tokenizer default ("right"), chopping the
    END of an over-length prompt -- including the `<start_of_turn>model`
    generation cue -- leaving the model nothing to answer.

    text= is passed as a keyword, not positionally: Gemma-4's processor is
    multi-modal (__call__(images=None, text=None, ...), images first), so a
    positional call lands the prompt in `images` and leaves text=None, which
    crashes downstream with "'NoneType' object is not subscriptable". text= is
    unambiguous for both plain tokenizers and multi-modal processors.
    """
    ids = tokenizer(text=prompt, add_special_tokens=False)["input_ids"]
    if max_prompt_tokens > 0 and len(ids) > max_prompt_tokens:
        ids = ids[-max_prompt_tokens:] if truncation_side == "left" else ids[:max_prompt_tokens]
    return ids


def finished_mask(gen_ids: torch.Tensor, eos_ids):
    """[B] bool: did each row emit a stop token? False => it hit the cap."""
    if gen_ids.numel() == 0:
        return torch.zeros(gen_ids.shape[0], dtype=torch.bool)
    hit = torch.zeros(gen_ids.shape[0], dtype=torch.bool, device=gen_ids.device)
    for tid in eos_ids or ():
        hit |= (gen_ids == int(tid)).any(dim=1)
    return hit.cpu()


@contextlib.contextmanager
def generation_mode(model, use_unsloth_inference: bool = False):
    """Cache-enabled eval state, exactly restored on exit. Yields the unwrapped module.

    Three things have to line up for the KV cache to actually be used:
      * `use_cache=True` on every config in the wrapper chain,
      * `model.eval()` -- HF decoder layers force the cache OFF whenever
        `self.gradient_checkpointing and self.training`, so without eval() the
        flag above is silently ignored,
      * `use_cache=True` passed to generate() as well (belt and braces; which of
        the three is authoritative varies by transformers version).

    Restoring is not optional: leaking `use_cache=True` into the next training
    step allocates a KV cache under gradient checkpointing, costing memory or
    OOMing mid-run.

    Gemma-4 MoE has incompatible cache structure (5D vs 4D tensor mismatch), so
    cache is always disabled for Gemma-4 during generation.
    """
    base = _unwrap(model)
    was_training = base.training

    # Gemma-4 MoE has incompatible cache; detect and disable
    model_type = getattr(base.config, "model_type", "").lower()
    is_gemma4_moe = "gemma4" in model_type

    saved = []
    for cfg in iter_configs(base):
        saved.append((cfg, getattr(cfg, "use_cache", _MISSING)))
        try:
            # Disable cache for Gemma-4 MoE to avoid index_copy_ dimension mismatch
            cfg.use_cache = False if is_gemma4_moe else True
        except Exception:
            pass
    base.eval()

    unsloth_applied = False
    if use_unsloth_inference:
        try:
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(base)
            unsloth_applied = True
        except Exception as exc:
            print(f"WARNING: FastLanguageModel.for_inference failed ({exc}); "
                  f"continuing on the standard path")
    try:
        yield base
    finally:
        if unsloth_applied:
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_training(base)
            except Exception as exc:
                print(f"WARNING: FastLanguageModel.for_training failed ({exc}); "
                      f"training state may be degraded")
        for cfg, old in saved:
            try:
                if old is _MISSING:
                    delattr(cfg, "use_cache")
                else:
                    cfg.use_cache = old
            except Exception:
                try:
                    cfg.use_cache = False
                except Exception:
                    pass
        if was_training:
            base.train()


# ---------------------------------------------------------------------------
# Distributed gather -- semantics identical to qat_utils.EvalPredictionCallback._gather
# ---------------------------------------------------------------------------

def gather_predictions(local_results, device, is_distributed, world_size):
    """Collect per-rank predictions and restore source order.

    Exactly 2 collectives regardless of how many predictions a rank produced --
    that invariant is what lets uneven shards and OOM-skipped rows coexist with
    DDP without deadlocking.
    """
    if not is_distributed:
        gathered = list(local_results)
    else:
        payload = [json.dumps(p) for p in local_results]
        local_count = torch.tensor([len(payload)], device=device, dtype=torch.long)
        counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(counts, local_count)                       # collective 1
        max_count = max(int(c.item()) for c in counts)
        padded = payload + [""] * (max_count - len(payload))
        all_padded = [None] * world_size
        dist.all_gather_object(all_padded, padded)                 # collective 2
        gathered = [json.loads(item) for proc_list in all_padded
                    for item in (proc_list or []) if isinstance(item, str) and item]

    gathered.sort(key=lambda p: p["global_idx"])
    for p in gathered:
        p.pop("global_idx", None)
    return gathered


# ---------------------------------------------------------------------------
# The callback
# ---------------------------------------------------------------------------

class FastEvalCallback(TrainerCallback):
    """Batched, cache-enabled generation eval over a validation and/or test source.

    Both sources arrive as plain `list[dict]` already normalized and capped by
    the caller. The callback never reads `eval_dataloader`, which is what makes
    the per-eval plan identical on every rank (see `on_evaluate`).
    """

    def __init__(self, tokenizer, input_field, output_field, *,
                 val_data=None, test_data=None,
                 max_length=6144, max_prompt_tokens=0, max_new_tokens=1536,
                 batch_size=8, min_batch_size=1, token_budget=0,
                 sort_by_length=True, truncation_side="left",
                 stop_strings=DEFAULT_STOP_STRINGS,
                 prediction_interval=1,
                 metric_for_best_model="mean_direct_match_score",
                 best_source="auto", output_dir=None,
                 save_predictions=True, save_predictions_interval=1,
                 save_predictions_limit=0, save_best_checkpoints=True,
                 use_unsloth_inference=False, empty_cache_after=True,
                 selftest=False):
        self.tokenizer = tokenizer
        self.input_field = input_field
        self.output_field = output_field
        self.val_data = list(val_data) if val_data else None
        self.test_data = list(test_data) if test_data else None
        self.max_length = max_length
        self.max_prompt_tokens = max_prompt_tokens or max_length
        self.batch_size = max(1, batch_size)
        self.min_batch_size = max(1, min_batch_size)
        self.token_budget = token_budget
        self.sort_by_length = sort_by_length
        self.truncation_side = truncation_side
        self.prediction_interval = max(1, prediction_interval)
        self.metric_for_best_model = metric_for_best_model
        self.best_source = best_source
        self.output_dir = output_dir
        self.save_predictions = save_predictions
        self.save_predictions_interval = max(1, save_predictions_interval)
        self.save_predictions_limit = save_predictions_limit
        self.save_best_checkpoints = save_best_checkpoints
        self.use_unsloth_inference = use_unsloth_inference
        self.empty_cache_after = empty_cache_after
        self.selftest = selftest

        self.pad_token_id = resolve_pad_token_id(tokenizer)
        self.eos_token_ids = resolve_eos_token_ids(tokenizer, stop_strings)
        self.metrics_aggregator = MetricsAggregator()

        if max_new_tokens is None or max_new_tokens <= 0:
            source = self.test_data or self.val_data or []
            max_new_tokens = suggest_max_new_tokens(tokenizer, source, output_field,
                                                    hard_cap=max_length)
            print(f"[fast_eval] measured max_new_tokens from data: {max_new_tokens}")
        self.max_new_tokens = max_new_tokens

        # Same field names as EvalPredictionCallback so print_best_checkpoints()
        # (qat_utils.py:1422) duck-types onto this object unchanged.
        self.best_metric_value = -float("inf")
        self.best_metric_step = -1
        self.best_model_saved = False
        self.best_val_loss = float("inf")
        self.best_val_loss_step = -1
        self.best_val_loss_model_saved = False

        # Set post-construction (trainer = SFTTrainer(...); callback.trainer = trainer)
        # so on_evaluate can force a checkpoint save BEFORE running generation. See
        # on_evaluate for why: HF's Trainer._maybe_log_save_evaluate always evaluates
        # before saving, so if generation OOMs, that step's checkpoint never happens.
        self.trainer = None

        self._bs_ceiling = self.batch_size   # sticky OOM degradation
        self._n_evals = 0
        self._eval_counters = {}             # per-PREFIX archive counter
        self._archived = {}                  # per-PREFIX archived paths
        self._selftest_done = False
        self._device = None
        self._reset_stats()

        if self.max_prompt_tokens + self.max_new_tokens > self.max_length:
            print(f"WARNING: max_prompt_tokens ({self.max_prompt_tokens}) + "
                  f"max_new_tokens ({self.max_new_tokens}) exceeds max_length "
                  f"({self.max_length}); generation may overrun the model window.")

        decoded = []
        for tid in self.eos_token_ids:
            try:
                decoded.append(f"{tid}={self.tokenizer.decode([tid])!r}")
            except Exception:
                decoded.append(str(tid))
        print(f"[fast_eval] stop tokens: {', '.join(decoded) or 'NONE'}")
        print(f"[fast_eval] batch_size={self.batch_size} "
              f"max_new_tokens={self.max_new_tokens} "
              f"sources={'+'.join(p for p, _, _ in self._plan()) or 'none'}")

    # -- stats -------------------------------------------------------------

    def _reset_stats(self):
        self._stat_new_tokens = []
        self._stat_truncated = 0
        self._stat_oom_skipped = 0

    def _free_cuda(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- planning ----------------------------------------------------------

    def _plan(self):
        """[(prefix, rows, num_samples)] built ONLY from constructor state.

        Rank-invariant by construction: no filesystem access, no `eval_dataloader`,
        no branch on rank. Every rank therefore runs the same number of
        collectives, which is the deadlock contract for DDP.
        """
        plan = []
        if self.val_data:
            plan.append(("eval", self.val_data, len(self.val_data)))
        if self.test_data:
            plan.append(("test", self.test_data, len(self.test_data)))
        return plan

    def _resolve_best_source(self, plan):
        names = [p for p, _, _ in plan]
        if self.best_source in ("test", "eval"):
            return self.best_source if self.best_source in names else None
        return "test" if "test" in names else ("eval" if "eval" in names else None)

    # -- generation --------------------------------------------------------

    def _build_items(self, data, num_samples, world_size, rank):
        """Round-robin shard, tokenize once, optionally length-sort.

        Shard matches qat_utils.py:673 so per-rank load is identical. `global_idx`
        rides along as a dict field, so the length sort is fully reversible --
        `gather_predictions` re-sorts on it.
        """
        items = []
        for gi in range(rank, num_samples, world_size):
            ex = data[gi]
            inp = ex.get(self.input_field, "") or ""
            if not inp:
                continue
            ids = encode_prompt_ids(self.tokenizer, build_prompt(self.tokenizer, inp),
                                    self.max_prompt_tokens, self.truncation_side)
            items.append({
                "global_idx": gi,
                "input": inp,
                "reference": ex.get(self.output_field, "") or "",
                "ids": ids,
                "n_tok": len(ids),
            })
        if self.sort_by_length:
            items.sort(key=lambda it: it["n_tok"])
        return items

    def _generate_one_batch(self, model, chunk):
        enc = left_pad_batch([it["ids"] for it in chunk], self.pad_token_id, self._device)
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            # Gemma-4 MoE: use_cache will be False if detected; other models use True
            use_cache_setting = getattr(model.config, "use_cache", True)
            out = model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=use_cache_setting,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_ids or None,
            )

        # Correct slice for LEFT padding: generated tokens start at the padded
        # width, NOT at attention_mask.sum(). eval.py:143-144 gets this wrong and
        # leaks prompt text into predictions for every row shorter than the max.
        gen = out[:, prompt_len:]
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        fin = finished_mask(gen, self.eos_token_ids)

        results = []
        for k, it in enumerate(chunk):
            results.append({
                "input": it["input"],
                "reference": it["reference"],
                "prediction": texts[k].strip(),
                "global_idx": it["global_idx"],
            })
            self._stat_new_tokens.append(int(gen.shape[1]))
            if not bool(fin[k]):
                self._stat_truncated += 1
        del enc, out, gen
        return results

    def _generate_local(self, model, data, num_samples, world_size, rank):
        """Generate this rank's shard in batches, halving on OOM.

        Contains NO `dist.*` call -- ranks may take completely different retry
        paths and still agree on collective count.
        """
        items = self._build_items(data, num_samples, world_size, rank)
        if rank == 0:
            print(f"\n[fast_eval] generating {num_samples} samples across "
                  f"{world_size} process(es), batch_size={self._bs_ceiling}")

        results, i = [], 0
        bs = max(self.min_batch_size, min(self.batch_size, self._bs_ceiling))
        while i < len(items):
            if self.token_budget > 0:
                longest = max(it["n_tok"] for it in items[i:i + bs])
                bs = max(self.min_batch_size,
                         min(bs, self.token_budget // max(1, longest)))
            chunk = items[i:i + bs]
            try:
                results.extend(self._generate_one_batch(model, chunk))
                i += len(chunk)
            except torch.cuda.OutOfMemoryError:
                self._free_cuda()
                if bs <= self.min_batch_size:
                    gi = chunk[0]["global_idx"]
                    print(f"WARNING: rank {rank} OOM at batch_size={bs}; emitting "
                          f"empty prediction for global_idx={gi}")
                    results.append({"input": chunk[0]["input"],
                                    "reference": chunk[0]["reference"],
                                    "prediction": "", "global_idx": gi})
                    self._stat_oom_skipped += 1
                    i += 1
                    continue
                bs = max(self.min_batch_size, bs // 2)
                self._bs_ceiling = bs   # sticky: fragmentation only grows
                print(f"WARNING: rank {rank} eval OOM; halving eval batch size to {bs}")
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                self._free_cuda()
                bs = max(self.min_batch_size, bs // 2)
                self._bs_ceiling = bs
                print(f"WARNING: rank {rank} eval OOM (RuntimeError); "
                      f"halving eval batch size to {bs}")
            if rank == 0 and results and len(results) % 25 < bs:
                print(f"  [fast_eval] {len(results)}/{len(items)} on rank 0")
        return results

    def _run_selftest(self, model, rows):
        """Batch-vs-single equivalence check on the first eval only.

        Greedy decoding must agree between bs=1 and bs=4 up to bf16
        nondeterminism. A systematic mismatch on the SHORT rows means left
        padding is mishandled for this transformers/model pair -- the check for
        cache_position / sliding-window cache bugs.
        """
        sample = rows[:4]
        if len(sample) < 2:
            return
        saved_mnt, self.max_new_tokens = self.max_new_tokens, 64
        try:
            items = self._build_items(sample, len(sample), 1, 0)
            single = []
            for it in items:
                single.extend(self._generate_one_batch(model, [it]))
            batched = self._generate_one_batch(model, items)
            single.sort(key=lambda p: p["global_idx"])
            batched.sort(key=lambda p: p["global_idx"])
            mismatches = sum(1 for a, b in zip(single, batched)
                             if a["prediction"] != b["prediction"])
            print(f"[fast_eval][selftest] {len(single) - mismatches}/{len(single)} rows "
                  f"agree between batch_size=1 and batch_size={len(items)}")
            if mismatches:
                print("[fast_eval][selftest] WARNING: batched output differs from "
                      "single-sample output. Left-padding handling may be broken for "
                      "this model/transformers pair -- consider "
                      "--fast_eval_batch_size 1 and verify before trusting metrics.")
        except Exception as exc:
            print(f"[fast_eval][selftest] skipped ({exc})")
        finally:
            self.max_new_tokens = saved_mnt

    # -- scoring / persistence --------------------------------------------

    def _score_and_save(self, state, prefix, predictions_data, elapsed):
        """Rank-0 only. Tolerates an empty prediction list."""
        if not predictions_data:
            print(f"WARNING: no predictions gathered for '{prefix}' at step "
                  f"{state.global_step}; skipping scoring")
            return None

        references = [p["reference"] for p in predictions_data]
        predictions = [p["prediction"] for p in predictions_data]
        batch = self.metrics_aggregator.compute_batch_metrics(references, predictions)

        n = len(predictions_data)
        mean_new = (sum(self._stat_new_tokens) / len(self._stat_new_tokens)
                    if self._stat_new_tokens else 0.0)
        trunc_ratio = self._stat_truncated / max(1, n)

        print(f"\n{prefix.capitalize()} prediction metrics (step {state.global_step}):")
        print(f"  Exact Match:      {batch['exact_match_accuracy']:.4f}")
        print(f"  Valid JSON:       {batch['valid_prediction_json_ratio']:.4f}")
        print(f"  Direct Match:     {batch['mean_direct_match_score']:.4f}")
        print(f"  Value-Only Match: {batch['mean_value_only_match_score']:.4f}")
        print(f"  Generation:       {elapsed:.1f}s for {n} samples "
              f"({n / max(elapsed, 1e-6):.2f}/s), mean {mean_new:.0f} new tokens")
        if self._stat_truncated:
            print(f"  WARNING: {self._stat_truncated}/{n} samples hit the "
                  f"max_new_tokens cap ({self.max_new_tokens}) without emitting a "
                  f"stop token -- predictions may be truncated")
        if self._stat_oom_skipped:
            print(f"  WARNING: {self._stat_oom_skipped} sample(s) emitted empty "
                  f"due to OOM")

        with open(os.path.join(self.output_dir, f"{prefix}_predictions.json"), "w",
                  encoding="utf-8") as f:
            json.dump(predictions_data, f, indent=2, ensure_ascii=False)

        self._archive_predictions(prefix, predictions_data, batch, state.global_step)

        summary = {
            "step": state.global_step,
            "num_samples": batch["num_samples"],
            "exact_match_accuracy": batch["exact_match_accuracy"],
            "valid_prediction_json_ratio": batch["valid_prediction_json_ratio"],
            "mean_direct_match_score": batch["mean_direct_match_score"],
            "std_direct_match_score": batch["std_direct_match_score"],
            "mean_value_only_match_score": batch["mean_value_only_match_score"],
            "std_value_only_match_score": batch["std_value_only_match_score"],
        }
        with open(os.path.join(self.output_dir, f"{prefix}_metrics.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        writer = SummaryWriter(log_dir=self.output_dir)
        for tag, key in (
            ("exact_match_accuracy", "exact_match_accuracy"),
            ("valid_json_ratio", "valid_prediction_json_ratio"),
            ("direct_match_score", "mean_direct_match_score"),
            ("value_only_match_score", "mean_value_only_match_score"),
            ("direct_match_std", "std_direct_match_score"),
            ("value_only_match_std", "std_value_only_match_score"),
        ):
            writer.add_scalar(f"{prefix}/{tag}", batch[key], state.global_step)
        writer.add_scalar(f"{prefix}/gen_seconds", elapsed, state.global_step)
        writer.add_scalar(f"{prefix}/gen_samples_per_s", n / max(elapsed, 1e-6),
                          state.global_step)
        writer.add_scalar(f"{prefix}/gen_mean_new_tokens", mean_new, state.global_step)
        writer.add_scalar(f"{prefix}/gen_truncated_ratio", trunc_ratio, state.global_step)
        writer.add_scalar(f"{prefix}/gen_batch_size", self._bs_ceiling, state.global_step)
        writer.add_scalar(f"{prefix}/gen_oom_skipped", self._stat_oom_skipped,
                          state.global_step)
        writer.close()

        return summary

    def _archive_predictions(self, prefix, predictions_data, batch, step):
        """Per-step copy so eval history survives the rolling file being rewritten.

        Counter is PER PREFIX. qat_utils.py:623 uses one shared counter, which
        with two sources per eval increments twice -- so
        save_predictions_interval=2 would archive only one source, never both.
        """
        if not self.save_predictions:
            return
        self._eval_counters[prefix] = self._eval_counters.get(prefix, 0) + 1
        if self._eval_counters[prefix] % self.save_predictions_interval != 0:
            return

        pred_dir = os.path.join(self.output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        path = os.path.join(pred_dir, f"{prefix}_predictions_step{step}.json")
        payload = {
            "step": step,
            "source": prefix,
            "num_samples": batch["num_samples"],
            "metrics": {
                "exact_match_accuracy": batch["exact_match_accuracy"],
                "valid_prediction_json_ratio": batch["valid_prediction_json_ratio"],
                "mean_direct_match_score": batch["mean_direct_match_score"],
                "mean_value_only_match_score": batch["mean_value_only_match_score"],
            },
            "predictions": predictions_data,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            print(f"WARNING: could not archive predictions to {path}: {exc}")
            return
        print(f"  Predictions archived: {path}")

        kept = self._archived.setdefault(prefix, [])
        kept.append(path)
        if self.save_predictions_limit > 0:
            while len(kept) > self.save_predictions_limit:
                stale = kept.pop(0)
                try:
                    os.remove(stale)
                except OSError:
                    pass

    def _maybe_save_best(self, model, state, source, summary):
        value = summary.get(self.metric_for_best_model, 0.0)
        if value <= self.best_metric_value:
            print(f"  {self.metric_for_best_model}: {value:.4f} "
                  f"(best {self.best_metric_value:.4f} @ step {self.best_metric_step})")
            return
        self.best_metric_value, self.best_metric_step = value, state.global_step
        print(f"  *** New best {self.metric_for_best_model}: {value:.4f} "
              f"at step {state.global_step} (source: {source}) ***")
        _save_checkpoint(
            model, self.tokenizer,
            os.path.join(self.output_dir, "best_test_checkpoint"),
            info={"best_metric": self.metric_for_best_model,
                  "best_metric_value": value, "best_step": state.global_step,
                  "selection_source": source, "all_metrics": summary},
            info_name="best_metric_info.json",
            copy_files=[os.path.join(self.output_dir, f"{source}_predictions.json"),
                        os.path.join(self.output_dir, f"{source}_metrics.json")],
        )
        self.best_model_saved = True

    # -- the hook ----------------------------------------------------------

    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None,
                    metrics=None, **kwargs):
        """Collective ledger (must be identical on every rank):

            interval skip   -> 0 all_gather, 0 all_gather_object, 1 barrier
            empty plan      -> 0, 0, 1
            one source      -> 1, 1, 2
            both sources    -> 2, 2, 3
            generation raised / batch halved -> unchanged

        Any divergence here hangs the job until the NCCL timeout.
        """
        is_distributed, world_size, rank = _dist_info()

        # ---- save checkpoint FIRST, before the generation loop below ----
        # HF's Trainer._maybe_log_save_evaluate calls evaluate() (which fires this
        # on_evaluate) BEFORE checking control.should_save and calling
        # _save_checkpoint -- unconditionally, hardcoded in transformers' Trainer.
        # If generation OOMs below, that step's checkpoint never happens. Save now
        # instead, then suppress HF's later save. Called unconditionally on every
        # rank (not rank-gated) -- matches how _maybe_log_save_evaluate itself
        # calls _save_checkpoint on every rank; introduces no new asymmetric
        # collective beyond what HF already does, so the ledger below is unaffected.
        if control.should_save and self.trainer is not None:
            if rank == 0:
                print(f"\n  [checkpoint-first] Saving checkpoint for step {state.global_step} "
                      f"before generation-based eval (avoids losing it if generation OOMs)")
            self.trainer._save_checkpoint(model, trial=None)
            control.should_save = False

        self._n_evals += 1   # unconditional on every rank -> cannot diverge

        # ---- best checkpoint by validation loss (rank 0, no collective) ----
        if metrics and "eval_loss" in metrics and rank == 0:
            val_loss = metrics["eval_loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss, self.best_val_loss_step = val_loss, state.global_step
                print(f"\n  *** New best eval_loss: {val_loss:.4f} at step "
                      f"{state.global_step} ***")
                _save_checkpoint(
                    model, self.tokenizer,
                    os.path.join(self.output_dir, "best_val_loss_checkpoint"),
                    info={"best_val_loss": val_loss, "best_step": state.global_step,
                          "all_metrics": metrics},
                    info_name="best_val_loss_info.json",
                )
                self.best_val_loss_model_saved = True
            else:
                print(f"\n  eval_loss: {val_loss:.4f} "
                      f"(best {self.best_val_loss:.4f} @ step {self.best_val_loss_step})")

        # ---- run-or-skip, from the internal counter only ----
        if (self._n_evals - 1) % self.prediction_interval != 0:
            if is_distributed:
                dist.barrier()
            return

        plan = self._plan()
        if not plan:
            if is_distributed:
                dist.barrier()
            return

        self._device = get_model_device(model)
        summaries = {}

        with generation_mode(model, self.use_unsloth_inference) as gen_model:
            if self.selftest and not self._selftest_done:
                self._run_selftest(gen_model, plan[0][1])
                self._selftest_done = True

            for prefix, rows, num_samples in plan:
                self._reset_stats()
                t0 = time.perf_counter()
                try:
                    local = self._generate_local(gen_model, rows, num_samples,
                                                 world_size, rank)
                except Exception as exc:
                    # MUST fall through to the collectives below -- returning here
                    # would leave the other ranks blocked in all_gather.
                    traceback.print_exc()
                    print(f"ERROR: rank {rank} generation failed for '{prefix}': {exc}")
                    local = []
                elapsed = time.perf_counter() - t0

                preds = gather_predictions(local, self._device, is_distributed, world_size)

                if rank == 0:
                    summaries[prefix] = self._score_and_save(state, prefix, preds, elapsed)

                if is_distributed:
                    dist.barrier()

        # ---- best checkpoint by generation metric (rank 0, no collective) ----
        if self.save_best_checkpoints and rank == 0:
            source = self._resolve_best_source(plan)
            if source and summaries.get(source):
                self._maybe_save_best(model, state, source, summaries[source])

        if self.empty_cache_after:
            self._free_cuda()

        if is_distributed:
            dist.barrier()
