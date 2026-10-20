"""Two-rank CPU proof of serial DPO backward and accumulation semantics."""
from __future__ import annotations

import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
distributed = pytest.importorskip("torch.distributed")

from ir_training.train.qat_dpo import (
    ReferenceLogps,
    completion_log_probability,
    dpo_loss,
    serial_dpo_backward,
)


class DDPTestPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_delta = torch.nn.Parameter(torch.arange(36, dtype=torch.float32).reshape(6, 6) / 100)

    def forward(self, input_ids, attention_mask, use_cache=False):
        return SimpleNamespace(logits=self.lora_delta[input_ids])


def _pair(rank, micro):
    row = {"pair_id": f"p{rank}_{micro}", "prompt_input_ids": [0, 1]}
    for side, token in (("chosen", 2 + rank + micro), ("rejected", 1)):
        ids = [0, 1, token, 5]
        row.update({f"{side}_input_ids": ids, f"{side}_labels": [-100, -100, token, 5],
                    f"{side}_attention_mask": [1] * 4, f"{side}_completion_mask": [False, False, True, True]})
    return row


def _worker(rank, init_method):
    torch.set_num_threads(1)
    distributed.init_process_group("gloo", rank=rank, world_size=2,
                                   init_method=init_method, timeout=timedelta(seconds=90))
    try:
        model = DDPTestPolicy()
        ddp = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
                                                       find_unused_parameters=False)
        for micro in range(2):
            row = _pair(rank, micro)
            with ddp.no_sync() if micro == 0 else nullcontext():
                serial_dpo_backward(ddp, [row], {row["pair_id"]: ReferenceLogps(row["pair_id"], -3., -4.)},
                                    beta=.1, scale=.5, backward=lambda loss: loss.backward())
        expected = DDPTestPolicy()
        losses = [dpo_loss(completion_log_probability(expected, _pair(r, m), "chosen"),
            completion_log_probability(expected, _pair(r, m), "rejected"), -3., -4., .1)
            for r in range(2) for m in range(2)]
        torch.stack(losses).mean().backward()
        assert torch.allclose(model.lora_delta.grad, expected.lora_delta.grad, atol=1e-7, rtol=1e-5)
    finally:
        distributed.destroy_process_group()


@pytest.mark.skipif(not distributed.is_available() or not distributed.is_gloo_available(), reason="CPU Gloo unavailable")
@pytest.mark.skipif(sys.platform == "win32", reason="Run two-rank Gloo test on Linux; this Windows host cannot complete subprocess rendezvous")
def test_two_rank_serial_dpo_matches_global_paired_gradient(tmp_path):
    context = torch.multiprocessing.spawn(_worker, args=((tmp_path / "gloo_init").as_uri(),), nprocs=2, join=False)
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Two-rank DPO Gloo probe did not finish within 120 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=10)
