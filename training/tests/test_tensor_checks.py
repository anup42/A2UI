from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ir_training.train.tensor_checks import tensor_all_finite, tensor_finite_and_nonzero


def _record_bounded_pointwise_calls(
    monkeypatch: pytest.MonkeyPatch, *, bound: int
) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []
    original_isfinite = torch.isfinite
    original_ne = torch.ne

    def checked_isfinite(value: torch.Tensor, *args, **kwargs):
        calls.append(("isfinite", value.numel()))
        assert value.numel() <= bound
        return original_isfinite(value, *args, **kwargs)

    def checked_ne(value: torch.Tensor, other, *args, **kwargs):
        calls.append(("ne", value.numel()))
        assert value.numel() <= bound
        return original_ne(value, other, *args, **kwargs)

    monkeypatch.setattr(torch, "isfinite", checked_isfinite)
    monkeypatch.setattr(torch, "ne", checked_ne)
    return calls


def _forbid_full_tensor_materialization(
    monkeypatch: pytest.MonkeyPatch, tensor: torch.Tensor
) -> None:
    original_contiguous = torch.Tensor.contiguous
    original_detach = torch.Tensor.detach
    original_reshape = torch.Tensor.reshape

    def checked_contiguous(self: torch.Tensor, *args, **kwargs):
        if self is tensor:
            pytest.fail("the full input tensor must not be made contiguous")
        return original_contiguous(self, *args, **kwargs)

    def checked_reshape(self: torch.Tensor, *args, **kwargs):
        if self is tensor:
            pytest.fail("the full input tensor must not be reshaped")
        return original_reshape(self, *args, **kwargs)

    def checked_detach(self: torch.Tensor, *args, **kwargs):
        detached = original_detach(self, *args, **kwargs)
        if self is tensor and self.numel():
            assert detached.untyped_storage().data_ptr() == self.untyped_storage().data_ptr()
            assert detached.storage_offset() == self.storage_offset()
            assert detached.stride() == self.stride()
        return detached

    monkeypatch.setattr(torch.Tensor, "contiguous", checked_contiguous)
    monkeypatch.setattr(torch.Tensor, "detach", checked_detach)
    monkeypatch.setattr(torch.Tensor, "reshape", checked_reshape)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([], (True, False)),
        ([0.0] * 11, (True, False)),
        ([0.0] * 10 + [2.0], (True, True)),
        ([1.0] + [0.0] * 9 + [float("nan")], (False, True)),
        ([1.0] + [0.0] * 9 + [float("inf")], (False, True)),
        ([1.0] + [0.0] * 9 + [float("-inf")], (False, True)),
    ],
)
def test_dense_results_include_late_nonfinite_values(values, expected):
    tensor = torch.tensor(values, dtype=torch.float32)
    assert tensor_finite_and_nonzero(tensor, chunk_elements=4) == expected
    assert tensor_all_finite(tensor, chunk_elements=4) is expected[0]


def test_dense_checks_bound_every_pointwise_operand(monkeypatch):
    calls = _record_bounded_pointwise_calls(monkeypatch, bound=4)
    tensor = torch.arange(13, dtype=torch.float32)

    assert tensor_finite_and_nonzero(tensor, chunk_elements=4) == (True, True)
    assert tensor_all_finite(tensor, chunk_elements=4)

    assert calls
    assert sum(name == "isfinite" for name, _ in calls) >= 8
    assert any(name == "ne" for name, _ in calls)


@pytest.mark.parametrize(
    "make_tensor",
    [
        pytest.param(lambda: torch.arange(24.0).reshape(4, 6).t(), id="transpose"),
        pytest.param(lambda: torch.arange(30.0)[1::2], id="strided-slice"),
        pytest.param(lambda: torch.tensor([0.0, 3.0]).expand(7, 2), id="zero-stride"),
    ],
)
def test_noncontiguous_views_are_chunked_without_full_copy(
    monkeypatch, make_tensor: Callable[[], torch.Tensor]
):
    tensor = make_tensor()
    assert not tensor.is_contiguous()
    expected = (True, bool(torch.ne(tensor, 0).any().item()))
    calls = _record_bounded_pointwise_calls(monkeypatch, bound=4)
    _forbid_full_tensor_materialization(monkeypatch, tensor)

    assert tensor_finite_and_nonzero(tensor, chunk_elements=4) == expected
    assert tensor_all_finite(tensor, chunk_elements=4)
    assert calls


def test_noncontiguous_late_nan_is_not_hidden_by_early_nonzero():
    base = torch.zeros((3, 5), dtype=torch.float32)
    base[0, 0] = 1.0
    base[-1, -1] = float("nan")
    tensor = base.t()

    assert tensor_finite_and_nonzero(tensor, chunk_elements=4) == (False, True)
    assert not tensor_all_finite(tensor, chunk_elements=4)


def test_every_noncontiguous_position_is_inspected():
    for position in range(15):
        base = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        base.reshape(-1)[position] = float("nan")
        tensor = base.t()

        assert tensor_finite_and_nonzero(tensor, chunk_elements=4) == (False, True)
        assert not tensor_all_finite(tensor, chunk_elements=4)


def test_scalar_inputs():
    assert tensor_finite_and_nonzero(torch.tensor(0.0), chunk_elements=4) == (True, False)
    assert tensor_finite_and_nonzero(torch.tensor(-2.0), chunk_elements=4) == (True, True)
    assert tensor_finite_and_nonzero(torch.tensor(float("nan")), chunk_elements=4) == (
        False,
        True,
    )
    assert tensor_all_finite(torch.tensor(7.0), chunk_elements=4)


@pytest.mark.parametrize(
    "tensor",
    [
        pytest.param(torch.tensor([False, True]), id="bool"),
        pytest.param(torch.tensor([0, -3], dtype=torch.int64), id="int64"),
        pytest.param(torch.tensor([0.0, 2.0], dtype=torch.float16), id="float16"),
        pytest.param(torch.tensor([0.0, -2.0], dtype=torch.float64), id="float64"),
        pytest.param(torch.tensor([0j, 1 + 2j], dtype=torch.complex64), id="complex64"),
    ],
)
def test_supported_dense_dtypes(tensor):
    assert tensor_finite_and_nonzero(tensor, chunk_elements=1) == (True, True)
    assert tensor_all_finite(tensor, chunk_elements=1)


def test_uncoalesced_sparse_coo_uses_bounded_value_chunks(monkeypatch):
    indices = torch.tensor([[0, 0, 2, 3, 3], [1, 1, 2, 0, 3]])
    values = torch.tensor([1.0, -1.0, 0.0, 2.0, float("nan")])
    tensor = torch.sparse_coo_tensor(
        indices, values, size=(4, 4), check_invariants=False
    )
    assert not tensor.is_coalesced()
    calls = _record_bounded_pointwise_calls(monkeypatch, bound=4)

    assert tensor_finite_and_nonzero(tensor, chunk_elements=4) == (False, True)
    assert not tensor_all_finite(tensor, chunk_elements=4)
    assert calls


def test_empty_sparse_coo_is_finite_and_zero():
    indices = torch.empty((2, 0), dtype=torch.long)
    values = torch.empty((0,), dtype=torch.float32)
    tensor = torch.sparse_coo_tensor(
        indices, values, size=(3, 5), check_invariants=False
    )

    assert tensor_finite_and_nonzero(tensor, chunk_elements=4) == (True, False)
    assert tensor_all_finite(tensor, chunk_elements=4)


@pytest.mark.parametrize("check", [tensor_finite_and_nonzero, tensor_all_finite])
@pytest.mark.filterwarnings("ignore:Sparse CSR tensor support is in beta state")
def test_unsupported_sparse_layout_fails_closed(check):
    tensor = torch.sparse_csr_tensor(
        torch.tensor([0, 1, 1]),
        torch.tensor([0]),
        torch.tensor([1.0]),
        size=(2, 2),
        check_invariants=True,
    )

    with pytest.raises(ValueError, match="do not support layout"):
        check(tensor, chunk_elements=4)


@pytest.mark.parametrize("chunk_elements", [0, -1, 1.5, True, "4", None])
@pytest.mark.parametrize("check", [tensor_finite_and_nonzero, tensor_all_finite])
def test_chunk_elements_must_be_a_positive_integer(check, chunk_elements):
    with pytest.raises((TypeError, ValueError)):
        check(torch.tensor([1.0]), chunk_elements=chunk_elements)


def test_checks_do_not_use_count_nonzero(monkeypatch):
    monkeypatch.setattr(
        torch,
        "count_nonzero",
        lambda *args, **kwargs: pytest.fail("torch.count_nonzero must not be used"),
    )
    assert tensor_finite_and_nonzero(torch.tensor([0.0, 1.0]), chunk_elements=1) == (
        True,
        True,
    )


@pytest.mark.parametrize("check", [tensor_finite_and_nonzero, tensor_all_finite])
def test_checks_do_not_mutate_tensor_grad_or_version(check):
    tensor = torch.tensor([0.0, 1.0, 2.0, 3.0], requires_grad=True)
    tensor.grad = torch.tensor([4.0, 5.0, 6.0, 7.0])
    tensor_id = id(tensor)
    grad = tensor.grad
    before = tensor.detach().clone()
    version = tensor._version

    result = check(tensor, chunk_elements=2)

    assert result in (True, (True, True))
    assert id(tensor) == tensor_id
    assert tensor.grad is grad
    assert torch.equal(tensor.detach(), before)
    assert torch.equal(tensor.grad, torch.tensor([4.0, 5.0, 6.0, 7.0]))
    assert tensor._version == version


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_checks_keep_peak_allocated_memory_bounded():
    device = torch.device("cuda", torch.cuda.current_device())
    tensor = torch.ones(16 * 1024 * 1024, dtype=torch.float32, device=device)
    tensor[-1] = 2.0
    torch.cuda.synchronize(device)

    tensor_id = id(tensor)
    data_ptr = tensor.data_ptr()
    version = tensor._version
    first = tensor[0].item()
    last = tensor[-1].item()
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    assert tensor_finite_and_nonzero(tensor, chunk_elements=65_536) == (True, True)
    assert tensor_all_finite(tensor, chunk_elements=65_536)
    torch.cuda.synchronize(device)

    extra_peak = torch.cuda.max_memory_allocated(device) - baseline
    assert extra_peak <= 8 * 1024 * 1024
    assert id(tensor) == tensor_id
    assert tensor.data_ptr() == data_ptr
    assert tensor._version == version
    assert tensor[0].item() == first
    assert tensor[-1].item() == last
