"""Exhaustive tensor checks with bounded temporary memory, including on CUDA.

Full-model embedding gradients can contain billions of elements. A whole-tensor
pointwise check or reduction may allocate a temporary larger than the remaining
GPU memory. Inspect views in bounded chunks instead; never copy/flatten a whole
noncontiguous tensor, densify sparse gradients, or sample away invalid values.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

DEFAULT_CHECK_CHUNK_ELEMENTS = 1_048_576


def _chunks(tensor: Any, chunk_elements: int) -> Iterator[Any]:
    # Contiguous views may be flattened without a copy. For arbitrary strides,
    # recursively narrow a dimension; the stack contains only storage-sharing
    # views and has depth bounded by the tensor shape, not its element count.
    pending = [tensor]
    while pending:
        part = pending.pop()
        elements = part.numel()
        if elements <= chunk_elements:
            if elements:
                yield part
        elif part.is_contiguous():
            flat = part.view(-1)
            for offset in range(0, elements, chunk_elements):
                yield flat.narrow(0, offset, min(chunk_elements, elements - offset))
        else:
            dimension = max(range(part.ndim), key=lambda axis: part.shape[axis])
            length = part.shape[dimension]
            midpoint = length // 2
            pending.append(part.narrow(dimension, midpoint, length - midpoint))
            pending.append(part.narrow(dimension, 0, midpoint))


def _scan(tensor: Any, *, chunk_elements: int, check_nonzero: bool) -> tuple[bool, bool]:
    import torch

    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError("chunk_elements must be a positive integer.")
    if tensor.layout == torch.sparse_coo:
        # Preserve the existing gradient-check semantics: examine every stored
        # value, including uncoalesced entries, without allocating a dense copy.
        values = tensor._values().detach()
    elif tensor.layout == torch.strided:
        values = tensor.detach()
    else:
        raise ValueError(f"Bounded tensor checks do not support layout {tensor.layout}.")

    with torch.no_grad():
        finite = torch.ones((), dtype=torch.bool, device=values.device)
        nonzero = torch.zeros((), dtype=torch.bool, device=values.device)
        for chunk in _chunks(values, chunk_elements):
            finite.logical_and_(torch.isfinite(chunk).all())
            if check_nonzero:
                nonzero.logical_or_(torch.ne(chunk, 0).any())
        # Accumulate scalar flags on-device. Do not synchronize once per chunk,
        # and do not stop after finding a nonzero: a later NaN must still fail.
        return bool(finite.item()), bool(nonzero.item()) if check_nonzero else False


def tensor_finite_and_nonzero(
    tensor: Any, *, chunk_elements: int = DEFAULT_CHECK_CHUNK_ELEMENTS
) -> tuple[bool, bool]:
    """Return (all finite, any stored value nonzero); empty tensors are (True, False)."""
    return _scan(tensor, chunk_elements=chunk_elements, check_nonzero=True)


def tensor_all_finite(
    tensor: Any, *, chunk_elements: int = DEFAULT_CHECK_CHUNK_ELEMENTS
) -> bool:
    """Check every value without allocating a full-tensor mask or a nonzero mask."""
    return _scan(tensor, chunk_elements=chunk_elements, check_nonzero=False)[0]
