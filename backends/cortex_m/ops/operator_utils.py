# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dependency-light helpers shared by Cortex-M operator fake implementations."""

import torch

# L-shift value used in CMSIS-NN for int8 operations.
SHIFT_INT8 = 20


def dequantize_per_tensor_cmsis(
    qtensor: torch.Tensor, zero_point: int, multiplier: int, shift: int
) -> torch.Tensor:
    """Simulate CMSIS-NN fixed-point dequantization."""
    scale = multiplier * (2**shift) / (1 << 31)
    return (qtensor.float() - zero_point) * scale


def quantize_per_tensor_cmsis(
    tensor: torch.Tensor,
    zero_point: int,
    multiplier: int,
    shift: int,
    qmin: int = -128,
    qmax: int = 127,
) -> torch.Tensor:
    """Simulate CMSIS-NN fixed-point quantization."""
    scale = multiplier * (2**shift) / (1 << 31)
    quantized = torch.round(tensor / scale) + zero_point
    return quantized.clamp(qmin, qmax).to(torch.int8)


def requantize_cmsis(
    tensor: torch.Tensor,
    multiplier: int,
    shift: int,
) -> torch.Tensor:
    """Simulate CMSIS-NN's arm_nn_requantize helper."""
    tensor_64 = tensor.to(torch.int64)
    left_shift = max(shift, 0)
    right_shift = max(-shift, 0)
    value = tensor_64 << left_shift

    product = value * int(multiplier)
    result = (product + (1 << 30)) >> 31

    if right_shift:
        remainder_mask = (1 << right_shift) - 1
        remainder = torch.bitwise_and(result, remainder_mask)
        result = result >> right_shift
        threshold = remainder_mask >> 1
        threshold_tensor = torch.full_like(result, threshold, dtype=torch.int64)
        threshold_tensor = torch.where(
            result < 0, threshold_tensor + 1, threshold_tensor
        )
        result = result + torch.where(remainder > threshold_tensor, 1, 0)

    return result.to(torch.int32)


def is_channels_last(tensor: torch.Tensor) -> bool:
    """Check whether a 4D tensor uses channels-last physical storage."""
    if tensor.ndim != 4:
        return False

    # These degenerate dimensions make default and channels-last strides
    # indistinguishable; either interpretation is safe for the kernel.
    if tensor.shape[1] == 1 or tensor.shape[2] == tensor.shape[3] == 1:
        return True

    dim_order = list(tensor.dim_order())
    return dim_order[0:2] == [0, 2]


def is_default_dim_order(tensor: torch.Tensor) -> bool:
    """Check whether a tensor uses its default logical dimension order."""
    return list(tensor.dim_order()) == list(range(tensor.ndim))


def is_default_or_channels_last(tensor: torch.Tensor) -> bool:
    """Check whether a 4D tensor uses a layout understood by the backend."""
    return tensor.ndim == 4 and (
        is_default_dim_order(tensor) or is_channels_last(tensor)
    )


def is_channel_broadcast(tensor1: torch.Tensor, tensor2: torch.Tensor) -> bool:
    """Check for a channels-last broadcast where one tensor is channel-only."""
    if tensor1.dim() != tensor2.dim():
        return False
    if not is_channels_last(tensor1) or not is_channels_last(tensor2):
        return False

    channel_match = tensor1.size(1) == tensor2.size(1)
    tensor1_channels_only = tensor1.numel() == tensor1.size(1)
    tensor2_channels_only = tensor2.numel() == tensor2.size(1)
    return channel_match and (tensor1_channels_only or tensor2_channels_only)
