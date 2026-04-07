# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from math import prod
from typing import Sequence

import torch
import torch.nn.functional as F
from executorch.backends.cortex_m.float_activation_constants import (
    CMSIS_FLOAT_ACT_HARDSIGMOID,
    CMSIS_FLOAT_ACT_HARDSWISH,
    CMSIS_FLOAT_ACT_HARDTANH,
    CMSIS_FLOAT_ACT_LEAKY_RELU,
    CMSIS_FLOAT_ACT_NONE,
    CMSIS_FLOAT_ACT_RELU,
    CMSIS_FLOAT_ACT_RELU6,
    CMSIS_FLOAT_ACT_SIGMOID,
    CMSIS_FLOAT_ACT_TANH,
)
from executorch.backends.cortex_m.passes.float_capabilities import (
    get_cortex_m_float_capabilities,
)
from executorch.backends.cortex_m.passes.passes_utils import (
    dequantize_per_tensor_cmsis,
    is_channel_broadcast,
    is_channels_last,
    is_default_dim_order,
    is_default_or_channels_last,
    quantize_per_tensor_cmsis,
    requantize_cmsis,
    SHIFT_INT8,
)
from executorch.backends.cortex_m.quantizer.quantization_configs import (
    CMSIS_SOFTMAX_SCALE,
    CMSIS_SOFTMAX_ZERO_POINT,
)
from executorch.exir._warnings import experimental
from executorch.exir.dialects._ops import ops as exir_ops

# To provide the implementation of the operators
from torch.library import impl, Library, register_fake

# New operator library with a custom namespace to allow fusion etc.
lib = Library("cortex_m", "DEF")

_EXPLICIT_LAYOUT_EXPERIMENTAL = (
    "This explicit-layout Cortex-M operator may change while the legacy "
    "dim-order operators remain supported."
)

SOFTMAX_INPUT_INTEGER_BITS = 5
_FLOAT_CAPABILITIES = get_cortex_m_float_capabilities()


def _cortex_m_float_enabled(dtype: torch.dtype) -> bool:
    return _FLOAT_CAPABILITIES.is_float_dtype_enabled(dtype)


def _register_fake_if_enabled(op_name: str, dtype: torch.dtype):
    def decorator(fn):
        if _cortex_m_float_enabled(dtype):
            return register_fake(op_name)(fn)  # type: ignore[misc]
        return fn

    return decorator


def _impl_if_enabled(lib: Library, op_name: str, dispatch: str, dtype: torch.dtype):
    def decorator(fn):
        if _cortex_m_float_enabled(dtype):
            return impl(lib, op_name, dispatch)(fn)  # type: ignore[misc]
        return fn

    return decorator


###
# dequantize_per_tensor
###

lib.define(
    "quantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> (Tensor Z)"
)

lib.define(
    "quantize_per_tensor.out(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantize_per_tensor")  # type: ignore[misc]
def quantize_per_tensor_meta(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty_like(input, dtype=dtype)


@impl(lib, "quantize_per_tensor", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantize_per_tensor_impl(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    The implementation of the quantize_per_tensor operator is the same as the
    quantize_per_tensor operator in the edge dialect.
    """
    return exir_ops.edge.quantized_decomposed.quantize_per_tensor.default(
        input, scale, zero_point, quant_min, quant_max, dtype
    )


###
# dequantize_per_tensor
###

lib.define(
    "dequantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> (Tensor Z)"
)
lib.define(
    "dequantize_per_tensor.out(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::dequantize_per_tensor")  # type: ignore[misc]
def dequantize_per_tensor_meta(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty_like(input, dtype=torch.float)


@impl(lib, "dequantize_per_tensor", "CompositeExplicitAutograd")  # type: ignore[misc]
def dequantize_per_tensor_impl(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    The implementation of the dequantize_per_tensor operator is the same as the
    dequantize_per_tensor operator in the edge dialect.
    """
    return exir_ops.edge.quantized_decomposed.dequantize_per_tensor.default(
        input, scale, zero_point, quant_min, quant_max, dtype
    )


# Define the operator schema with multipliers and shifts (11 args)
lib.define(
    "quantized_add("
    "Tensor self, int self_zero_point, int self_multiplier, int self_shift, "
    "Tensor other, int other_zero_point, int other_multiplier, int other_shift, "
    "int output_zero_point, int output_multiplier, int output_shift, "
    "int activation_min, int activation_max) -> Tensor"
)

lib.define(
    "quantized_add.out("
    "Tensor self, int self_zero_point, int self_multiplier, int self_shift, "
    "Tensor other, int other_zero_point, int other_multiplier, int other_shift, "
    "int output_zero_point, int output_multiplier, int output_shift, "
    "int activation_min, int activation_max, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_add")  # type: ignore[misc]
def quantized_add_meta(
    self: torch.Tensor,
    self_zero_point: int,
    self_multiplier: int,
    self_shift: int,
    other: torch.Tensor,
    other_zero_point: int,
    other_multiplier: int,
    other_shift: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    assert self.shape == other.shape or is_channel_broadcast(self, other), (
        "Cortex-M quantized_add: broadcasting is not yet supported except for channel dim — "
        f"got self.shape={self.shape}, other.shape={other.shape}"
    )
    if self.numel() > other.numel():
        output_tensor = self
    else:
        output_tensor = other
    return torch.empty_like(output_tensor)


@impl(lib, "quantized_add", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_add_impl(
    self: torch.Tensor,
    self_zero_point: int,
    self_multiplier: int,
    self_shift: int,
    other: torch.Tensor,
    other_zero_point: int,
    other_multiplier: int,
    other_shift: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    assert self.shape == other.shape or is_channel_broadcast(self, other), (
        "Cortex-M quantized_add: broadcasting is not yet supported except for channel dim — "
        f"got self.shape={self.shape}, other.shape={other.shape}"
    )
    self_shifted = (self.to(torch.int32) - self_zero_point) << SHIFT_INT8
    self_fp = requantize_cmsis(self_shifted, self_multiplier, self_shift)

    other_shifted = (other.to(torch.int32) - other_zero_point) << SHIFT_INT8
    other_fp = requantize_cmsis(other_shifted, other_multiplier, other_shift)

    result_fp = self_fp + other_fp
    result_quantized = requantize_cmsis(result_fp, output_multiplier, output_shift)
    result = torch.clamp(
        result_quantized + output_zero_point, activation_min, activation_max
    ).to(torch.int8)
    return result


# ===================================================================
# QUANTIZED MUL OPERATION DEFINITION
# ===================================================================
lib.define(
    "quantized_mul("
    "Tensor self, int self_zero_point, "
    "Tensor other, int other_zero_point, "
    "int output_zero_point, int output_multiplier, int output_shift) -> Tensor"
)
lib.define(
    "quantized_mul.out("
    "Tensor self, int self_zero_point, "
    "Tensor other, int other_zero_point, "
    "int output_zero_point, int output_multiplier, int output_shift, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_mul")  # type: ignore[misc]
def quantized_mul_meta(
    self: torch.Tensor,
    self_zero_point: int,
    other: torch.Tensor,
    other_zero_point: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
) -> torch.Tensor:
    # Broadcast to output shape
    assert self.shape == other.shape or is_channel_broadcast(self, other), (
        "Cortex-M quantized_mul: broadcasting is not yet supported except for channel dim — "
        f"got self.shape={self.shape}, other.shape={other.shape}"
    )
    if self.numel() > other.numel():
        output_tensor = self
    else:
        output_tensor = other
    return torch.empty_like(output_tensor)


@impl(lib, "quantized_mul", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_mul_impl(
    self: torch.Tensor,
    self_zero_point: int,
    other: torch.Tensor,
    other_zero_point: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
) -> torch.Tensor:
    # CMSIS-NN kernel multiplies raw int8 tensors (after zero-point offset) and
    # only uses the output multiplier/shift for rescaling. Mirror that here to
    # keep the composite implementation numerically aligned with the backend.
    assert self.shape == other.shape or is_channel_broadcast(self, other), (
        "Cortex-M quantized_mul: broadcasting is not yet supported except for channel dim — "
        f"got self.shape={self.shape}, other.shape={other.shape}"
    )
    self_int = self.to(torch.int32) - self_zero_point
    other_int = other.to(torch.int32) - other_zero_point
    result_fp = self_int * other_int
    result_quantized = requantize_cmsis(result_fp, output_multiplier, output_shift)
    result = torch.clamp(result_quantized + output_zero_point, -128, 127).to(torch.int8)
    return result


# ===================================================================
# QUANTIZED DIV OPERATION DEFINITION
# ===================================================================
lib.define(
    "quantized_div("
    "Tensor self, int self_zero_point, "
    "Tensor other, int other_zero_point, "
    "int output_zero_point, float output_scale) -> Tensor"
)
lib.define(
    "quantized_div.out("
    "Tensor self, int self_zero_point, "
    "Tensor other, int other_zero_point, "
    "int output_zero_point, float output_scale, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_div")  # type: ignore[misc]
def quantized_div_meta(
    self: torch.Tensor,
    self_zero_point: int,
    other: torch.Tensor,
    other_zero_point: int,
    output_zero_point: int,
    output_scale: float,
) -> torch.Tensor:
    # Division is not commutative, so broadcasting (handled via operand swaps in
    # quantized_mul) is not supported: require identical shapes.
    assert self.shape == other.shape, (
        "Cortex-M quantized_div: broadcasting is not supported — "
        f"got self.shape={self.shape}, other.shape={other.shape}"
    )
    return torch.empty_like(self)


@impl(lib, "quantized_div", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_div_impl(
    self: torch.Tensor,
    self_zero_point: int,
    other: torch.Tensor,
    other_zero_point: int,
    output_zero_point: int,
    output_scale: float,
) -> torch.Tensor:
    # Mirror the kernel: the quotient of the zero-point-corrected int8/int16
    # operands is evaluated in float and rescaled by the effective scale
    # (scale_in1 / (scale_in2 * scale_out)) that the AoT pass carries directly.
    assert self.shape == other.shape, (
        "Cortex-M quantized_div: broadcasting is not supported — "
        f"got self.shape={self.shape}, other.shape={other.shape}"
    )
    if self.dtype not in (torch.int8, torch.int16):
        raise TypeError(
            f"cortex_m.quantized_div: expected int8 or int16 inputs, got {self.dtype}"
        )
    self_fp = (self.to(torch.int32) - self_zero_point).to(torch.float32)
    other_fp = (other.to(torch.int32) - other_zero_point).to(torch.float32)

    quotient = torch.where(other_fp != 0, self_fp / other_fp, torch.zeros_like(self_fp))
    result = torch.round(quotient * output_scale) + output_zero_point
    dtype_info = torch.iinfo(self.dtype)
    return torch.clamp(result, dtype_info.min, dtype_info.max).to(self.dtype)


# ===================================================================
# QUANTIZED ACTIVATION (LUT) OPERATION DEFINITION
# ===================================================================
# Generic table-lookup activation. The 256-entry int8 LUT is precomputed AoT
# from the input/output qparams and the activation function (sigmoid, tanh,
# silu, ...), so the kernel is identical regardless of which activation it
# evaluates: out[i] = lut[input[i] + 128].
lib.define("quantized_activation(Tensor input, Tensor lut) -> Tensor")
lib.define(
    "quantized_activation.out(Tensor input, Tensor lut, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_activation")  # type: ignore[misc]
def quantized_activation_meta(input: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    assert input.dtype == torch.int8, "quantized_activation input must be int8"
    assert lut.dtype == torch.int8 and lut.numel() == 256, (
        "quantized_activation lut must be int8 with 256 entries; "
        f"got dtype={lut.dtype}, numel={lut.numel()}"
    )
    return torch.empty_like(input)


@impl(lib, "quantized_activation", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_activation_impl(input: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    indices = input.to(torch.int32) + 128
    return lut[indices].to(torch.int8)


# ===================================================================
# QUANTIZED BATCH MATMUL OPERATION DEFINITION
# ===================================================================
lib.define(
    "quantized_batch_matmul("
    "Tensor lhs, int lhs_zero_point, "
    "Tensor rhs_transposed, int rhs_zero_point, "
    "int output_zero_point, int output_multiplier, int output_shift, "
    "Tensor scratch) -> Tensor"
)
lib.define(
    "quantized_batch_matmul.out("
    "Tensor lhs, int lhs_zero_point, "
    "Tensor rhs_transposed, int rhs_zero_point, "
    "int output_zero_point, int output_multiplier, int output_shift, "
    "Tensor scratch, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_batch_matmul")  # type: ignore[misc]
def quantized_batch_matmul_meta(
    lhs: torch.Tensor,
    lhs_zero_point: int,
    rhs_transposed: torch.Tensor,
    rhs_zero_point: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    batch, lhs_rows, inner = lhs.shape
    batch_rhs, rhs_cols, inner_rhs = rhs_transposed.shape
    assert batch == batch_rhs and inner == inner_rhs
    return torch.empty((batch, lhs_rows, rhs_cols), dtype=torch.int8, device=lhs.device)


@impl(lib, "quantized_batch_matmul", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_batch_matmul_impl(
    lhs: torch.Tensor,
    lhs_zero_point: int,
    rhs_transposed: torch.Tensor,
    rhs_zero_point: int,
    output_zero_point: int,
    output_multiplier: int,
    output_shift: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    # Offsets are negated zero points (CMSIS-NN convention)
    lhs_fp = lhs.to(torch.float32) + float(lhs_zero_point)
    rhs_t_fp = rhs_transposed.to(torch.float32) + float(rhs_zero_point)
    rhs_fp = rhs_t_fp.permute(0, 2, 1)
    acc = torch.bmm(lhs_fp, rhs_fp).to(torch.int32)
    result = requantize_cmsis(acc, output_multiplier, output_shift)
    return torch.clamp(result + output_zero_point, -128, 127).to(torch.int8)


# ===================================================================
# MINIMUM/MAXIMUM OPERATION DEFINITIONS
# ===================================================================
lib.define("minimum(Tensor self, Tensor other) -> Tensor")
lib.define("minimum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("cortex_m::minimum")  # type: ignore[misc]
def minimum_meta(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    # other is a scalar, so use initial shape.
    if other.numel() == 1:
        return torch.empty_like(self)
    else:
        # otherwise broadcast the shape.
        return torch.empty(
            torch.broadcast_shapes(self.shape, other.shape),
            dtype=self.dtype,
            device=self.device,
        )


@impl(lib, "minimum", "CompositeExplicitAutograd")  # type: ignore[misc]
def minimum_impl(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.minimum(self, other)


lib.define("maximum(Tensor self, Tensor other) -> Tensor")
lib.define("maximum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("cortex_m::maximum")  # type: ignore[misc]
def maximum_meta(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    assert self.dtype == other.dtype, (
        "Cortex-M maximum: dtype mismatch — "
        f"got self.dtype={self.dtype}, other.dtype={other.dtype}"
    )
    # other is a scalar, so use initial shape.
    if other.numel() == 1:
        return torch.empty_like(self)
    else:
        # otherwise broadcast the shape.
        return torch.empty(
            torch.broadcast_shapes(self.shape, other.shape),
            dtype=self.dtype,
            device=self.device,
        )


@impl(lib, "maximum", "CompositeExplicitAutograd")  # type: ignore[misc]
def maximum_impl(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.maximum(self, other)


# ===================================================================
# QUANTIZED LINEAR OPERATION DEFINITION
# ===================================================================

lib.define(
    "quantized_linear.out("
    "Tensor input,  "
    "Tensor weights, "
    "Tensor? bias, "
    "Tensor? kernel_sum, "
    "int input_offset, "
    "int filter_offset, "
    "int output_offset, "
    "int[] requantize_multipliers, "
    "int[] requantize_shifts, "
    "int activation_max, "
    "int activation_min, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)

# Define functional variant (non-out version)
lib.define(
    "quantized_linear("
    "Tensor input,  "
    "Tensor weights, "
    "Tensor? bias, "
    "Tensor? kernel_sum, "
    "int input_offset, "
    "int filter_offset, "
    "int output_offset, "
    "int[] requantize_multipliers, "
    "int[] requantize_shifts, "
    "int activation_max, "
    "int activation_min"
    ") -> Tensor"
)


# Fake meta function for shape inference (functional variant)
@register_fake("cortex_m::quantized_linear")  # type: ignore[misc]
def quantized_linear_meta(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    kernel_sum: torch.Tensor | None,
    input_offset: int,
    filter_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_max: int,
    activation_min: int,
) -> torch.Tensor:

    shape = (*input.shape[:-1], weights.shape[0])
    return torch.empty(shape, dtype=input.dtype, device=input.device)


# Functional variant implementation
@impl(lib, "quantized_linear", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_linear_impl(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    kernel_sum: torch.Tensor | None,
    input_offset: int,
    filter_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_max: int,
    activation_min: int,
) -> torch.Tensor:
    """
    Functional variant - creates output tensor and calls out variant
    """

    # Mirror CMSIS-NN's arm_fully_connected_s8 contract: the MVE path reads
    # kernel_sum (ctx.buf) and ignores bias; the DSP and scalar paths read
    # bias and ignore kernel_sum. The AOT pass populates exactly one of them
    # based on the target ISA, so dispatch off which one is present.
    if kernel_sum is not None:
        weights_int32 = weights.to(torch.int32)

        input_int32 = input.to(torch.int32)
        new_shape = (prod(input.shape[:-1]), input.shape[-1])
        input_reshaped = input_int32.reshape(new_shape)

        lhs_sum = torch.sum(input_reshaped, dim=-1, keepdim=True) * filter_offset
        output = torch.mm(input_reshaped, weights_int32.T) + lhs_sum + kernel_sum
        output_shape = (*input.shape[:-1], output.shape[-1])
        output_reshaped = output.reshape(output_shape)
    else:
        weights_int32 = weights.to(torch.int32) + filter_offset

        input_int32 = input.to(torch.int32) + input_offset
        new_shape = (prod(input.shape[:-1]), input.shape[-1])
        input_reshaped = input_int32.reshape(new_shape)

        output = torch.mm(input_reshaped, weights_int32.T)
        if bias is not None:
            output = output + bias
        output_shape = (*input.shape[:-1], output.shape[-1])
        output_reshaped = output.reshape(output_shape)

    output = requantize_cmsis(
        output_reshaped,
        int(requantize_multipliers[0]),
        int(requantize_shifts[0]),
    )
    output += output_offset
    output = torch.clamp(output, activation_min, activation_max).to(torch.int8)
    return output


# ===================================================================
# SOFTMAX OPERATION DEFINITION
# ===================================================================

lib.define(
    "softmax(Tensor input, int dim, int input_zero_point, int output_zero_point, int input_multiplier, int input_shift, int diff_min) -> Tensor"
)
lib.define(
    "softmax.out(Tensor input, int dim, int input_zero_point, int output_zero_point, int input_multiplier, int input_shift, int diff_min, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::softmax")  # type: ignore[misc]
def softmax_meta(
    input: torch.Tensor,
    dim: int,
    input_zero_point: int,
    output_zero_point: int,
    input_multiplier: int,
    input_shift: int,
    diff_min: int,
) -> torch.Tensor:
    return torch.empty_like(input, dtype=torch.int8)


@impl(lib, "softmax", "CompositeExplicitAutograd")  # type: ignore[misc]
def softmax_impl(
    input: torch.Tensor,
    dim: int,
    input_zero_point: int,
    output_zero_point: int,
    input_multiplier: int,
    input_shift: int,
    diff_min: int,
) -> torch.Tensor:
    del diff_min  # not used in reference path
    if input.dtype != torch.int8:
        raise TypeError(
            f"cortex_m.softmax: expected int8 input tensor, got {input.dtype}"
        )
    if output_zero_point != CMSIS_SOFTMAX_ZERO_POINT:
        raise ValueError(
            f"cortex_m.softmax: expected output_zero_point {CMSIS_SOFTMAX_ZERO_POINT}, got {output_zero_point}"
        )

    real_multiplier = float(input_multiplier) / float(1 << 31)
    real_multiplier = math.ldexp(real_multiplier, input_shift)
    input_scale = real_multiplier / float(1 << (31 - SOFTMAX_INPUT_INTEGER_BITS))
    if input_scale <= 0:
        raise ValueError(
            f"cortex_m.softmax: derived non-positive input scale {input_scale}"
        )

    input_fp = (input.to(torch.int32) - int(input_zero_point)).float() * input_scale
    probs = torch.softmax(input_fp, dim=dim)
    quantized = torch.round(probs / CMSIS_SOFTMAX_SCALE) + int(output_zero_point)
    return quantized.clamp(-128, 127).to(torch.int8)


# ===================================================================
# TRANSPOSE OPERATION DEFINITION
# ===================================================================
lib.define("transpose(Tensor input, int[] perm) -> Tensor")
lib.define("transpose.out(Tensor input, int[] perm, *, Tensor(a!) out) -> Tensor(a!)")


@register_fake("cortex_m::transpose")  # type: ignore[misc]
def transpose_meta(input: torch.Tensor, perm: Sequence[int]) -> torch.Tensor:
    output_shape = [input.shape[idx] for idx in perm]
    return torch.empty(output_shape, dtype=input.dtype, device=input.device)


@impl(lib, "transpose", "CompositeExplicitAutograd")  # type: ignore[misc]
def transpose_impl(input: torch.Tensor, perm: Sequence[int]) -> torch.Tensor:
    return input.permute(tuple(perm)).contiguous()


# ===================================================================
# PAD OPERATION DEFINITION
# ===================================================================
lib.define("pad(Tensor input, int[] pre_pad, int[] post_pad, int pad_value) -> Tensor")
lib.define(
    "pad.out(Tensor input, int[] pre_pad, int[] post_pad, int pad_value, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


_NHWC_INV_ORDER = [0, 3, 1, 2]


def _pad_to_logical_order(physical_pad: list[int], input: torch.Tensor) -> list[int]:
    """Inverse of _to_physical_order: map physical-order padding back to logical."""
    if not is_channels_last(input):
        return list(physical_pad)
    return [physical_pad[_NHWC_INV_ORDER[i]] for i in range(4)]


@register_fake("cortex_m::pad")  # type: ignore[misc]
def pad_meta(
    input: torch.Tensor,
    pre_pad: list[int],
    post_pad: list[int],
    pad_value: int,
) -> torch.Tensor:
    rank = input.dim()
    offset = 4 - rank
    logical_pre = _pad_to_logical_order(pre_pad, input)
    logical_post = _pad_to_logical_order(post_pad, input)

    output_shape = list(input.shape)
    for i in range(rank):
        output_shape[i] += logical_pre[offset + i] + logical_post[offset + i]
    result = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    if is_channels_last(input):
        result = result.to(memory_format=torch.channels_last)
    return result


@impl(lib, "pad", "CompositeExplicitAutograd")  # type: ignore[misc]
def pad_impl(
    input: torch.Tensor,
    pre_pad: list[int],
    post_pad: list[int],
    pad_value: int,
) -> torch.Tensor:
    rank = input.dim()
    offset = 4 - rank
    logical_pre = _pad_to_logical_order(pre_pad, input)
    logical_post = _pad_to_logical_order(post_pad, input)

    padding = []
    for i in reversed(range(rank)):
        padding.extend([logical_pre[offset + i], logical_post[offset + i]])
    return F.pad(input, padding, mode="constant", value=pad_value)


# ===================================================================
# QUANTIZED CONV2D OPERATION DEFINITION
# ===================================================================

lib.define(
    "quantized_conv2d("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] dilation, "
    "int input_offset, "
    "int output_offset, "
    "Tensor requantize_multipliers, "
    "Tensor requantize_shifts, "
    "int activation_min, "
    "int activation_max, "
    "Tensor scratch"
    ") -> Tensor"
)


lib.define(
    "quantized_conv2d.out("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] dilation, "
    "int input_offset, "
    "int output_offset, "
    "Tensor requantize_multipliers, "
    "Tensor requantize_shifts, "
    "int activation_min, "
    "int activation_max, "
    "Tensor scratch, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)


def _compute_conv2d_output_shape(
    input_shape: torch.Size,
    weight_shape: torch.Size,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> torch.Size:
    batch = input_shape[0]
    in_height = input_shape[2]
    in_width = input_shape[3]
    # We store the weights in OHWI layout (out, kernel_h, kernel_w, in)
    kernel_height = weight_shape[1]
    kernel_width = weight_shape[2]

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    out_channels = weight_shape[0]
    out_height = (
        in_height + 2 * pad_h - dilation_h * (kernel_height - 1) - 1
    ) // stride_h + 1
    out_width = (
        in_width + 2 * pad_w - dilation_w * (kernel_width - 1) - 1
    ) // stride_w + 1
    return torch.Size([batch, out_channels, out_height, out_width])


def _compute_depthwise_conv2d_output_shape(
    input_shape: torch.Size,
    weight_shape: torch.Size,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> torch.Size:
    batch = input_shape[0]
    in_height = input_shape[2]
    in_width = input_shape[3]
    # For depthwise conv, we store the weights in IHWO layout (1, kernel_h, kernel_w, out)
    # where dimension 3 contains the output channels
    kernel_height = weight_shape[1]
    kernel_width = weight_shape[2]

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    out_channels = weight_shape[3]  # IHWO format: output channels at dimension 3
    out_height = (
        in_height + 2 * pad_h - dilation_h * (kernel_height - 1) - 1
    ) // stride_h + 1
    out_width = (
        in_width + 2 * pad_w - dilation_w * (kernel_width - 1) - 1
    ) // stride_w + 1
    return torch.Size([batch, out_channels, out_height, out_width])


@register_fake("cortex_m::quantized_conv2d")  # type: ignore[misc]
def quantized_conv2d_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    stride_vals = list(stride)
    padding_vals = list(padding)
    dilation_vals = list(dilation)
    output_shape = _compute_conv2d_output_shape(
        input.shape, weight.shape, stride_vals, padding_vals, dilation_vals
    )
    return torch.empty(
        output_shape,
        dtype=torch.int8,
        device=input.device,
        memory_format=torch.channels_last,
    )


@impl(lib, "quantized_conv2d", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_conv2d_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    if input.dim() != 4 or weight.dim() != 4:
        raise RuntimeError("quantized_conv2d expects 4D input and weight tensors")
    # Convert to int32 for accumulation and apply offsets
    input_int32 = input.to(torch.int32) + int(input_offset)
    weight_int32 = weight.to(torch.int32)

    if bias is None:
        bias_int32 = torch.zeros(
            weight.shape[0], dtype=torch.int32, device=input.device
        )
    else:
        bias_int32 = bias.to(torch.int32)

    input_channels = input.shape[1]
    kernel_input_channels = weight.shape[3]
    groups = input_channels // kernel_input_channels

    # Convert weights back to OIHW layout expected by torch.nn.functional.conv2d
    weight_oi_hw = weight_int32.permute(0, 3, 1, 2).contiguous()

    conv_acc = F.conv2d(
        input_int32,
        weight_oi_hw,
        bias_int32,
        stride=tuple(stride),
        padding=tuple(padding),
        dilation=tuple(dilation),
        groups=groups,
    )

    result_channels = []
    for output_channel_i in range(conv_acc.shape[1]):
        result_channel = requantize_cmsis(
            conv_acc[:, output_channel_i, :, :],
            int(requantize_multipliers[output_channel_i]),
            int(requantize_shifts[output_channel_i]),
        )
        result_channels.append(result_channel)

    result = torch.stack(result_channels, dim=1)

    result += output_offset
    result = torch.clamp(result, activation_min, activation_max)

    # TODO - this enforces all convolution layers to result in channels last.
    # This issue does comes min/mul layers from decomposition of hard swish
    return result.to(torch.int8, memory_format=torch.channels_last)


lib.define(
    "quantized_conv2d_nhwc("
    "Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, "
    "int[] dilation, int input_offset, int output_offset, "
    "Tensor requantize_multipliers, Tensor requantize_shifts, "
    "int activation_min, int activation_max, Tensor scratch) -> Tensor"
)
lib.define(
    "quantized_conv2d_nhwc.out("
    "Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, "
    "int[] dilation, int input_offset, int output_offset, "
    "Tensor requantize_multipliers, Tensor requantize_shifts, "
    "int activation_min, int activation_max, Tensor scratch, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_conv2d_nhwc")  # type: ignore[misc]
@experimental(_EXPLICIT_LAYOUT_EXPERIMENTAL)  # type: ignore[misc]
def quantized_conv2d_nhwc_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    nchw = quantized_conv2d_meta(
        input.permute(0, 3, 1, 2),
        weight,
        bias,
        stride,
        padding,
        dilation,
        input_offset,
        output_offset,
        requantize_multipliers,
        requantize_shifts,
        activation_min,
        activation_max,
        scratch,
    )
    return nchw.permute(0, 2, 3, 1).contiguous()


# -------------------------------------------------------------------
# Float arithmetic operators
# -------------------------------------------------------------------


def _define_float_binary_op(op_name: str, dtype: torch.dtype, pt_op) -> None:
    if not _cortex_m_float_enabled(dtype):
        return

    expected_dtype_str = "float32" if dtype == torch.float32 else "float16"
    lib.define(
        f"{op_name}("
        "Tensor self, Tensor other, float activation_min, float activation_max"
        ") -> Tensor"
    )
    lib.define(
        f"{op_name}.out("
        "Tensor self, Tensor other, float activation_min, float activation_max, "
        "*, Tensor(a!) out) -> Tensor(a!)"
    )

    def _meta(
        self: torch.Tensor,
        other: torch.Tensor,
        activation_min: float,
        activation_max: float,
    ) -> torch.Tensor:
        del activation_min, activation_max
        assert self.dtype == dtype, (
            f"Cortex-M {op_name} expects {expected_dtype_str} inputs, "
            f"got self.dtype={self.dtype}"
        )
        assert other.dtype == dtype, (
            f"Cortex-M {op_name} expects {expected_dtype_str} inputs, "
            f"got other.dtype={other.dtype}"
        )
        assert self.shape == other.shape or is_channel_broadcast(self, other), (
            f"Cortex-M {op_name} currently requires same-shape tensors or "
            f"channel broadcast, got self.shape={self.shape}, "
            f"other.shape={other.shape}"
        )
        output_tensor = self if self.numel() >= other.numel() else other
        return torch.empty_like(output_tensor)

    def _impl(
        self: torch.Tensor,
        other: torch.Tensor,
        activation_min: float,
        activation_max: float,
    ) -> torch.Tensor:
        assert self.dtype == dtype, (
            f"Cortex-M {op_name} expects {expected_dtype_str} inputs, "
            f"got self.dtype={self.dtype}"
        )
        assert other.dtype == dtype, (
            f"Cortex-M {op_name} expects {expected_dtype_str} inputs, "
            f"got other.dtype={other.dtype}"
        )
        assert self.shape == other.shape or is_channel_broadcast(self, other), (
            f"Cortex-M {op_name} currently requires same-shape tensors or "
            f"channel broadcast, got self.shape={self.shape}, "
            f"other.shape={other.shape}"
        )
        return torch.clamp(pt_op(self, other), min=activation_min, max=activation_max)

    register_fake(f"cortex_m::{op_name}")(_meta)  # type: ignore[misc]
    impl(lib, op_name, "CompositeExplicitAutograd")(_impl)  # type: ignore[misc]


_define_float_binary_op("add_f32", torch.float32, lambda a, b: a + b)
_define_float_binary_op("add_f16", torch.float16, lambda a, b: a + b)
_define_float_binary_op("mul_f32", torch.float32, lambda a, b: a * b)
_define_float_binary_op("mul_f16", torch.float16, lambda a, b: a * b)


# -------------------------------------------------------------------
# Float batched matmul operators
# -------------------------------------------------------------------
# Packed RHS uses the same backend contract as the C++ kernel:
#   unpacked rhs_transposed : [B, N, K]
#   packed rhs_transposed   : flat rank-1 buffer + rhs_cols=N metadata
lib.define(
    "batch_matmul_f32(Tensor lhs, Tensor rhs_transposed, bool rhs_is_packed, int rhs_cols, float activation_min, float activation_max) -> Tensor"
)
lib.define(
    "batch_matmul_f32.out(Tensor lhs, Tensor rhs_transposed, bool rhs_is_packed, int rhs_cols, float activation_min, float activation_max, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "batch_matmul_f16(Tensor lhs, Tensor rhs_transposed, bool rhs_is_packed, int rhs_cols, float activation_min, float activation_max) -> Tensor"
)
lib.define(
    "batch_matmul_f16.out(Tensor lhs, Tensor rhs_transposed, bool rhs_is_packed, int rhs_cols, float activation_min, float activation_max, *, Tensor(a!) out) -> Tensor(a!)"
)


def _float_batch_matmul_meta(
    lhs: torch.Tensor,
    rhs_transposed: torch.Tensor,
    rhs_is_packed: bool,
    rhs_cols: int,
) -> torch.Tensor:
    assert lhs.dtype == rhs_transposed.dtype, (
        "Cortex-M float batch_matmul: lhs/rhs dtype mismatch — "
        f"got lhs.dtype={lhs.dtype}, rhs.dtype={rhs_transposed.dtype}"
    )
    assert lhs.dim() == 3, (
        "Cortex-M float batch_matmul: lhs must be rank-3 — "
        f"got lhs.dim()={lhs.dim()}"
    )
    batch, lhs_rows, inner = lhs.shape
    if rhs_is_packed:
        # Packed buffers are opaque kernel data, so the logical output column
        # count N is carried separately through rhs_cols. The K dimension is the
        # lhs inner dimension.
        assert rhs_transposed.dim() == 1, (
            "Cortex-M float batch_matmul: packed rhs_transposed must be rank-1 — "
            f"got rhs_transposed.dim()={rhs_transposed.dim()}"
        )
        assert rhs_cols > 0, "Cortex-M float batch_matmul: rhs_cols must be > 0"
    else:
        assert rhs_transposed.dim() == 3, (
            "Cortex-M float batch_matmul: rhs_transposed must be rank-3 when unpacked — "
            f"got rhs_transposed.dim()={rhs_transposed.dim()}"
        )
        rhs_batch, rhs_cols_actual, rhs_inner = rhs_transposed.shape
        assert batch == rhs_batch and inner == rhs_inner, (
            "Cortex-M float batch_matmul: shape mismatch — "
            f"lhs.shape={tuple(lhs.shape)}, rhs_transposed.shape={tuple(rhs_transposed.shape)}"
        )
        if rhs_cols > 0:
            assert rhs_cols == rhs_cols_actual, (
                "Cortex-M float batch_matmul: rhs_cols argument mismatch — "
                f"got rhs_cols={rhs_cols}, rhs_transposed.shape[1]={rhs_cols_actual}"
            )
        rhs_cols = rhs_cols_actual
    return torch.empty((batch, lhs_rows, rhs_cols), dtype=lhs.dtype, device=lhs.device)


@_register_fake_if_enabled("cortex_m::batch_matmul_f32", torch.float32)
def batch_matmul_f32_meta(
    lhs: torch.Tensor,
    rhs_transposed: torch.Tensor,
    rhs_is_packed: bool,
    rhs_cols: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    return _float_batch_matmul_meta(lhs, rhs_transposed, rhs_is_packed, rhs_cols)


@_register_fake_if_enabled("cortex_m::batch_matmul_f16", torch.float16)
def batch_matmul_f16_meta(
    lhs: torch.Tensor,
    rhs_transposed: torch.Tensor,
    rhs_is_packed: bool,
    rhs_cols: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    return _float_batch_matmul_meta(lhs, rhs_transposed, rhs_is_packed, rhs_cols)


def _float_batch_matmul_impl(
    lhs: torch.Tensor,
    rhs_transposed: torch.Tensor,
    rhs_is_packed: bool,
    rhs_cols: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    if rhs_is_packed:
        # Python reference path for offline-packed RHS.
        #
        # Logical BMM contract:
        #
        #   lhs            [B, M, K]
        #   rhs_transposed [B, N, K]
        #   out            [B, M, N]
        #
        # Packed storage groups N rows into CMSIS-NN lane blocks:
        #
        #   packed[batch, n_block, k, lane] = rhs_transposed[batch, n, k]
        #
        # where n = n_block * block_cols + lane and N is padded up to
        # block_cols. The public schema only carries the flat packed tensor,
        # so rhs_cols is the original, unpadded N used to crop the unpacked
        # reference tensor back to logical shape.
        block_cols = 8 if lhs.dtype == torch.float16 else 4
        batch = lhs.shape[0]
        inner = lhs.shape[2]
        packed_blocks = (rhs_cols + block_cols - 1) // block_cols
        packed_4d = rhs_transposed.reshape(batch, packed_blocks, inner, block_cols)
        unpacked = torch.zeros(
            (batch, packed_blocks * block_cols, inner),
            dtype=lhs.dtype,
            device=lhs.device,
        )
        for batch_idx in range(batch):
            for out_row in range(packed_blocks * block_cols):
                block = out_row // block_cols
                lane = out_row % block_cols
                unpacked[batch_idx, out_row].copy_(packed_4d[batch_idx, block, :, lane])
        rhs_transposed = unpacked[:, :rhs_cols, :]
    else:
        del rhs_cols
    rhs = rhs_transposed.permute(0, 2, 1)
    result = torch.bmm(lhs, rhs)
    return torch.clamp(result, min=activation_min, max=activation_max)


@_impl_if_enabled(lib, "batch_matmul_f32", "CompositeExplicitAutograd", torch.float32)
def batch_matmul_f32_impl(
    lhs: torch.Tensor,
    rhs_transposed: torch.Tensor,
    rhs_is_packed: bool,
    rhs_cols: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    return _float_batch_matmul_impl(
        lhs, rhs_transposed, rhs_is_packed, rhs_cols, activation_min, activation_max
    )


@_impl_if_enabled(lib, "batch_matmul_f16", "CompositeExplicitAutograd", torch.float16)
def batch_matmul_f16_impl(
    lhs: torch.Tensor,
    rhs_transposed: torch.Tensor,
    rhs_is_packed: bool,
    rhs_cols: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    return _float_batch_matmul_impl(
        lhs, rhs_transposed, rhs_is_packed, rhs_cols, activation_min, activation_max
    )


lib.define(
    "linear_f32.out("
    "Tensor input, "
    "Tensor weights, "
    "Tensor? bias, "
    "bool weight_is_packed, "
    "int packed_out_features, "
    "float activation_min, "
    "float activation_max, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)
lib.define(
    "linear_f32("
    "Tensor input, "
    "Tensor weights, "
    "Tensor? bias, "
    "bool weight_is_packed, "
    "int packed_out_features, "
    "float activation_min, "
    "float activation_max"
    ") -> Tensor"
)

lib.define(
    "linear_f16.out("
    "Tensor input, "
    "Tensor weights, "
    "Tensor? bias, "
    "bool weight_is_packed, "
    "int packed_out_features, "
    "float activation_min, "
    "float activation_max, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)
lib.define(
    "linear_f16("
    "Tensor input, "
    "Tensor weights, "
    "Tensor? bias, "
    "bool weight_is_packed, "
    "int packed_out_features, "
    "float activation_min, "
    "float activation_max"
    ") -> Tensor"
)


def _float_linear_meta_impl(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    weight_is_packed: bool,
    packed_out_features: int,
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    # Standard linear weights are rank-2 [O, I]. Offline-packed weights become
    # an opaque rank-1 buffer, so packed_out_features carries the logical O.
    assert (
        input.dtype == expected_dtype
    ), f"Cortex-M float linear expects input dtype={expected_dtype}, got {input.dtype}"
    assert (
        weights.dtype == expected_dtype
    ), f"Cortex-M float linear expects weights dtype={expected_dtype}, got {weights.dtype}"
    assert input.dim() >= 1, "Cortex-M float linear expects input rank >= 1"
    if weight_is_packed:
        assert weights.dim() == 1, (
            "Cortex-M float linear expects rank-1 packed weights when "
            "weight_is_packed=True"
        )
    else:
        assert weights.dim() == 2, "Cortex-M float linear expects rank-2 weights"
    out_features = packed_out_features if weight_is_packed else weights.shape[0]
    if weight_is_packed:
        assert out_features > 0, (
            "Cortex-M float linear expects packed_out_features > 0 when "
            "weight_is_packed=True"
        )
    block_cols = 8 if expected_dtype == torch.float16 else 4
    logical_in_features = (
        weights.numel()
        // (((out_features + block_cols - 1) // block_cols) * block_cols)
        if weight_is_packed
        else int(weights.shape[1])
    )
    direct_nhwc_input_features = None
    nhwc_input_to_matrix_output = False
    if input.dim() == 4 and is_channels_last(input):
        direct_nhwc_input_features = input.shape[1] * input.shape[2] * input.shape[3]
        nhwc_input_to_matrix_output = direct_nhwc_input_features == logical_in_features
    if nhwc_input_to_matrix_output:
        input_features = direct_nhwc_input_features
        assert input_features is not None
        assert input_features == logical_in_features, (
            "Cortex-M float linear expects channels-last 4D input with "
            "C*H*W == logical in_features, "
            f"got {tuple(input.shape)} vs logical_in_features={logical_in_features}"
        )
    else:
        input_last_dim = input.shape[-1]
        if not weight_is_packed and input_last_dim == 0 and weights.shape[1] > 0:
            # Some export paths can transiently report a flattened size of 0
            # even though the stabilized edge graph carries the correct shape.
            # Use the weight metadata as the source of truth for this
            # export-time case.
            input_last_dim = weights.shape[1]
        assert input_last_dim == logical_in_features, (
            "Cortex-M float linear expects input.shape[-1] == logical in_features, "
            f"got {input.shape[-1]} vs {logical_in_features}"
        )
    if bias is not None:
        assert (
            bias.dtype == expected_dtype
        ), f"Cortex-M float linear expects bias dtype={expected_dtype}, got {bias.dtype}"
        assert bias.dim() == 1 and bias.shape[0] == out_features, (
            "Cortex-M float linear expects bias shape [out_features], "
            f"got bias.shape={tuple(bias.shape)}, out_features={out_features}"
        )
    output_shape = (
        (input.shape[0], out_features)
        if nhwc_input_to_matrix_output
        else (*input.shape[:-1], out_features)
    )
    return torch.empty(output_shape, dtype=expected_dtype, device=input.device)


def _float_linear_impl(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    weight_is_packed: bool,
    packed_out_features: int,
    activation_min: float,
    activation_max: float,
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    _float_linear_meta_impl(
        input, weights, bias, weight_is_packed, packed_out_features, expected_dtype
    )
    if weight_is_packed:
        block_cols = 8 if expected_dtype == torch.float16 else 4
        in_features = weights.numel() // (
            ((packed_out_features + block_cols - 1) // block_cols) * block_cols
        )
        packed_blocks = (packed_out_features + block_cols - 1) // block_cols
        packed_3d = weights.reshape(packed_blocks, in_features, block_cols)
        unpacked = torch.zeros(
            (packed_out_features, in_features),
            dtype=expected_dtype,
            device=input.device,
        )
        for out_feature in range(packed_out_features):
            block = out_feature // block_cols
            lane = out_feature % block_cols
            unpacked[out_feature].copy_(packed_3d[block, :, lane])
        weights = unpacked
    nhwc_input_to_matrix_output = (
        input.dim() == 4
        and is_channels_last(input)
        and input.shape[1] * input.shape[2] * input.shape[3] == weights.shape[1]
    )
    if nhwc_input_to_matrix_output:
        input = input.permute(0, 2, 3, 1).reshape(input.shape[0], -1)
    out = F.linear(input, weights, bias)
    return torch.clamp(out, min=activation_min, max=activation_max)


@_register_fake_if_enabled("cortex_m::linear_f32", torch.float32)
def linear_f32_meta(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    weight_is_packed: bool,
    packed_out_features: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    del activation_min, activation_max
    return _float_linear_meta_impl(
        input, weights, bias, weight_is_packed, packed_out_features, torch.float32
    )


@_impl_if_enabled(lib, "linear_f32", "CompositeExplicitAutograd", torch.float32)
def linear_f32_impl(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    weight_is_packed: bool,
    packed_out_features: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    return _float_linear_impl(
        input,
        weights,
        bias,
        weight_is_packed,
        packed_out_features,
        activation_min,
        activation_max,
        torch.float32,
    )


@_register_fake_if_enabled("cortex_m::linear_f16", torch.float16)
def linear_f16_meta(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    weight_is_packed: bool,
    packed_out_features: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    del activation_min, activation_max
    return _float_linear_meta_impl(
        input, weights, bias, weight_is_packed, packed_out_features, torch.float16
    )


@_impl_if_enabled(lib, "linear_f16", "CompositeExplicitAutograd", torch.float16)
def linear_f16_impl(
    input: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    weight_is_packed: bool,
    packed_out_features: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    return _float_linear_impl(
        input,
        weights,
        bias,
        weight_is_packed,
        packed_out_features,
        activation_min,
        activation_max,
        torch.float16,
    )


lib.define(
    "lstm_unidirectional_f32.out("
    "Tensor input, "
    "Tensor forget_input_weights, Tensor forget_hidden_weights, Tensor forget_bias, "
    "Tensor input_input_weights, Tensor input_hidden_weights, Tensor input_bias, "
    "Tensor cell_input_weights, Tensor cell_hidden_weights, Tensor cell_bias, "
    "Tensor output_input_weights, Tensor output_hidden_weights, Tensor output_bias, "
    "bool time_major, float cell_clip, *, Tensor(a!) out"
    ") -> Tensor(a!)"
)
lib.define(
    "lstm_unidirectional_f32("
    "Tensor input, "
    "Tensor forget_input_weights, Tensor forget_hidden_weights, Tensor forget_bias, "
    "Tensor input_input_weights, Tensor input_hidden_weights, Tensor input_bias, "
    "Tensor cell_input_weights, Tensor cell_hidden_weights, Tensor cell_bias, "
    "Tensor output_input_weights, Tensor output_hidden_weights, Tensor output_bias, "
    "bool time_major, float cell_clip"
    ") -> Tensor"
)

lib.define(
    "lstm_unidirectional_f16.out("
    "Tensor input, "
    "Tensor forget_input_weights, Tensor forget_hidden_weights, Tensor forget_bias, "
    "Tensor input_input_weights, Tensor input_hidden_weights, Tensor input_bias, "
    "Tensor cell_input_weights, Tensor cell_hidden_weights, Tensor cell_bias, "
    "Tensor output_input_weights, Tensor output_hidden_weights, Tensor output_bias, "
    "bool time_major, float cell_clip, *, Tensor(a!) out"
    ") -> Tensor(a!)"
)
lib.define(
    "lstm_unidirectional_f16("
    "Tensor input, "
    "Tensor forget_input_weights, Tensor forget_hidden_weights, Tensor forget_bias, "
    "Tensor input_input_weights, Tensor input_hidden_weights, Tensor input_bias, "
    "Tensor cell_input_weights, Tensor cell_hidden_weights, Tensor cell_bias, "
    "Tensor output_input_weights, Tensor output_hidden_weights, Tensor output_bias, "
    "bool time_major, float cell_clip"
    ") -> Tensor"
)


def _float_lstm_output_shape(
    input: torch.Tensor, hidden_size: int, time_major: bool
) -> tuple[int, ...]:
    if time_major:
        time_steps, batch_size, _ = input.shape
        return (time_steps, batch_size, hidden_size)
    batch_size, time_steps, _ = input.shape
    return (batch_size, time_steps, hidden_size)


def _validate_float_lstm_gate(
    gate_name: str,
    input_weights: torch.Tensor,
    hidden_weights: torch.Tensor,
    bias: torch.Tensor,
    expected_dtype: torch.dtype,
    input_size: int,
    hidden_size: int,
) -> None:
    assert (
        input_weights.dtype == expected_dtype
    ), f"{gate_name} input weights must use dtype={expected_dtype}, got {input_weights.dtype}"
    assert (
        hidden_weights.dtype == expected_dtype
    ), f"{gate_name} hidden weights must use dtype={expected_dtype}, got {hidden_weights.dtype}"
    assert (
        bias.dtype == expected_dtype
    ), f"{gate_name} bias must use dtype={expected_dtype}, got {bias.dtype}"
    assert input_weights.shape == (hidden_size, input_size), (
        f"{gate_name} input weights must have shape ({hidden_size}, {input_size}), "
        f"got {tuple(input_weights.shape)}"
    )
    assert hidden_weights.shape == (hidden_size, hidden_size), (
        f"{gate_name} hidden weights must have shape ({hidden_size}, {hidden_size}), "
        f"got {tuple(hidden_weights.shape)}"
    )
    assert bias.shape == (
        hidden_size,
    ), f"{gate_name} bias must have shape ({hidden_size},), got {tuple(bias.shape)}"


def _float_lstm_meta_impl(
    input: torch.Tensor,
    forget_input_weights: torch.Tensor,
    forget_hidden_weights: torch.Tensor,
    forget_bias: torch.Tensor,
    input_input_weights: torch.Tensor,
    input_hidden_weights: torch.Tensor,
    input_bias: torch.Tensor,
    cell_input_weights: torch.Tensor,
    cell_hidden_weights: torch.Tensor,
    cell_bias: torch.Tensor,
    output_input_weights: torch.Tensor,
    output_hidden_weights: torch.Tensor,
    output_bias: torch.Tensor,
    time_major: bool,
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    assert (
        input.dtype == expected_dtype
    ), f"Cortex-M float LSTM expects input dtype={expected_dtype}, got {input.dtype}"
    assert (
        input.dim() == 3
    ), f"Cortex-M float LSTM expects rank-3 input, got rank {input.dim()}"
    input_size = input.shape[-1]
    hidden_size = forget_bias.shape[0]
    _validate_float_lstm_gate(
        "forget_gate",
        forget_input_weights,
        forget_hidden_weights,
        forget_bias,
        expected_dtype,
        input_size,
        hidden_size,
    )
    _validate_float_lstm_gate(
        "input_gate",
        input_input_weights,
        input_hidden_weights,
        input_bias,
        expected_dtype,
        input_size,
        hidden_size,
    )
    _validate_float_lstm_gate(
        "cell_gate",
        cell_input_weights,
        cell_hidden_weights,
        cell_bias,
        expected_dtype,
        input_size,
        hidden_size,
    )
    _validate_float_lstm_gate(
        "output_gate",
        output_input_weights,
        output_hidden_weights,
        output_bias,
        expected_dtype,
        input_size,
        hidden_size,
    )
    return torch.empty(
        _float_lstm_output_shape(input, hidden_size, time_major),
        dtype=expected_dtype,
        device=input.device,
    )


def _float_lstm_impl(
    input: torch.Tensor,
    forget_input_weights: torch.Tensor,
    forget_hidden_weights: torch.Tensor,
    forget_bias: torch.Tensor,
    input_input_weights: torch.Tensor,
    input_hidden_weights: torch.Tensor,
    input_bias: torch.Tensor,
    cell_input_weights: torch.Tensor,
    cell_hidden_weights: torch.Tensor,
    cell_bias: torch.Tensor,
    output_input_weights: torch.Tensor,
    output_hidden_weights: torch.Tensor,
    output_bias: torch.Tensor,
    time_major: bool,
    cell_clip: float,
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    _float_lstm_meta_impl(
        input,
        forget_input_weights,
        forget_hidden_weights,
        forget_bias,
        input_input_weights,
        input_hidden_weights,
        input_bias,
        cell_input_weights,
        cell_hidden_weights,
        cell_bias,
        output_input_weights,
        output_hidden_weights,
        output_bias,
        time_major,
        expected_dtype,
    )

    sequence = input if time_major else input.transpose(0, 1)
    time_steps, batch_size, _ = sequence.shape
    hidden_size = forget_bias.shape[0]
    h = torch.zeros(
        (batch_size, hidden_size), dtype=expected_dtype, device=input.device
    )
    c = torch.zeros(
        (batch_size, hidden_size), dtype=expected_dtype, device=input.device
    )
    outputs = []

    def gate(
        x_t: torch.Tensor,
        input_w: torch.Tensor,
        hidden_w: torch.Tensor,
        bias: torch.Tensor,
        activation: str,
    ) -> torch.Tensor:
        logits = F.linear(x_t, input_w) + F.linear(h, hidden_w) + bias
        if activation == "sigmoid":
            return torch.sigmoid(logits)
        if activation == "tanh":
            return torch.tanh(logits)
        raise AssertionError(f"Unsupported LSTM gate activation {activation}")

    for t in range(time_steps):
        x_t = sequence[t]
        f = gate(
            x_t,
            forget_input_weights,
            forget_hidden_weights,
            forget_bias,
            "sigmoid",
        )
        i = gate(
            x_t,
            input_input_weights,
            input_hidden_weights,
            input_bias,
            "sigmoid",
        )
        g = gate(
            x_t,
            cell_input_weights,
            cell_hidden_weights,
            cell_bias,
            "tanh",
        )
        o = gate(
            x_t,
            output_input_weights,
            output_hidden_weights,
            output_bias,
            "sigmoid",
        )
        c = f * c + i * g
        if cell_clip > 0:
            c = torch.clamp(c, min=-cell_clip, max=cell_clip)
        h = o * torch.tanh(c)
        outputs.append(h)

    output = torch.stack(outputs, dim=0)
    return output if time_major else output.transpose(0, 1)


@_register_fake_if_enabled("cortex_m::lstm_unidirectional_f32", torch.float32)
def lstm_unidirectional_f32_meta(
    input: torch.Tensor,
    forget_input_weights: torch.Tensor,
    forget_hidden_weights: torch.Tensor,
    forget_bias: torch.Tensor,
    input_input_weights: torch.Tensor,
    input_hidden_weights: torch.Tensor,
    input_bias: torch.Tensor,
    cell_input_weights: torch.Tensor,
    cell_hidden_weights: torch.Tensor,
    cell_bias: torch.Tensor,
    output_input_weights: torch.Tensor,
    output_hidden_weights: torch.Tensor,
    output_bias: torch.Tensor,
    time_major: bool,
    cell_clip: float,
) -> torch.Tensor:
    del cell_clip
    return _float_lstm_meta_impl(
        input,
        forget_input_weights,
        forget_hidden_weights,
        forget_bias,
        input_input_weights,
        input_hidden_weights,
        input_bias,
        cell_input_weights,
        cell_hidden_weights,
        cell_bias,
        output_input_weights,
        output_hidden_weights,
        output_bias,
        time_major,
        torch.float32,
    )


@_impl_if_enabled(
    lib, "lstm_unidirectional_f32", "CompositeExplicitAutograd", torch.float32
)
def lstm_unidirectional_f32_impl(
    input: torch.Tensor,
    forget_input_weights: torch.Tensor,
    forget_hidden_weights: torch.Tensor,
    forget_bias: torch.Tensor,
    input_input_weights: torch.Tensor,
    input_hidden_weights: torch.Tensor,
    input_bias: torch.Tensor,
    cell_input_weights: torch.Tensor,
    cell_hidden_weights: torch.Tensor,
    cell_bias: torch.Tensor,
    output_input_weights: torch.Tensor,
    output_hidden_weights: torch.Tensor,
    output_bias: torch.Tensor,
    time_major: bool,
    cell_clip: float,
) -> torch.Tensor:
    return _float_lstm_impl(
        input,
        forget_input_weights,
        forget_hidden_weights,
        forget_bias,
        input_input_weights,
        input_hidden_weights,
        input_bias,
        cell_input_weights,
        cell_hidden_weights,
        cell_bias,
        output_input_weights,
        output_hidden_weights,
        output_bias,
        time_major,
        cell_clip,
        torch.float32,
    )


@_register_fake_if_enabled("cortex_m::lstm_unidirectional_f16", torch.float16)
def lstm_unidirectional_f16_meta(
    input: torch.Tensor,
    forget_input_weights: torch.Tensor,
    forget_hidden_weights: torch.Tensor,
    forget_bias: torch.Tensor,
    input_input_weights: torch.Tensor,
    input_hidden_weights: torch.Tensor,
    input_bias: torch.Tensor,
    cell_input_weights: torch.Tensor,
    cell_hidden_weights: torch.Tensor,
    cell_bias: torch.Tensor,
    output_input_weights: torch.Tensor,
    output_hidden_weights: torch.Tensor,
    output_bias: torch.Tensor,
    time_major: bool,
    cell_clip: float,
) -> torch.Tensor:
    del cell_clip
    return _float_lstm_meta_impl(
        input,
        forget_input_weights,
        forget_hidden_weights,
        forget_bias,
        input_input_weights,
        input_hidden_weights,
        input_bias,
        cell_input_weights,
        cell_hidden_weights,
        cell_bias,
        output_input_weights,
        output_hidden_weights,
        output_bias,
        time_major,
        torch.float16,
    )


@_impl_if_enabled(
    lib, "lstm_unidirectional_f16", "CompositeExplicitAutograd", torch.float16
)
def lstm_unidirectional_f16_impl(
    input: torch.Tensor,
    forget_input_weights: torch.Tensor,
    forget_hidden_weights: torch.Tensor,
    forget_bias: torch.Tensor,
    input_input_weights: torch.Tensor,
    input_hidden_weights: torch.Tensor,
    input_bias: torch.Tensor,
    cell_input_weights: torch.Tensor,
    cell_hidden_weights: torch.Tensor,
    cell_bias: torch.Tensor,
    output_input_weights: torch.Tensor,
    output_hidden_weights: torch.Tensor,
    output_bias: torch.Tensor,
    time_major: bool,
    cell_clip: float,
) -> torch.Tensor:
    return _float_lstm_impl(
        input,
        forget_input_weights,
        forget_hidden_weights,
        forget_bias,
        input_input_weights,
        input_hidden_weights,
        input_bias,
        cell_input_weights,
        cell_hidden_weights,
        cell_bias,
        output_input_weights,
        output_hidden_weights,
        output_bias,
        time_major,
        cell_clip,
        torch.float16,
    )


# ===================================================================
# SOFTMAX OPERATION DEFINITION
# ===================================================================

lib.define("softmax_f32(Tensor input, int dim) -> Tensor")
lib.define("softmax_f32.out(Tensor input, int dim, *, Tensor(a!) out) -> Tensor(a!)")


@_register_fake_if_enabled("cortex_m::softmax_f32", torch.float32)
def softmax_f32_meta(input: torch.Tensor, dim: int) -> torch.Tensor:
    del dim
    return torch.empty_like(input, dtype=torch.float32)


@_impl_if_enabled(lib, "softmax_f32", "CompositeExplicitAutograd", torch.float32)
def softmax_f32_impl(input: torch.Tensor, dim: int) -> torch.Tensor:
    if input.dtype != torch.float32:
        raise TypeError(
            f"cortex_m.softmax_f32: expected float32 input tensor, got {input.dtype}"
        )
    return torch.softmax(input, dim=dim)


lib.define("softmax_f16(Tensor input, int dim) -> Tensor")
lib.define("softmax_f16.out(Tensor input, int dim, *, Tensor(a!) out) -> Tensor(a!)")
lib.define(
    "activation_f32(Tensor input, int activation_type, float act_param) -> Tensor"
)
lib.define(
    "activation_f32.out(Tensor input, int activation_type, float act_param, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "activation_f16(Tensor input, int activation_type, float act_param) -> Tensor"
)
lib.define(
    "activation_f16.out(Tensor input, int activation_type, float act_param, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)
lib.define("batch_norm_f32(Tensor input, Tensor scale, Tensor bias) -> Tensor")
lib.define(
    "batch_norm_f32.out(Tensor input, Tensor scale, Tensor bias, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)
lib.define("batch_norm_f16(Tensor input, Tensor scale, Tensor bias) -> Tensor")
lib.define(
    "batch_norm_f16.out(Tensor input, Tensor scale, Tensor bias, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "batch_norm_native_f32("
    "Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, float eps"
    ") -> Tensor"
)
lib.define(
    "batch_norm_native_f32.out("
    "Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, float eps, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)
lib.define(
    "batch_norm_native_f16("
    "Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, float eps"
    ") -> Tensor"
)
lib.define(
    "batch_norm_native_f16.out("
    "Tensor input, Tensor weight, Tensor bias, Tensor running_mean, Tensor running_var, float eps, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)
lib.define(
    "svdf_f32("
    "Tensor input, Tensor initial_state, Tensor weights_feature, Tensor weights_time, "
    "Tensor bias, bool time_major, int rank, "
    "float input_activation_min, float input_activation_max, "
    "float output_activation_min, float output_activation_max"
    ") -> Tensor"
)
lib.define(
    "svdf_f32.out("
    "Tensor input, Tensor initial_state, Tensor weights_feature, Tensor weights_time, "
    "Tensor bias, bool time_major, int rank, "
    "float input_activation_min, float input_activation_max, "
    "float output_activation_min, float output_activation_max, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)
lib.define(
    "svdf_f16("
    "Tensor input, Tensor initial_state, Tensor weights_feature, Tensor weights_time, "
    "Tensor bias, bool time_major, int rank, "
    "float input_activation_min, float input_activation_max, "
    "float output_activation_min, float output_activation_max"
    ") -> Tensor"
)
lib.define(
    "svdf_f16.out("
    "Tensor input, Tensor initial_state, Tensor weights_feature, Tensor weights_time, "
    "Tensor bias, bool time_major, int rank, "
    "float input_activation_min, float input_activation_max, "
    "float output_activation_min, float output_activation_max, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)


@_register_fake_if_enabled("cortex_m::softmax_f16", torch.float16)
def softmax_f16_meta(input: torch.Tensor, dim: int) -> torch.Tensor:
    del dim
    return torch.empty_like(input, dtype=torch.float16)


@_impl_if_enabled(lib, "softmax_f16", "CompositeExplicitAutograd", torch.float16)
def softmax_f16_impl(input: torch.Tensor, dim: int) -> torch.Tensor:
    if input.dtype != torch.float16:
        raise TypeError(
            f"cortex_m.softmax_f16: expected float16 input tensor, got {input.dtype}"
        )
    return torch.softmax(input, dim=dim)


def _activation_meta_float(input: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if input.ndim == 4 and input.is_contiguous(memory_format=torch.channels_last):
        return torch.empty(
            input.shape,
            dtype=dtype,
            device=input.device,
            memory_format=torch.channels_last,
        )
    return torch.empty_like(input, dtype=dtype)


def _activation_impl_float(
    input: torch.Tensor, activation_type: int, act_param: float, dtype: torch.dtype
) -> torch.Tensor:
    if input.dtype != dtype:
        raise TypeError(
            f"cortex_m.activation: expected {dtype} input tensor, got {input.dtype}"
        )

    x = input
    channels_last_4d = x.ndim == 4 and x.is_contiguous(
        memory_format=torch.channels_last
    )
    if activation_type == CMSIS_FLOAT_ACT_NONE:
        out = x.clone()
        return (
            out.contiguous(memory_format=torch.channels_last)
            if channels_last_4d
            else out
        )
    if activation_type == CMSIS_FLOAT_ACT_SIGMOID:
        out = torch.sigmoid(x)
        return (
            out.contiguous(memory_format=torch.channels_last)
            if channels_last_4d
            else out
        )
    if activation_type == CMSIS_FLOAT_ACT_TANH:
        out = torch.tanh(x)
        return (
            out.contiguous(memory_format=torch.channels_last)
            if channels_last_4d
            else out
        )
    if activation_type == CMSIS_FLOAT_ACT_RELU:
        out = torch.relu(x)
        return (
            out.contiguous(memory_format=torch.channels_last)
            if channels_last_4d
            else out
        )
    if activation_type == CMSIS_FLOAT_ACT_RELU6:
        out = torch.clamp(x, min=0.0, max=6.0)
        return (
            out.contiguous(memory_format=torch.channels_last)
            if channels_last_4d
            else out
        )
    if activation_type == CMSIS_FLOAT_ACT_HARDSWISH:
        out = torch.nn.functional.hardswish(x)
        return (
            out.contiguous(memory_format=torch.channels_last)
            if channels_last_4d
            else out
        )
    if activation_type == CMSIS_FLOAT_ACT_LEAKY_RELU:
        out = torch.nn.functional.leaky_relu(x, negative_slope=float(act_param))
        return (
            out.contiguous(memory_format=torch.channels_last)
            if channels_last_4d
            else out
        )
    if activation_type == CMSIS_FLOAT_ACT_HARDSIGMOID:
        out = torch.nn.functional.hardsigmoid(x)
        return (
            out.contiguous(memory_format=torch.channels_last)
            if channels_last_4d
            else out
        )
    if activation_type == CMSIS_FLOAT_ACT_HARDTANH:
        out = torch.clamp(x, min=-1.0, max=1.0)
        return (
            out.contiguous(memory_format=torch.channels_last)
            if channels_last_4d
            else out
        )
    raise ValueError(
        f"cortex_m.activation: unsupported activation_type {activation_type}"
    )


@_register_fake_if_enabled("cortex_m::activation_f32", torch.float32)
def activation_f32_meta(
    input: torch.Tensor, activation_type: int, act_param: float
) -> torch.Tensor:
    del activation_type, act_param
    return _activation_meta_float(input, torch.float32)


@_impl_if_enabled(lib, "activation_f32", "CompositeExplicitAutograd", torch.float32)
def activation_f32_impl(
    input: torch.Tensor, activation_type: int, act_param: float
) -> torch.Tensor:
    return _activation_impl_float(input, activation_type, act_param, torch.float32)


@_register_fake_if_enabled("cortex_m::activation_f16", torch.float16)
def activation_f16_meta(
    input: torch.Tensor, activation_type: int, act_param: float
) -> torch.Tensor:
    del activation_type, act_param
    return _activation_meta_float(input, torch.float16)


@_impl_if_enabled(lib, "activation_f16", "CompositeExplicitAutograd", torch.float16)
def activation_f16_impl(
    input: torch.Tensor, activation_type: int, act_param: float
) -> torch.Tensor:
    return _activation_impl_float(input, activation_type, act_param, torch.float16)


def _batch_norm_meta_float(
    input: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    assert (
        input.dtype == dtype
    ), f"cortex_m.batch_norm: expected {dtype} input tensor, got {input.dtype}"
    assert (
        scale.dtype == dtype and bias.dtype == dtype
    ), f"cortex_m.batch_norm: expected {dtype} scale/bias tensors"
    assert (
        input.dim() == 4
    ), f"cortex_m.batch_norm: expected rank-4 input, got rank {input.dim()}"
    assert is_default_or_channels_last(
        input
    ), "cortex_m.batch_norm: input must use default or channels_last memory format"
    channels = input.shape[1]
    assert scale.shape == (
        channels,
    ), f"cortex_m.batch_norm: scale must have shape ({channels},), got {tuple(scale.shape)}"
    assert bias.shape == (
        channels,
    ), f"cortex_m.batch_norm: bias must have shape ({channels},), got {tuple(bias.shape)}"
    return torch.empty_like(input, dtype=dtype)


def _batch_norm_impl_float(
    input: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    _batch_norm_meta_float(input, scale, bias, dtype)
    scale_view = scale.view(1, -1, 1, 1)
    bias_view = bias.view(1, -1, 1, 1)
    # Preserve the incoming layout family so the Python fallback matches the
    # C++ kernel contract: channels_last stays channels_last, default stays default.
    output = input * scale_view + bias_view
    if is_channels_last(input):
        return output.contiguous(memory_format=torch.channels_last)
    if is_default_dim_order(input):
        return output.contiguous()
    return output


@_register_fake_if_enabled("cortex_m::batch_norm_f32", torch.float32)
def batch_norm_f32_meta(
    input: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return _batch_norm_meta_float(input, scale, bias, torch.float32)


@_impl_if_enabled(lib, "batch_norm_f32", "CompositeExplicitAutograd", torch.float32)
def batch_norm_f32_impl(
    input: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return _batch_norm_impl_float(input, scale, bias, torch.float32)


@_register_fake_if_enabled("cortex_m::batch_norm_f16", torch.float16)
def batch_norm_f16_meta(
    input: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return _batch_norm_meta_float(input, scale, bias, torch.float16)


@_impl_if_enabled(lib, "batch_norm_f16", "CompositeExplicitAutograd", torch.float16)
def batch_norm_f16_impl(
    input: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return _batch_norm_impl_float(input, scale, bias, torch.float16)


def _batch_norm_native_meta_float(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    assert (
        input.dtype == dtype
    ), f"cortex_m.batch_norm_native: expected {dtype} input tensor, got {input.dtype}"
    assert input.dim() in (
        2,
        4,
    ), f"cortex_m.batch_norm_native: expected rank-2 or rank-4 input, got rank {input.dim()}"
    if input.dim() == 4:
        assert is_default_or_channels_last(
            input
        ), "cortex_m.batch_norm_native: input must use default or channels_last memory format"
    channels = input.shape[1]
    expected = (channels,)
    for name, tensor in (
        ("weight", weight),
        ("bias", bias),
        ("running_mean", running_mean),
        ("running_var", running_var),
    ):
        assert (
            tensor.dtype == dtype
        ), f"cortex_m.batch_norm_native: expected {dtype} {name} tensor, got {tensor.dtype}"
        assert (
            tensor.shape == expected
        ), f"cortex_m.batch_norm_native: {name} must have shape {expected}, got {tuple(tensor.shape)}"
    return torch.empty_like(input, dtype=dtype)


def _batch_norm_native_impl_float(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    _batch_norm_native_meta_float(input, weight, bias, running_mean, running_var, dtype)
    output = torch.nn.functional.batch_norm(
        input,
        running_mean,
        running_var,
        weight,
        bias,
        training=False,
        momentum=0.0,
        eps=float(eps),
    )
    # As above, avoid forcing an unconditional channels-last materialization.
    if is_channels_last(input):
        return output.contiguous(memory_format=torch.channels_last)
    if is_default_dim_order(input):
        return output.contiguous()
    return output


@_register_fake_if_enabled("cortex_m::batch_norm_native_f32", torch.float32)
def batch_norm_native_f32_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del eps
    return _batch_norm_native_meta_float(
        input, weight, bias, running_mean, running_var, torch.float32
    )


@_impl_if_enabled(
    lib, "batch_norm_native_f32", "CompositeExplicitAutograd", torch.float32
)
def batch_norm_native_f32_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _batch_norm_native_impl_float(
        input, weight, bias, running_mean, running_var, eps, torch.float32
    )


@_register_fake_if_enabled("cortex_m::batch_norm_native_f16", torch.float16)
def batch_norm_native_f16_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del eps
    return _batch_norm_native_meta_float(
        input, weight, bias, running_mean, running_var, torch.float16
    )


@_impl_if_enabled(
    lib, "batch_norm_native_f16", "CompositeExplicitAutograd", torch.float16
)
def batch_norm_native_f16_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return _batch_norm_native_impl_float(
        input, weight, bias, running_mean, running_var, eps, torch.float16
    )


def _svdf_meta_float(
    input: torch.Tensor,
    initial_state: torch.Tensor,
    weights_feature: torch.Tensor,
    weights_time: torch.Tensor,
    bias: torch.Tensor,
    time_major: bool,
    rank: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    assert (
        input.dtype == dtype
    ), f"cortex_m.svdf: expected {dtype} input tensor, got {input.dtype}"
    assert (
        initial_state.dtype == dtype
        and weights_feature.dtype == dtype
        and weights_time.dtype == dtype
        and bias.dtype == dtype
    ), f"cortex_m.svdf: expected {dtype} state/weight/bias tensors"
    assert input.dim() == 3, f"cortex_m.svdf: expected rank-3 input, got {input.dim()}"
    assert (
        initial_state.dim() == 3
    ), f"cortex_m.svdf: expected rank-3 initial_state, got {initial_state.dim()}"
    assert (
        weights_feature.dim() == 2 and weights_time.dim() == 2
    ), "cortex_m.svdf: weights_feature and weights_time must be rank-2"
    assert bias.dim() == 1, f"cortex_m.svdf: expected rank-1 bias, got {bias.dim()}"

    batch_size = input.shape[1] if time_major else input.shape[0]
    input_size = input.shape[2]
    feature_batches = weights_feature.shape[0]
    memory_size = weights_time.shape[1]

    assert (
        weights_feature.shape[1] == input_size
    ), "cortex_m.svdf: weights_feature second dimension must match input_size"
    assert (
        weights_time.shape[0] == feature_batches
    ), "cortex_m.svdf: weights_time first dimension must match feature_batches"
    assert initial_state.shape == (batch_size, feature_batches, memory_size), (
        f"cortex_m.svdf: initial_state must have shape {(batch_size, feature_batches, memory_size)}, "
        f"got {tuple(initial_state.shape)}"
    )
    assert (
        rank > 0 and feature_batches % rank == 0
    ), "cortex_m.svdf: feature_batches must be divisible by rank"
    unit_count = feature_batches // rank
    assert bias.shape == (
        unit_count,
    ), f"cortex_m.svdf: bias must have shape ({unit_count},), got {tuple(bias.shape)}"
    return torch.empty((batch_size, unit_count), dtype=dtype, device=input.device)


def _svdf_impl_float(
    input: torch.Tensor,
    initial_state: torch.Tensor,
    weights_feature: torch.Tensor,
    weights_time: torch.Tensor,
    bias: torch.Tensor,
    time_major: bool,
    rank: int,
    input_activation_min: float,
    input_activation_max: float,
    output_activation_min: float,
    output_activation_max: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    out = _svdf_meta_float(
        input,
        initial_state,
        weights_feature,
        weights_time,
        bias,
        time_major,
        rank,
        dtype,
    )
    sequence = input if time_major else input.transpose(0, 1)
    state = initial_state.clone()
    unit_count = bias.shape[0]
    output = out
    for step in range(sequence.shape[0]):
        x_t = sequence[step]
        projected = torch.clamp(
            x_t @ weights_feature.transpose(0, 1),
            min=input_activation_min,
            max=input_activation_max,
        )
        state = torch.roll(state, shifts=-1, dims=2)
        state[:, :, -1] = projected
        out_a = torch.sum(state * weights_time.unsqueeze(0), dim=2)
        out_b = out_a.reshape(x_t.shape[0], unit_count, rank).sum(dim=2) + bias
        output = torch.clamp(
            out_b, min=output_activation_min, max=output_activation_max
        )
    return output


@_register_fake_if_enabled("cortex_m::svdf_f32", torch.float32)
def svdf_f32_meta(
    input: torch.Tensor,
    initial_state: torch.Tensor,
    weights_feature: torch.Tensor,
    weights_time: torch.Tensor,
    bias: torch.Tensor,
    time_major: bool,
    rank: int,
    input_activation_min: float,
    input_activation_max: float,
    output_activation_min: float,
    output_activation_max: float,
) -> torch.Tensor:
    del (
        input_activation_min,
        input_activation_max,
        output_activation_min,
        output_activation_max,
    )
    return _svdf_meta_float(
        input,
        initial_state,
        weights_feature,
        weights_time,
        bias,
        time_major,
        rank,
        torch.float32,
    )


@_impl_if_enabled(lib, "svdf_f32", "CompositeExplicitAutograd", torch.float32)
def svdf_f32_impl(
    input: torch.Tensor,
    initial_state: torch.Tensor,
    weights_feature: torch.Tensor,
    weights_time: torch.Tensor,
    bias: torch.Tensor,
    time_major: bool,
    rank: int,
    input_activation_min: float,
    input_activation_max: float,
    output_activation_min: float,
    output_activation_max: float,
) -> torch.Tensor:
    return _svdf_impl_float(
        input,
        initial_state,
        weights_feature,
        weights_time,
        bias,
        time_major,
        rank,
        input_activation_min,
        input_activation_max,
        output_activation_min,
        output_activation_max,
        torch.float32,
    )


@_register_fake_if_enabled("cortex_m::svdf_f16", torch.float16)
def svdf_f16_meta(
    input: torch.Tensor,
    initial_state: torch.Tensor,
    weights_feature: torch.Tensor,
    weights_time: torch.Tensor,
    bias: torch.Tensor,
    time_major: bool,
    rank: int,
    input_activation_min: float,
    input_activation_max: float,
    output_activation_min: float,
    output_activation_max: float,
) -> torch.Tensor:
    del (
        input_activation_min,
        input_activation_max,
        output_activation_min,
        output_activation_max,
    )
    return _svdf_meta_float(
        input,
        initial_state,
        weights_feature,
        weights_time,
        bias,
        time_major,
        rank,
        torch.float16,
    )


@_impl_if_enabled(lib, "svdf_f16", "CompositeExplicitAutograd", torch.float16)
def svdf_f16_impl(
    input: torch.Tensor,
    initial_state: torch.Tensor,
    weights_feature: torch.Tensor,
    weights_time: torch.Tensor,
    bias: torch.Tensor,
    time_major: bool,
    rank: int,
    input_activation_min: float,
    input_activation_max: float,
    output_activation_min: float,
    output_activation_max: float,
) -> torch.Tensor:
    return _svdf_impl_float(
        input,
        initial_state,
        weights_feature,
        weights_time,
        bias,
        time_major,
        rank,
        input_activation_min,
        input_activation_max,
        output_activation_min,
        output_activation_max,
        torch.float16,
    )


# ===================================================================
# TRANSPOSE OPERATION DEFINITION
# ===================================================================
lib.define(
    "pad_f32(Tensor input, int[] pre_pad, int[] post_pad, float pad_value) -> Tensor"
)
lib.define(
    "pad_f32.out(Tensor input, int[] pre_pad, int[] post_pad, float pad_value, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)
lib.define(
    "pad_f16(Tensor input, int[] pre_pad, int[] post_pad, float pad_value) -> Tensor"
)
lib.define(
    "pad_f16.out(Tensor input, int[] pre_pad, int[] post_pad, float pad_value, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


def _pad_meta_float(
    input: torch.Tensor,
    pre_pad: list[int],
    post_pad: list[int],
) -> torch.Tensor:
    rank = input.dim()
    offset = 4 - rank
    logical_pre = _pad_to_logical_order(pre_pad, input)
    logical_post = _pad_to_logical_order(post_pad, input)

    output_shape = list(input.shape)
    for i in range(rank):
        output_shape[i] += logical_pre[offset + i] + logical_post[offset + i]
    result = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    if is_channels_last(input):
        result = result.to(memory_format=torch.channels_last)
    return result


def _pad_impl_float(
    input: torch.Tensor,
    pre_pad: list[int],
    post_pad: list[int],
    pad_value: float,
) -> torch.Tensor:
    rank = input.dim()
    offset = 4 - rank
    logical_pre = _pad_to_logical_order(pre_pad, input)
    logical_post = _pad_to_logical_order(post_pad, input)

    padding = []
    for i in reversed(range(rank)):
        padding.extend([logical_pre[offset + i], logical_post[offset + i]])
    return F.pad(input, padding, mode="constant", value=float(pad_value))


@_register_fake_if_enabled("cortex_m::pad_f32", torch.float32)
def pad_f32_meta(
    input: torch.Tensor,
    pre_pad: list[int],
    post_pad: list[int],
    pad_value: float,
) -> torch.Tensor:
    del pad_value
    return _pad_meta_float(input, pre_pad, post_pad)


@_impl_if_enabled(lib, "pad_f32", "CompositeExplicitAutograd", torch.float32)
def pad_f32_impl(
    input: torch.Tensor,
    pre_pad: list[int],
    post_pad: list[int],
    pad_value: float,
) -> torch.Tensor:
    if input.dtype != torch.float32:
        raise TypeError(
            f"cortex_m.pad_f32: expected float32 input tensor, got {input.dtype}"
        )
    return _pad_impl_float(input, pre_pad, post_pad, pad_value)


@_register_fake_if_enabled("cortex_m::pad_f16", torch.float16)
def pad_f16_meta(
    input: torch.Tensor,
    pre_pad: list[int],
    post_pad: list[int],
    pad_value: float,
) -> torch.Tensor:
    del pad_value
    return _pad_meta_float(input, pre_pad, post_pad)


@_impl_if_enabled(lib, "pad_f16", "CompositeExplicitAutograd", torch.float16)
def pad_f16_impl(
    input: torch.Tensor,
    pre_pad: list[int],
    post_pad: list[int],
    pad_value: float,
) -> torch.Tensor:
    if input.dtype != torch.float16:
        raise TypeError(
            f"cortex_m.pad_f16: expected float16 input tensor, got {input.dtype}"
        )
    return _pad_impl_float(input, pre_pad, post_pad, pad_value)


# ===================================================================
# FLOAT / QUANTIZED CONV2D OPERATION DEFINITION
# ===================================================================
# Packed float conv weights follow the same idea as packed linear/BMM weights:
# the tensor becomes a flat kernel buffer and the original [O, KH, KW, I]
# shape is carried explicitly through packed_* metadata.

lib.define(
    "conv2d_f32("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] dilation, "
    "bool weight_is_packed, "
    "int packed_output_channels, "
    "int packed_kernel_height, "
    "int packed_kernel_width, "
    "int packed_kernel_input_channels, "
    "float activation_min, "
    "float activation_max"
    ") -> Tensor"
)
lib.define(
    "conv2d_f32.out("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] dilation, "
    "bool weight_is_packed, "
    "int packed_output_channels, "
    "int packed_kernel_height, "
    "int packed_kernel_width, "
    "int packed_kernel_input_channels, "
    "float activation_min, "
    "float activation_max, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)

lib.define(
    "conv2d_f16("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] dilation, "
    "bool weight_is_packed, "
    "int packed_output_channels, "
    "int packed_kernel_height, "
    "int packed_kernel_width, "
    "int packed_kernel_input_channels, "
    "float activation_min, "
    "float activation_max"
    ") -> Tensor"
)
lib.define(
    "conv2d_f16.out("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] dilation, "
    "bool weight_is_packed, "
    "int packed_output_channels, "
    "int packed_kernel_height, "
    "int packed_kernel_width, "
    "int packed_kernel_input_channels, "
    "float activation_min, "
    "float activation_max, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)


def _float_conv2d_meta_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    weight_is_packed: bool,
    packed_output_channels: int,
    packed_kernel_height: int,
    packed_kernel_width: int,
    packed_kernel_input_channels: int,
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    assert (
        input.dtype == expected_dtype
    ), f"Cortex-M float conv expects input dtype={expected_dtype}, got {input.dtype}"
    assert (
        weight.dtype == expected_dtype
    ), f"Cortex-M float conv expects weight dtype={expected_dtype}, got {weight.dtype}"
    assert input.dim() == 4, "Cortex-M float conv expects 4D input tensors"
    if weight_is_packed:
        # Packed conv constants are stored as a flat buffer. The original
        # [O, KH, KW, I] shape is reconstructed from the packed_* metadata.
        assert (
            weight.dim() == 1
        ), "Cortex-M float conv expects rank-1 packed weights when weight_is_packed=True"
        assert (
            packed_output_channels > 0
        ), "Cortex-M float conv expects packed_output_channels > 0 when weight_is_packed=True"
        assert (
            packed_kernel_height > 0
            and packed_kernel_width > 0
            and packed_kernel_input_channels > 0
        ), (
            "Cortex-M float conv expects packed kernel shape metadata when "
            "weight_is_packed=True"
        )
    else:
        assert (
            weight.dim() == 4
        ), "Cortex-M float conv expects 4D weight tensors when unpacked"
    assert (
        len(stride) == 2 and len(padding) == 2 and len(dilation) == 2
    ), "Cortex-M float conv expects stride/padding/dilation length == 2"
    assert is_channels_last(input), "Cortex-M float conv expects channels-last input"
    if bias is not None:
        assert (
            bias.dtype == expected_dtype and bias.dim() == 1
        ), "Cortex-M float conv expects optional bias with shape [out_channels]"
        expected_out_channels = (
            packed_output_channels if weight_is_packed else weight.shape[0]
        )
        assert (
            bias.shape[0] == expected_out_channels
        ), f"Cortex-M float conv bias size mismatch: {bias.shape[0]} vs {expected_out_channels}"

    if weight_is_packed:
        output_shape = _compute_conv2d_output_shape(
            input.shape,
            torch.Size(
                [
                    packed_output_channels,
                    packed_kernel_height,
                    packed_kernel_width,
                    packed_kernel_input_channels,
                ]
            ),
            list(stride),
            list(padding),
            list(dilation),
        )
    else:
        output_shape = _compute_conv2d_output_shape(
            input.shape, weight.shape, list(stride), list(padding), list(dilation)
        )
    return torch.empty(
        output_shape,
        dtype=expected_dtype,
        device=input.device,
        memory_format=torch.channels_last,
    )


def _float_conv2d_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    weight_is_packed: bool,
    packed_output_channels: int,
    packed_kernel_height: int,
    packed_kernel_width: int,
    packed_kernel_input_channels: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    _float_conv2d_meta_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        weight_is_packed,
        packed_output_channels,
        packed_kernel_height,
        packed_kernel_width,
        packed_kernel_input_channels,
        input.dtype,
    )
    if weight_is_packed:
        # Reconstruct a standard 4D kernel for the Python reference path so the
        # eager fallback stays numerically comparable to the backend execution.
        block_cols = 8 if input.dtype == torch.float16 else 4
        flat_features = (
            packed_kernel_height * packed_kernel_width * packed_kernel_input_channels
        )
        packed_blocks = (packed_output_channels + block_cols - 1) // block_cols
        expected_numel = packed_blocks * flat_features * block_cols
        assert weight.numel() == expected_numel, (
            "Packed Cortex-M float conv weight size mismatch: "
            f"{weight.numel()} vs {expected_numel}"
        )
        packed_3d = weight.reshape(packed_blocks, flat_features, block_cols)
        unpacked = torch.zeros(
            (packed_output_channels, flat_features),
            dtype=input.dtype,
            device=input.device,
        )
        for out_channel in range(packed_output_channels):
            block = out_channel // block_cols
            lane = out_channel % block_cols
            unpacked[out_channel].copy_(packed_3d[block, :, lane])
        weight = unpacked.reshape(
            packed_output_channels,
            packed_kernel_height,
            packed_kernel_width,
            packed_kernel_input_channels,
        )
    weight_oihw = weight.permute(0, 3, 1, 2).contiguous()
    result = F.conv2d(
        input,
        weight_oihw,
        bias,
        stride=tuple(stride),
        padding=tuple(padding),
        dilation=tuple(dilation),
        groups=1,
    )
    result = torch.clamp(result, min=activation_min, max=activation_max)
    return result.contiguous(memory_format=torch.channels_last)


def _float_depthwise_conv2d_meta_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    depth_multiplier: int,
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    assert (
        input.dtype == expected_dtype
    ), f"Cortex-M float depthwise conv expects input dtype={expected_dtype}, got {input.dtype}"
    assert (
        weight.dtype == expected_dtype
    ), f"Cortex-M float depthwise conv expects weight dtype={expected_dtype}, got {weight.dtype}"
    assert (
        input.dim() == 4 and weight.dim() == 4
    ), "Cortex-M float depthwise conv expects 4D input and weight tensors"
    assert (
        len(stride) == 2 and len(padding) == 2 and len(dilation) == 2
    ), "Cortex-M float depthwise conv expects stride/padding/dilation length == 2"
    assert is_channels_last(
        input
    ), "Cortex-M float depthwise conv expects channels-last input"
    assert (
        weight.shape[0] == 1
    ), f"Cortex-M float depthwise conv expects IHWO weights with dim0==1, got {weight.shape[0]}"
    in_channels = input.shape[1]
    out_channels = weight.shape[3]
    assert out_channels == in_channels * depth_multiplier, (
        "Cortex-M float depthwise conv expects out_channels == in_channels * depth_multiplier, "
        f"got out_channels={out_channels}, in_channels={in_channels}, depth_multiplier={depth_multiplier}"
    )
    if bias is not None:
        assert (
            bias.dtype == expected_dtype and bias.dim() == 1
        ), "Cortex-M float depthwise conv expects optional bias with shape [out_channels]"
        assert (
            bias.shape[0] == out_channels
        ), f"Cortex-M float depthwise conv bias size mismatch: {bias.shape[0]} vs {out_channels}"

    output_shape = _compute_depthwise_conv2d_output_shape(
        input.shape, weight.shape, list(stride), list(padding), list(dilation)
    )
    return torch.empty(
        output_shape,
        dtype=expected_dtype,
        device=input.device,
        memory_format=torch.channels_last,
    )


def _float_depthwise_conv2d_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    depth_multiplier: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    del depth_multiplier
    weight_oihw = weight.permute(3, 0, 1, 2).contiguous()
    result = F.conv2d(
        input,
        weight_oihw,
        bias,
        stride=tuple(stride),
        padding=tuple(padding),
        dilation=tuple(dilation),
        groups=input.shape[1],
    )
    result = torch.clamp(result, min=activation_min, max=activation_max)
    return result.contiguous(memory_format=torch.channels_last)


@_register_fake_if_enabled("cortex_m::conv2d_f32", torch.float32)
def conv2d_f32_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    weight_is_packed: bool,
    packed_output_channels: int,
    packed_kernel_height: int,
    packed_kernel_width: int,
    packed_kernel_input_channels: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    del activation_min, activation_max
    return _float_conv2d_meta_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        weight_is_packed,
        packed_output_channels,
        packed_kernel_height,
        packed_kernel_width,
        packed_kernel_input_channels,
        torch.float32,
    )


@_impl_if_enabled(lib, "conv2d_f32", "CompositeExplicitAutograd", torch.float32)
def conv2d_f32_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    weight_is_packed: bool,
    packed_output_channels: int,
    packed_kernel_height: int,
    packed_kernel_width: int,
    packed_kernel_input_channels: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    return _float_conv2d_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        weight_is_packed,
        packed_output_channels,
        packed_kernel_height,
        packed_kernel_width,
        packed_kernel_input_channels,
        activation_min,
        activation_max,
    )


@_register_fake_if_enabled("cortex_m::conv2d_f16", torch.float16)
def conv2d_f16_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    weight_is_packed: bool,
    packed_output_channels: int,
    packed_kernel_height: int,
    packed_kernel_width: int,
    packed_kernel_input_channels: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    del activation_min, activation_max
    return _float_conv2d_meta_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        weight_is_packed,
        packed_output_channels,
        packed_kernel_height,
        packed_kernel_width,
        packed_kernel_input_channels,
        torch.float16,
    )


@_impl_if_enabled(lib, "conv2d_f16", "CompositeExplicitAutograd", torch.float16)
def conv2d_f16_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    weight_is_packed: bool,
    packed_output_channels: int,
    packed_kernel_height: int,
    packed_kernel_width: int,
    packed_kernel_input_channels: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    return _float_conv2d_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        weight_is_packed,
        packed_output_channels,
        packed_kernel_height,
        packed_kernel_width,
        packed_kernel_input_channels,
        activation_min,
        activation_max,
    )


@_register_fake_if_enabled("cortex_m::depthwise_conv2d_f32", torch.float32)
def depthwise_conv2d_f32_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    depth_multiplier: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    del activation_min, activation_max
    return _float_depthwise_conv2d_meta_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        depth_multiplier,
        torch.float32,
    )


@_impl_if_enabled(
    lib, "depthwise_conv2d_f32", "CompositeExplicitAutograd", torch.float32
)
def depthwise_conv2d_f32_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    depth_multiplier: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    return _float_depthwise_conv2d_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        depth_multiplier,
        activation_min,
        activation_max,
    )


@_register_fake_if_enabled("cortex_m::depthwise_conv2d_f16", torch.float16)
def depthwise_conv2d_f16_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    depth_multiplier: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    del activation_min, activation_max
    return _float_depthwise_conv2d_meta_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        depth_multiplier,
        torch.float16,
    )


@_impl_if_enabled(
    lib, "depthwise_conv2d_f16", "CompositeExplicitAutograd", torch.float16
)
def depthwise_conv2d_f16_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    depth_multiplier: int,
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    return _float_depthwise_conv2d_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        depth_multiplier,
        activation_min,
        activation_max,
    )


lib.define(
    "transpose_conv2d_f32("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] output_padding, "
    "int[] dilation, "
    "float activation_min, "
    "float activation_max"
    ") -> Tensor"
)

lib.define(
    "transpose_conv2d_f32.out("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] output_padding, "
    "int[] dilation, "
    "float activation_min, "
    "float activation_max, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)

lib.define(
    "transpose_conv2d_f16("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] output_padding, "
    "int[] dilation, "
    "float activation_min, "
    "float activation_max"
    ") -> Tensor"
)

lib.define(
    "transpose_conv2d_f16.out("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] output_padding, "
    "int[] dilation, "
    "float activation_min, "
    "float activation_max, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


def _transpose_conv2d_float_meta_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    output_padding: Sequence[int],
    dilation: Sequence[int],
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    assert (
        input.dtype == expected_dtype
    ), f"Cortex-M float transpose_conv2d expects input dtype={expected_dtype}, got {input.dtype}"
    assert (
        weight.dtype == expected_dtype
    ), f"Cortex-M float transpose_conv2d expects weight dtype={expected_dtype}, got {weight.dtype}"
    assert input.dim() == 4 and weight.dim() == 4, (
        "Cortex-M float transpose_conv2d expects 4D input and weight tensors, "
        f"got input.dim()={input.dim()}, weight.dim()={weight.dim()}"
    )
    assert is_channels_last(
        input
    ), "Cortex-M float transpose_conv2d expects channels-last input"
    if bias is not None:
        assert (
            bias.dtype == expected_dtype
        ), f"Cortex-M float transpose_conv2d expects bias dtype={expected_dtype}, got {bias.dtype}"
        assert bias.dim() == 1 and bias.shape[0] == weight.shape[0], (
            "Cortex-M float transpose_conv2d expects bias shape [out_channels], "
            f"got bias.shape={tuple(bias.shape)}, out_channels={weight.shape[0]}"
        )

    output_shape = _compute_conv_transpose2d_output_shape(
        input.shape,
        weight.shape,
        list(stride),
        list(padding),
        list(output_padding),
        list(dilation),
    )

    return torch.empty(
        output_shape,
        dtype=expected_dtype,
        device=input.device,
        memory_format=torch.channels_last,
    )


def _transpose_conv2d_float_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    output_padding: Sequence[int],
    dilation: Sequence[int],
    activation_min: float,
    activation_max: float,
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    _transpose_conv2d_float_meta_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        expected_dtype,
    )

    weight_iohw = weight.permute(3, 0, 1, 2).contiguous()
    result = F.conv_transpose2d(
        input,
        weight_iohw,
        bias,
        stride=_ensure_tuple2(stride),
        padding=_ensure_tuple2(padding),
        output_padding=_ensure_tuple2(output_padding),
        dilation=_ensure_tuple2(dilation),
        groups=1,
    )
    result = torch.clamp(result, min=activation_min, max=activation_max)
    return result.contiguous(memory_format=torch.channels_last)


@_register_fake_if_enabled("cortex_m::transpose_conv2d_f32", torch.float32)
def transpose_conv2d_f32_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    output_padding: Sequence[int],
    dilation: Sequence[int],
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    del activation_min, activation_max
    return _transpose_conv2d_float_meta_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        torch.float32,
    )


@_impl_if_enabled(
    lib, "transpose_conv2d_f32", "CompositeExplicitAutograd", torch.float32
)
def transpose_conv2d_f32_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    output_padding: Sequence[int],
    dilation: Sequence[int],
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    return _transpose_conv2d_float_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        activation_min,
        activation_max,
        torch.float32,
    )


@_register_fake_if_enabled("cortex_m::transpose_conv2d_f16", torch.float16)
def transpose_conv2d_f16_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    output_padding: Sequence[int],
    dilation: Sequence[int],
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    del activation_min, activation_max
    return _transpose_conv2d_float_meta_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        torch.float16,
    )


@_impl_if_enabled(
    lib, "transpose_conv2d_f16", "CompositeExplicitAutograd", torch.float16
)
def transpose_conv2d_f16_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    output_padding: Sequence[int],
    dilation: Sequence[int],
    activation_min: float,
    activation_max: float,
) -> torch.Tensor:
    return _transpose_conv2d_float_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        activation_min,
        activation_max,
        torch.float16,
    )


# ===================================================================
# QUANTIZED AVG_POOL2D OPERATION DEFINITION
# ===================================================================

lib.define(
    "avg_pool2d_f32("
    "Tensor input, "
    "int[] kernel_size, "
    "int[] stride, "
    "int[] padding"
    ") -> Tensor"
)
lib.define(
    "avg_pool2d_f32.out("
    "Tensor input, "
    "int[] kernel_size, "
    "int[] stride, "
    "int[] padding, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@_register_fake_if_enabled("cortex_m::avg_pool2d_f32", torch.float32)
def avg_pool2d_f32_meta(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
) -> torch.Tensor:
    assert (
        input.dtype == torch.float32
    ), f"Cortex-M avg_pool2d_f32 expects float32 inputs, got {input.dtype}"
    output = F.avg_pool2d(
        input,
        _ensure_tuple2(kernel_size),
        stride=_ensure_tuple2(stride),
        padding=_ensure_tuple2(padding),
        ceil_mode=False,
        count_include_pad=False,
    )
    return torch.empty_like(output)


@_impl_if_enabled(lib, "avg_pool2d_f32", "CompositeExplicitAutograd", torch.float32)
def avg_pool2d_f32_impl(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
) -> torch.Tensor:
    assert (
        input.dtype == torch.float32
    ), f"Cortex-M avg_pool2d_f32 expects float32 inputs, got {input.dtype}"
    return F.avg_pool2d(
        input,
        _ensure_tuple2(kernel_size),
        stride=_ensure_tuple2(stride),
        padding=_ensure_tuple2(padding),
        ceil_mode=False,
        count_include_pad=False,
    )


lib.define(
    "avg_pool2d_f16("
    "Tensor input, "
    "int[] kernel_size, "
    "int[] stride, "
    "int[] padding"
    ") -> Tensor"
)
lib.define(
    "avg_pool2d_f16.out("
    "Tensor input, "
    "int[] kernel_size, "
    "int[] stride, "
    "int[] padding, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@_register_fake_if_enabled("cortex_m::avg_pool2d_f16", torch.float16)
def avg_pool2d_f16_meta(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
) -> torch.Tensor:
    assert (
        input.dtype == torch.float16
    ), f"Cortex-M avg_pool2d_f16 expects float16 inputs, got {input.dtype}"
    output = F.avg_pool2d(
        input,
        _ensure_tuple2(kernel_size),
        stride=_ensure_tuple2(stride),
        padding=_ensure_tuple2(padding),
        ceil_mode=False,
        count_include_pad=False,
    )
    return torch.empty_like(output)


@_impl_if_enabled(lib, "avg_pool2d_f16", "CompositeExplicitAutograd", torch.float16)
def avg_pool2d_f16_impl(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
) -> torch.Tensor:
    assert (
        input.dtype == torch.float16
    ), f"Cortex-M avg_pool2d_f16 expects float16 inputs, got {input.dtype}"
    return F.avg_pool2d(
        input,
        _ensure_tuple2(kernel_size),
        stride=_ensure_tuple2(stride),
        padding=_ensure_tuple2(padding),
        ceil_mode=False,
        count_include_pad=False,
    )


lib.define(
    "max_pool2d_f32("
    "Tensor input, "
    "int[] kernel_size, "
    "int[] stride, "
    "int[] padding"
    ") -> Tensor"
)

lib.define(
    "max_pool2d_f32.out("
    "Tensor input, "
    "int[] kernel_size, "
    "int[] stride, "
    "int[] padding, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)


@_register_fake_if_enabled("cortex_m::max_pool2d_f32", torch.float32)
def max_pool2d_f32_meta(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
) -> torch.Tensor:
    if input.dtype != torch.float32:
        raise RuntimeError(
            f"Cortex-M max_pool2d_f32 expects float32 inputs, got {input.dtype}"
        )

    kernel = _ensure_tuple2(kernel_size)
    stride_vals = _ensure_tuple2(stride)
    padding_vals = _ensure_tuple2(padding)
    output_shape = _compute_max_pool2d_output_shape(
        input.shape, kernel, stride_vals, padding_vals, (1, 1)
    )
    return torch.empty(
        output_shape,
        dtype=torch.float32,
        device=input.device,
        memory_format=torch.channels_last,
    )


@_impl_if_enabled(lib, "max_pool2d_f32", "CompositeExplicitAutograd", torch.float32)
def max_pool2d_f32_impl(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
) -> torch.Tensor:
    if input.dtype != torch.float32:
        raise RuntimeError(
            f"Cortex-M max_pool2d_f32 expects float32 inputs, got {input.dtype}"
        )

    kernel = _ensure_tuple2(kernel_size)
    stride_vals = _ensure_tuple2(stride)
    padding_vals = _ensure_tuple2(padding)
    return F.max_pool2d(
        input,
        kernel,
        stride=stride_vals,
        padding=padding_vals,
        dilation=(1, 1),
        ceil_mode=False,
    ).contiguous(memory_format=torch.channels_last)


lib.define(
    "max_pool2d_f16("
    "Tensor input, "
    "int[] kernel_size, "
    "int[] stride, "
    "int[] padding"
    ") -> Tensor"
)

lib.define(
    "max_pool2d_f16.out("
    "Tensor input, "
    "int[] kernel_size, "
    "int[] stride, "
    "int[] padding, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)


@_register_fake_if_enabled("cortex_m::max_pool2d_f16", torch.float16)
def max_pool2d_f16_meta(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
) -> torch.Tensor:
    if input.dtype != torch.float16:
        raise RuntimeError(
            f"Cortex-M max_pool2d_f16 expects float16 inputs, got {input.dtype}"
        )

    kernel = _ensure_tuple2(kernel_size)
    stride_vals = _ensure_tuple2(stride)
    padding_vals = _ensure_tuple2(padding)
    output_shape = _compute_max_pool2d_output_shape(
        input.shape, kernel, stride_vals, padding_vals, (1, 1)
    )
    return torch.empty(
        output_shape,
        dtype=torch.float16,
        device=input.device,
        memory_format=torch.channels_last,
    )


@_impl_if_enabled(lib, "max_pool2d_f16", "CompositeExplicitAutograd", torch.float16)
def max_pool2d_f16_impl(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
) -> torch.Tensor:
    if input.dtype != torch.float16:
        raise RuntimeError(
            f"Cortex-M max_pool2d_f16 expects float16 inputs, got {input.dtype}"
        )

    kernel = _ensure_tuple2(kernel_size)
    stride_vals = _ensure_tuple2(stride)
    padding_vals = _ensure_tuple2(padding)
    return F.max_pool2d(
        input,
        kernel,
        stride=stride_vals,
        padding=padding_vals,
        dilation=(1, 1),
        ceil_mode=False,
    ).contiguous(memory_format=torch.channels_last)


@register_fake("cortex_m::quantized_max_pool2d")  # type: ignore[misc]
@impl(lib, "quantized_conv2d_nhwc", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_conv2d_nhwc_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    nchw = quantized_conv2d_impl(
        input.permute(0, 3, 1, 2).contiguous(),
        weight,
        bias,
        stride,
        padding,
        dilation,
        input_offset,
        output_offset,
        requantize_multipliers,
        requantize_shifts,
        activation_min,
        activation_max,
        scratch,
    )
    return nchw.permute(0, 2, 3, 1).contiguous()


# ===================================================================
# QUANTIZED DEPTHWISE CONV2D OPERATION DEFINITION
# ===================================================================

lib.define(
    "quantized_depthwise_conv2d("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] dilation, "
    "int depth_multiplier, "
    "int input_offset, "
    "int output_offset, "
    "Tensor requantize_multipliers, "
    "Tensor requantize_shifts, "
    "int activation_min, "
    "int activation_max, "
    "Tensor scratch"
    ") -> Tensor"
)


lib.define(
    "quantized_depthwise_conv2d.out("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] dilation, "
    "int depth_multiplier, "
    "int input_offset, "
    "int output_offset, "
    "Tensor requantize_multipliers, "
    "Tensor requantize_shifts, "
    "int activation_min, "
    "int activation_max, "
    "Tensor scratch, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_depthwise_conv2d")  # type: ignore[misc]
def quantized_depthwise_conv2d_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    depth_multiplier: int,
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    stride_vals = list(stride)
    padding_vals = list(padding)
    dilation_vals = list(dilation)
    output_shape = _compute_depthwise_conv2d_output_shape(
        input.shape, weight.shape, stride_vals, padding_vals, dilation_vals
    )
    return torch.empty(
        output_shape,
        dtype=torch.int8,
        device=input.device,
        memory_format=torch.channels_last,
    )


@impl(lib, "quantized_depthwise_conv2d", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_depthwise_conv2d_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    depth_multiplier: int,
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    if input.dim() != 4 or weight.dim() != 4:
        raise RuntimeError(
            "quantized_depthwise_conv2d expects 4D input and weight tensors"
        )

    input_channels = input.shape[1]
    groups = input_channels

    # Convert to int32 for accumulation and apply offsets
    input_int32 = input.to(torch.int32) + int(input_offset)
    weight_int32 = weight.to(torch.int32)

    if bias is None:
        bias_int32 = torch.zeros(
            weight.shape[3],
            dtype=torch.int32,
            device=input.device,  # C_OUT is at dim 3 in IHWO
        )
    else:
        bias_int32 = bias.to(torch.int32)

    # Weight is in IHWO layout: [1, H, W, C_OUT]
    # Convert to OIHW layout expected by torch.nn.functional.conv2d
    # IHWO [1, H, W, C_OUT] -> OIHW [C_OUT, 1, H, W]
    weight_oi_hw = weight_int32.permute(3, 0, 1, 2).contiguous()

    # Depthwise convolution has groups == input_channels
    conv_acc = F.conv2d(
        input_int32,
        weight_oi_hw,
        bias_int32,
        stride=tuple(stride),
        padding=tuple(padding),
        dilation=tuple(dilation),
        groups=groups,
    )

    result_channels = []
    for output_channel_i in range(conv_acc.shape[1]):
        result_channel = requantize_cmsis(
            conv_acc[:, output_channel_i, :, :],
            int(requantize_multipliers[output_channel_i]),
            int(requantize_shifts[output_channel_i]),
        )
        result_channels.append(result_channel)

    result = torch.stack(result_channels, dim=1)

    result += output_offset
    result = torch.clamp(result, activation_min, activation_max)

    return result.to(torch.int8, memory_format=torch.channels_last)


lib.define(
    "quantized_depthwise_conv2d_nhwc("
    "Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, "
    "int[] dilation, int depth_multiplier, int input_offset, int output_offset, "
    "Tensor requantize_multipliers, Tensor requantize_shifts, "
    "int activation_min, int activation_max, Tensor scratch) -> Tensor"
)
lib.define(
    "quantized_depthwise_conv2d_nhwc.out("
    "Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, "
    "int[] dilation, int depth_multiplier, int input_offset, int output_offset, "
    "Tensor requantize_multipliers, Tensor requantize_shifts, "
    "int activation_min, int activation_max, Tensor scratch, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_depthwise_conv2d_nhwc")  # type: ignore[misc]
@experimental(_EXPLICIT_LAYOUT_EXPERIMENTAL)  # type: ignore[misc]
def quantized_depthwise_conv2d_nhwc_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    depth_multiplier: int,
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    nchw = quantized_depthwise_conv2d_meta(
        input.permute(0, 3, 1, 2),
        weight,
        bias,
        stride,
        padding,
        dilation,
        depth_multiplier,
        input_offset,
        output_offset,
        requantize_multipliers,
        requantize_shifts,
        activation_min,
        activation_max,
        scratch,
    )
    return nchw.permute(0, 2, 3, 1).contiguous()


@impl(lib, "quantized_depthwise_conv2d_nhwc", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_depthwise_conv2d_nhwc_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    depth_multiplier: int,
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    nchw = quantized_depthwise_conv2d_impl(
        input.permute(0, 3, 1, 2).contiguous(),
        weight,
        bias,
        stride,
        padding,
        dilation,
        depth_multiplier,
        input_offset,
        output_offset,
        requantize_multipliers,
        requantize_shifts,
        activation_min,
        activation_max,
        scratch,
    )
    return nchw.permute(0, 2, 3, 1).contiguous()


# ===================================================================
# QUANTIZED TRANSPOSE_CONV2D OPERATION DEFINITION
# ===================================================================

lib.define(
    "quantized_transpose_conv2d("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] output_padding, "
    "int[] dilation, "
    "int input_offset, "
    "int output_offset, "
    "Tensor requantize_multipliers, "
    "Tensor requantize_shifts, "
    "int activation_min, "
    "int activation_max, "
    "Tensor scratch, "
    "Tensor output_scratch"
    ") -> Tensor"
)

lib.define(
    "quantized_transpose_conv2d.out("
    "Tensor input, "
    "Tensor weight, "
    "Tensor? bias, "
    "int[] stride, "
    "int[] padding, "
    "int[] output_padding, "
    "int[] dilation, "
    "int input_offset, "
    "int output_offset, "
    "Tensor requantize_multipliers, "
    "Tensor requantize_shifts, "
    "int activation_min, "
    "int activation_max, "
    "Tensor scratch, "
    "Tensor output_scratch, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


def _compute_conv_transpose2d_output_shape(
    input_shape: torch.Size,
    weight_shape: torch.Size,
    stride: list[int],
    padding: list[int],
    output_padding: list[int],
    dilation: list[int],
) -> torch.Size:
    """
    Compute output shape for transposed 2D convolution.

    Formula for each dimension:
    out_size = (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    """
    batch = input_shape[0]
    in_height = input_shape[2]
    in_width = input_shape[3]

    # Weight is in OHWI format after permutation
    out_channels = weight_shape[0]
    kernel_height = weight_shape[1]
    kernel_width = weight_shape[2]

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    out_pad_h, out_pad_w = output_padding
    dilation_h, dilation_w = dilation

    out_height = (
        (in_height - 1) * stride_h
        - 2 * pad_h
        + dilation_h * (kernel_height - 1)
        + out_pad_h
        + 1
    )
    out_width = (
        (in_width - 1) * stride_w
        - 2 * pad_w
        + dilation_w * (kernel_width - 1)
        + out_pad_w
        + 1
    )

    return torch.Size([batch, out_channels, out_height, out_width])


@register_fake("cortex_m::quantized_transpose_conv2d")  # type: ignore[misc]
def quantized_transpose_conv2d_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    output_padding: Sequence[int],
    dilation: Sequence[int],
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
    scratch: torch.Tensor,
    output_scratch: torch.Tensor,
) -> torch.Tensor:
    stride_vals = list(stride)
    padding_vals = list(padding)
    output_padding_vals = list(output_padding)
    dilation_vals = list(dilation)

    output_shape = _compute_conv_transpose2d_output_shape(
        input.shape,
        weight.shape,
        stride_vals,
        padding_vals,
        output_padding_vals,
        dilation_vals,
    )

    return torch.empty(
        output_shape,
        dtype=torch.int8,
        device=input.device,
        memory_format=torch.channels_last,
    )


@impl(lib, "quantized_transpose_conv2d", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_transpose_conv2d_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    output_padding: Sequence[int],
    dilation: Sequence[int],
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
    scratch: torch.Tensor,
    output_scratch: torch.Tensor,
) -> torch.Tensor:
    """
    Reference implementation of quantized transposed convolution.
    Simulates CMSIS-NN behavior for testing.
    """
    if input.dim() != 4 or weight.dim() != 4:
        raise RuntimeError(
            "quantized_conv_transpose2d expects 4D input and weight tensors"
        )

    # Convert to int32 for accumulation and apply offsets
    input_int32 = input.to(torch.int32) + int(input_offset)
    weight_int32 = weight.to(torch.int32)

    if bias is None:
        bias_int32 = torch.zeros(
            weight.shape[0], dtype=torch.int32, device=input.device
        )
    else:
        bias_int32 = bias.to(torch.int32)

    input_channels = input.shape[1]
    kernel_input_channels = weight.shape[3]
    groups = input_channels // kernel_input_channels

    # Convert weights from OHWI to IOHW layout for torch.nn.functional.conv_transpose2d
    # Weight is in OHWI (out_channels, H, W, in_channels)
    # F.conv_transpose2d expects IOHW (in_channels, out_channels, H, W)
    weight_iohw = weight_int32.permute(3, 0, 1, 2).contiguous()

    # PyTorch doesn't support int32 for conv_transpose2d, so convert to float
    # Perform transposed convolution with float tensors
    conv_transpose_acc = F.conv_transpose2d(
        input_int32.to(torch.float32),
        weight_iohw.to(torch.float32),
        bias_int32.to(torch.float32),
        stride=tuple(stride),
        padding=tuple(padding),
        output_padding=tuple(output_padding),
        dilation=tuple(dilation),
        groups=groups,
    ).to(torch.int32)

    # Apply per-channel requantization
    result_channels = []
    for output_channel_i in range(conv_transpose_acc.shape[1]):
        result_channel = requantize_cmsis(
            conv_transpose_acc[:, output_channel_i, :, :],
            int(requantize_multipliers[output_channel_i]),
            int(requantize_shifts[output_channel_i]),
        )
        result_channels.append(result_channel)

    result = torch.stack(result_channels, dim=1)
    result += output_offset
    result = result.clamp(activation_min, activation_max)

    return result.to(torch.int8).to(memory_format=torch.channels_last)


lib.define(
    "quantized_transpose_conv2d_nhwc("
    "Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, "
    "int[] output_padding, int[] dilation, int input_offset, int output_offset, "
    "Tensor requantize_multipliers, Tensor requantize_shifts, "
    "int activation_min, int activation_max, Tensor scratch, "
    "Tensor output_scratch) -> Tensor"
)
lib.define(
    "quantized_transpose_conv2d_nhwc.out("
    "Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, "
    "int[] output_padding, int[] dilation, int input_offset, int output_offset, "
    "Tensor requantize_multipliers, Tensor requantize_shifts, "
    "int activation_min, int activation_max, Tensor scratch, "
    "Tensor output_scratch, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_transpose_conv2d_nhwc")  # type: ignore[misc]
@experimental(_EXPLICIT_LAYOUT_EXPERIMENTAL)  # type: ignore[misc]
def quantized_transpose_conv2d_nhwc_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    output_padding: Sequence[int],
    dilation: Sequence[int],
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
    scratch: torch.Tensor,
    output_scratch: torch.Tensor,
) -> torch.Tensor:
    nchw = quantized_transpose_conv2d_meta(
        input.permute(0, 3, 1, 2),
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        input_offset,
        output_offset,
        requantize_multipliers,
        requantize_shifts,
        activation_min,
        activation_max,
        scratch,
        output_scratch,
    )
    return nchw.permute(0, 2, 3, 1).contiguous()


@impl(lib, "quantized_transpose_conv2d_nhwc", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_transpose_conv2d_nhwc_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    output_padding: Sequence[int],
    dilation: Sequence[int],
    input_offset: int,
    output_offset: int,
    requantize_multipliers: torch.Tensor,
    requantize_shifts: torch.Tensor,
    activation_min: int,
    activation_max: int,
    scratch: torch.Tensor,
    output_scratch: torch.Tensor,
) -> torch.Tensor:
    nchw = quantized_transpose_conv2d_impl(
        input.permute(0, 3, 1, 2).contiguous(),
        weight,
        bias,
        stride,
        padding,
        output_padding,
        dilation,
        input_offset,
        output_offset,
        requantize_multipliers,
        requantize_shifts,
        activation_min,
        activation_max,
        scratch,
        output_scratch,
    )
    return nchw.permute(0, 2, 3, 1).contiguous()


# ===================================================================
# QUANTIZED AVG_POOL2D OPERATION DEFINITION
# ===================================================================

lib.define(
    "quantized_avg_pool2d("
    "Tensor input, "
    "int[] kernel_size, "
    "int[] stride, "
    "int[] padding, "
    "bool ceil_mode, "
    "int zero_point, "
    "int multiplier, "
    "int shift, "
    "Tensor scratch"
    ") -> Tensor"
)
lib.define(
    "quantized_avg_pool2d.out("
    "Tensor input, "
    "int[] kernel_size, "
    "int[] stride, "
    "int[] padding, "
    "bool ceil_mode, "
    "int zero_point, "
    "int multiplier, "
    "int shift, "
    "Tensor scratch, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_avg_pool2d")  # type: ignore[misc]
def quantized_avg_pool2d_meta(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    ceil_mode: bool,
    zero_point: int,
    multiplier: int,
    shift: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    kernel = _ensure_tuple2(kernel_size)
    stride_vals = _ensure_tuple2(stride)
    padding_vals = _ensure_tuple2(padding)
    output = F.avg_pool2d(
        input.to(torch.float),
        kernel,
        stride=stride_vals,
        padding=padding_vals,
        ceil_mode=ceil_mode,
        count_include_pad=False,
    )
    return torch.empty(
        output.shape,
        dtype=torch.int8,
        device=input.device,
        memory_format=torch.channels_last,
    )


@impl(lib, "quantized_avg_pool2d", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_avg_pool2d_impl(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    ceil_mode: bool,
    zero_point: int,
    multiplier: int,
    shift: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    dequant_input = dequantize_per_tensor_cmsis(input, zero_point, multiplier, shift)

    kernel = _ensure_tuple2(kernel_size)
    stride_vals = _ensure_tuple2(stride)
    padding_vals = _ensure_tuple2(padding)

    # TODO: implement dilation != 1.
    result = F.avg_pool2d(
        dequant_input,
        kernel,
        stride=stride_vals,
        padding=padding_vals,
        ceil_mode=ceil_mode,
        count_include_pad=False,
    )
    result = quantize_per_tensor_cmsis(result, zero_point, multiplier, shift)
    output = torch.clamp(result, -128, 127)
    return output.to(torch.int8)


lib.define(
    "quantized_avg_pool2d_nhwc("
    "Tensor input, int[] kernel_size, int[] stride, int[] padding, "
    "bool ceil_mode, int zero_point, int multiplier, int shift, "
    "Tensor scratch) -> Tensor"
)
lib.define(
    "quantized_avg_pool2d_nhwc.out("
    "Tensor input, int[] kernel_size, int[] stride, int[] padding, "
    "bool ceil_mode, int zero_point, int multiplier, int shift, "
    "Tensor scratch, *, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_avg_pool2d_nhwc")  # type: ignore[misc]
@experimental(_EXPLICIT_LAYOUT_EXPERIMENTAL)  # type: ignore[misc]
def quantized_avg_pool2d_nhwc_meta(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    ceil_mode: bool,
    zero_point: int,
    multiplier: int,
    shift: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    nchw = quantized_avg_pool2d_meta(
        input.permute(0, 3, 1, 2),
        kernel_size,
        stride,
        padding,
        ceil_mode,
        zero_point,
        multiplier,
        shift,
        scratch,
    )
    return nchw.permute(0, 2, 3, 1).contiguous()


@impl(lib, "quantized_avg_pool2d_nhwc", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_avg_pool2d_nhwc_impl(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    ceil_mode: bool,
    zero_point: int,
    multiplier: int,
    shift: int,
    scratch: torch.Tensor,
) -> torch.Tensor:
    nchw = quantized_avg_pool2d_impl(
        input.permute(0, 3, 1, 2).contiguous(),
        kernel_size,
        stride,
        padding,
        ceil_mode,
        zero_point,
        multiplier,
        shift,
        scratch,
    )
    return nchw.permute(0, 2, 3, 1).contiguous()


# ===================================================================
# QUANTIZED MAX POOL2D OPERATION DEFINITION
# ===================================================================

lib.define(
    "quantized_max_pool2d("
    "Tensor input, "
    "int[] kernel_size, "
    "int[] stride, "
    "int[] padding, "
    "int[] dilation, "
    "bool ceil_mode, "
    "int input_zero_point, "
    "int output_zero_point, "
    "int activation_min, "
    "int activation_max"
    ") -> Tensor"
)

lib.define(
    "quantized_max_pool2d.out("
    "Tensor input, "
    "int[] kernel_size, "
    "int[] stride, "
    "int[] padding, "
    "int[] dilation, "
    "bool ceil_mode, "
    "int input_zero_point, "
    "int output_zero_point, "
    "int activation_min, "
    "int activation_max, "
    "*, Tensor(a!) out"
    ") -> Tensor(a!)"
)


def _ensure_tuple2(value: Sequence[int]) -> tuple[int, int]:
    if len(value) == 1:
        return (int(value[0]), int(value[0]))
    if len(value) != 2:
        raise RuntimeError(f"Expected length-2 sequence, got {value}")
    return (int(value[0]), int(value[1]))


def _compute_max_pool2d_output_shape(
    input_shape: torch.Size,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> torch.Size:
    batch = input_shape[0]
    channels = input_shape[1]
    in_height = input_shape[2]
    in_width = input_shape[3]

    kernel_height, kernel_width = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    out_height = (
        in_height + 2 * pad_h - dilation_h * (kernel_height - 1) - 1
    ) // stride_h + 1
    out_width = (
        in_width + 2 * pad_w - dilation_w * (kernel_width - 1) - 1
    ) // stride_w + 1
    return torch.Size([batch, channels, out_height, out_width])


@register_fake("cortex_m::quantized_max_pool2d")  # type: ignore[misc]
def quantized_max_pool2d_meta(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
    input_zero_point: int,
    output_zero_point: int,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    kernel = _ensure_tuple2(kernel_size)
    stride_vals = _ensure_tuple2(stride)
    padding_vals = _ensure_tuple2(padding)
    dilation_vals = _ensure_tuple2(dilation)

    output_shape = _compute_max_pool2d_output_shape(
        input.shape, kernel, stride_vals, padding_vals, dilation_vals
    )
    return torch.empty(
        output_shape,
        dtype=torch.int8,
        device=input.device,
        memory_format=torch.channels_last,
    )


@impl(lib, "quantized_max_pool2d", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_max_pool2d_impl(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
    input_zero_point: int,
    output_zero_point: int,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    if input.dim() != 4:
        raise RuntimeError("quantized_max_pool2d expects 4D input tensor")

    if input_zero_point != output_zero_point:
        raise RuntimeError(
            "quantized_max_pool2d expects matching input/output zero points"
        )

    kernel = _ensure_tuple2(kernel_size)
    stride_vals = _ensure_tuple2(stride)
    padding_vals = _ensure_tuple2(padding)
    dilation_vals = _ensure_tuple2(dilation)

    if dilation_vals != (1, 1):
        raise RuntimeError("quantized_max_pool2d only supports dilation == 1")
    if ceil_mode:
        raise RuntimeError("quantized_max_pool2d does not support ceil_mode=True")

    # aten's channels-last max_pool2d caps how large an image it will take, so
    # pool a contiguous copy instead. Pooling is layout-invariant, and the
    # return below puts the result back in channels-last either way.
    #
    # The cap: cpu_max_pool_channels_last buffers each window index in
    # vec::int_same_size_t<opmath_t> and guards it with
    # TORCH_CHECK(input_depth * input_height * input_width <= max), so int8
    # rejects any image with more than 127 spatial elements -- H*W, with
    # channels not counted. int16 hits the same wall at 32767, which a future
    # quantized_max_pool2d_s16 will need to handle the same way.
    #
    # .to(memory_format=...) rather than .contiguous(): for C == 1 the
    # channels-last strides also satisfy plain contiguity, so .contiguous()
    # returns the same tensor while aten still dispatches on the memory-format
    # hint and raises anyway.
    result = F.max_pool2d(
        input.to(memory_format=torch.contiguous_format),
        kernel,
        stride=stride_vals,
        padding=padding_vals,
        dilation=dilation_vals,
        ceil_mode=ceil_mode,
    )
    result = torch.clamp(result, activation_min, activation_max)
    return result.to(torch.int8).contiguous(memory_format=torch.channels_last)


lib.define(
    "quantized_max_pool2d_nhwc("
    "Tensor input, int[] kernel_size, int[] stride, int[] padding, "
    "int[] dilation, bool ceil_mode, int input_zero_point, "
    "int output_zero_point, int activation_min, int activation_max) -> Tensor"
)
lib.define(
    "quantized_max_pool2d_nhwc.out("
    "Tensor input, int[] kernel_size, int[] stride, int[] padding, "
    "int[] dilation, bool ceil_mode, int input_zero_point, "
    "int output_zero_point, int activation_min, int activation_max, "
    "*, Tensor(a!) out) -> Tensor(a!)"
)


@register_fake("cortex_m::quantized_max_pool2d_nhwc")  # type: ignore[misc]
@experimental(_EXPLICIT_LAYOUT_EXPERIMENTAL)  # type: ignore[misc]
def quantized_max_pool2d_nhwc_meta(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
    input_zero_point: int,
    output_zero_point: int,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    nchw = quantized_max_pool2d_meta(
        input.permute(0, 3, 1, 2),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        input_zero_point,
        output_zero_point,
        activation_min,
        activation_max,
    )
    return nchw.permute(0, 2, 3, 1).contiguous()


@impl(lib, "quantized_max_pool2d_nhwc", "CompositeExplicitAutograd")  # type: ignore[misc]
def quantized_max_pool2d_nhwc_impl(
    input: torch.Tensor,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    ceil_mode: bool,
    input_zero_point: int,
    output_zero_point: int,
    activation_min: int,
    activation_max: int,
) -> torch.Tensor:
    nchw = quantized_max_pool2d_impl(
        input.permute(0, 3, 1, 2).contiguous(),
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        input_zero_point,
        output_zero_point,
        activation_min,
        activation_max,
    )
    return nchw.permute(0, 2, 3, 1).contiguous()
