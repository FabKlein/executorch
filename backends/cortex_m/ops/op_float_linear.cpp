/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "cortex_m_ops_common.h"

#include <limits>

namespace cortex_m {
namespace native {

namespace {

template <typename ScalarT, typename FcParamsT>
inline void fill_float_fc_params(
    FcParamsT& fc_params,
    float activation_min,
    float activation_max) {
  fc_params.activation.min = static_cast<ScalarT>(activation_min);
  fc_params.activation.max = static_cast<ScalarT>(activation_max);
}

template <typename ScalarT>
bool prepare_float_linear_config(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Tensor& weights,
    const torch::executor::optional<Tensor>& bias,
    bool weight_is_packed,
    int64_t packed_out_features,
    Tensor& output,
    cmsis_nn_dims& input_dims,
    cmsis_nn_dims& filter_dims,
    cmsis_nn_dims& bias_dims,
    cmsis_nn_dims& output_dims,
    int32_t& batches,
    int32_t& in_features,
    int32_t& out_features) {
  // Linear supports two logical input shapes here:
  //
  //   1) generic matrix-style input
  //        input  = [..., in_features]
  //        output = [..., out_features]
  //
  //   2) NHWC feature map feeding a fully connected layer directly
  //        input  = [N, C, H, W]   (logical NHWC metadata in ET/CMSIS dims)
  //        output = [N, out_features]
  //
  // We also accept two weight encodings:
  //   standard: rank-2 [out_features, in_features]
  //   packed  : rank-1 flat buffer + packed_out_features metadata
  const ScalarType expected_dtype =
      std::is_same_v<ScalarT, float32_t> ? ScalarType::Float : ScalarType::Half;
  const char* expected_dtype_name =
      expected_dtype == ScalarType::Float ? "float32" : "float16";

  if (input.scalar_type() != expected_dtype ||
      weights.scalar_type() != expected_dtype ||
      output.scalar_type() != expected_dtype ||
      (bias.has_value() && bias->scalar_type() != expected_dtype)) {
    ET_LOG(
        Error,
        "%s: input/weights/bias/output must all be %s",
        op_name,
        expected_dtype_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (input.dim() < 1 || output.dim() < 1 ||
      (!weight_is_packed && weights.dim() != 2) ||
      (weight_is_packed && weights.dim() != 1)) {
    ET_LOG(
        Error,
        "%s: expected input/output rank >= 1 and %s weights",
        op_name,
        weight_is_packed ? "rank-1 packed" : "rank-2");
    context.fail(Error::InvalidArgument);
    return false;
  }
  int32_t weight_in_features = 0;
  if (weight_is_packed) {
    // Packed linear weights no longer have the normal rank-2 shape that would
    // tell us {out_features, in_features}. Lowering therefore passes
    // out_features separately, and we recover in_features from the packed byte
    // count.
    //
    // Logical matrix before packing:
    //
    //   standard weight [O, I]
    //        O rows = output features
    //        I cols = input features
    //
    // Packed storage groups O rows into CMSIS-NN lane blocks:
    //
    //   block 0, k=0: row0[k0] row1[k0] ... rowN[k0]
    //   block 0, k=1: row0[k1] row1[k1] ... rowN[k1]
    //   ...
    //   block 1, k=0: rowN+1[k0] ...
    //
    // The flat tensor size is:
    //
    //   I * ceil(O / block_cols) * block_cols
    //
    // where block_cols is 8 for f16 and 4 for f32.
    if (!check_int32_within_range(
            context,
            op_name,
            packed_out_features,
            "packed output features",
            out_features)) {
      return false;
    }
    if (out_features <= 0) {
      ET_LOG(Error, "%s: packed output features must be > 0", op_name);
      context.fail(Error::InvalidArgument);
      return false;
    }
    const int32_t packed_block_cols = get_cmsis_packed_n_block_cols<ScalarT>();
    const int64_t packed_block_count64 =
        (static_cast<int64_t>(out_features) + packed_block_cols - 1) /
        packed_block_cols;
    const int64_t packed_block_stride64 =
        packed_block_count64 * packed_block_cols;
    if (packed_block_stride64 <= 0 ||
        weights.numel() % packed_block_stride64 != 0) {
      ET_LOG(
          Error,
          "%s: invalid packed linear weight buffer size=%lld for out_features=%d",
          op_name,
          static_cast<long long>(weights.numel()),
          out_features);
      context.fail(Error::InvalidArgument);
      return false;
    }
    if (!check_int32_within_range(
            context,
            op_name,
            weights.numel() / packed_block_stride64,
            "packed weight input features",
            weight_in_features)) {
      return false;
    }
  } else {
    if (!check_int32_within_range(
            context,
            op_name,
            weights.size(0),
            "output features",
            out_features) ||
        !check_int32_within_range(
            context,
            op_name,
            weights.size(1),
            "weight input features",
            weight_in_features)) {
      return false;
    }
  }

  const bool nhwc_input_to_matrix_output = input.dim() == 4 &&
      output.dim() == 2 &&
      (static_cast<int64_t>(input.size(1)) * input.size(2) * input.size(3) ==
       weight_in_features);

  if (nhwc_input_to_matrix_output) {
    // Fast path used when a channels-last feature map feeds an FC directly.
    // Instead of materializing an explicit flatten buffer first, we describe
    // the input with NHWC-like dims and let CMSIS consume it that way.
    //
    //   input feature map       logical flattened matrix
    //   [N, C, H, W]      ->    [N, C*H*W]
    //        |
    //        | encoded as CMSIS dims
    //        v
    //   input_dims = {N, H, W, C}
    //
    // This path is only valid when C*H*W matches the recovered/declared weight
    // input feature count. Otherwise a 4D tensor could be accidentally treated
    // as FC input even though its layout does not match the weight matrix.
    int32_t input_channels = 0;
    int32_t input_height = 0;
    int32_t input_width = 0;
    if (!check_int32_within_range(
            context, op_name, input.size(0), "batches", batches) ||
        !check_int32_within_range(
            context,
            op_name,
            input.size(1),
            "input channels",
            input_channels) ||
        !check_int32_within_range(
            context, op_name, input.size(2), "input height", input_height) ||
        !check_int32_within_range(
            context, op_name, input.size(3), "input width", input_width)) {
      return false;
    }

    const int64_t input_features64 =
        static_cast<int64_t>(input_channels) * input_height * input_width;
    if (!check_int32_within_range(
            context,
            op_name,
            input_features64,
            "input features",
            in_features)) {
      return false;
    }
    if (!weight_is_packed && weight_in_features != in_features) {
      ET_LOG(
          Error,
          "%s: weights/input feature mismatch (weight=%d, input=%d)",
          op_name,
          weight_in_features,
          in_features);
      context.fail(Error::InvalidArgument);
      return false;
    }

    if (output.size(0) != batches || output.size(1) != out_features) {
      ET_LOG(
          Error,
          "%s: output shape mismatch for NHWC FC path (expected=[%d,%d], got=[%lld,%lld])",
          op_name,
          batches,
          out_features,
          static_cast<long long>(output.size(0)),
          static_cast<long long>(output.size(1)));
      context.fail(Error::InvalidArgument);
      return false;
    }

    input_dims =
        cmsis_nn_dims{batches, input_height, input_width, input_channels};
    filter_dims = cmsis_nn_dims{in_features, 1, 1, out_features};
    bias_dims = cmsis_nn_dims{1, 1, 1, out_features};
    output_dims = cmsis_nn_dims{batches, 1, 1, out_features};
  } else {
    // Generic matrix-style path:
    //
    //   [..., in_features] x [out_features, in_features]^T
    //          ->
    //   [..., out_features]
    if (!check_int32_within_range(
            context,
            op_name,
            input.size(input.dim() - 1),
            "input features",
            in_features)) {
      return false;
    }

    if (!weight_is_packed && weight_in_features != in_features) {
      ET_LOG(
          Error,
          "%s: weights/input feature mismatch (weight=%d, input=%d)",
          op_name,
          weight_in_features,
          in_features);
      context.fail(Error::InvalidArgument);
      return false;
    }

    if (input.dim() != output.dim()) {
      ET_LOG(Error, "%s: input/output rank mismatch", op_name);
      context.fail(Error::InvalidArgument);
      return false;
    }

    int64_t batches64 = 1;
    for (size_t dim = 0; dim + 1 < input.dim(); ++dim) {
      if (input.size(dim) != output.size(dim)) {
        ET_LOG(
            Error,
            "%s: leading dimension %zu mismatch (input=%lld, out=%lld)",
            op_name,
            static_cast<size_t>(dim),
            static_cast<long long>(input.size(dim)),
            static_cast<long long>(output.size(dim)));
        context.fail(Error::InvalidArgument);
        return false;
      }
      batches64 *= input.size(dim);
    }
    if (!check_int32_within_range(
            context, op_name, batches64, "batches", batches)) {
      return false;
    }

    input_dims = cmsis_nn_dims{batches, 1, 1, in_features};
    filter_dims = cmsis_nn_dims{in_features, 1, 1, out_features};
    bias_dims = cmsis_nn_dims{1, 1, 1, out_features};
    output_dims = cmsis_nn_dims{batches, 1, 1, out_features};
  }

  if (bias.has_value()) {
    if (bias->dim() != 1) {
      ET_LOG(Error, "%s: bias must be rank-1", op_name);
      context.fail(Error::InvalidArgument);
      return false;
    }
    int32_t bias_size = 0;
    if (!check_int32_within_range(
            context, op_name, bias->size(0), "bias size", bias_size)) {
      return false;
    }
    if (bias_size != out_features) {
      ET_LOG(
          Error,
          "%s: bias/output feature mismatch (bias=%d, out=%d)",
          op_name,
          bias_size,
          out_features);
      context.fail(Error::InvalidArgument);
      return false;
    }
  }

  if (output.size(output.dim() - 1) != out_features) {
    ET_LOG(
        Error,
        "%s: output feature dim mismatch (expected=%d, got=%lld)",
        op_name,
        out_features,
        static_cast<long long>(output.size(output.dim() - 1)));
    context.fail(Error::InvalidArgument);
    return false;
  }
  return true;
}

template <typename ScalarT, typename FcParamsT>
Tensor& linear_out_impl(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Tensor& weights,
    const torch::executor::optional<Tensor>& bias,
    bool weight_is_packed,
    int64_t packed_out_features,
    double activation_min,
    double activation_max,
    Tensor& out,
    arm_cmsis_nn_status (*cmsis_fn)(
        const cmsis_nn_context*,
        const FcParamsT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        ScalarT*,
        arm_nn_tensor_layout),
    int32_t (*buffer_size_fn)(
        const FcParamsT*,
        const cmsis_nn_dims*,
        const cmsis_nn_dims*,
        const cmsis_nn_dims*,
        arm_nn_tensor_layout)) {
  cmsis_nn_dims input_dims;
  cmsis_nn_dims filter_dims;
  cmsis_nn_dims bias_dims;
  cmsis_nn_dims output_dims;
  int32_t batches = 0;
  int32_t in_features = 0;
  int32_t out_features = 0;
  if (!prepare_float_linear_config<ScalarT>(
          context,
          op_name,
          input,
          weights,
          bias,
          weight_is_packed,
          packed_out_features,
          out,
          input_dims,
          filter_dims,
          bias_dims,
          output_dims,
          batches,
          in_features,
          out_features)) {
    return out;
  }

  FcParamsT fc_params;
  fill_float_fc_params<ScalarT>(
      fc_params,
      static_cast<float>(activation_min),
      static_cast<float>(activation_max));
  // Start from the standard weight format. The packed branch below flips this
  // to NT_N_PACKED only when we already have a packed constant buffer, or when
  // we build a temporary packed copy at runtime.
  set_standard_cmsis_weight_format(fc_params);

  const auto layout = ARM_NN_LAYOUT_NHWC;
  const int32_t buffer_size = buffer_size_fn(
      &fc_params, &input_dims, &filter_dims, &output_dims, layout);
  cmsis_nn_context cmsis_ctx{nullptr, 0};
  if (buffer_size > 0) {
    auto buffer_or_error = context.allocate_temp(buffer_size);
    if (!buffer_or_error.ok()) {
      ET_LOG(
          Error,
          "%s: failed to allocate %d scratch bytes",
          op_name,
          buffer_size);
      context.fail(Error::MemoryAllocationFailed);
      return out;
    }
    cmsis_ctx.buf = buffer_or_error.get();
    cmsis_ctx.size = buffer_size;
  }

  const ScalarT* bias_ptr = bias.has_value()
      ? reinterpret_cast<const ScalarT*>(bias->const_data_ptr())
      : nullptr;
  const ScalarT* weight_ptr =
      reinterpret_cast<const ScalarT*>(weights.const_data_ptr());
  if (weight_is_packed) {
    // Offline-packed path:
    //
    //   standard constant [O, I]
    //        --(Cortex-M lowering pass)-->
    //   packed flat buffer [block][k][lane]
    //
    // Runtime can use the packed bytes directly with no extra conversion.
    set_packed_cmsis_weight_format(fc_params);
  } else if constexpr (has_cmsis_weight_format<FcParamsT>::value) {
    // Fallback path for unpacked weights:
    //
    //   standard weight tensor              temporary packed buffer
    //   [out_features, in_features]   ->    [block][k][lane]
    //
    // Example with f16 block_cols=8:
    //
    //   W rows 0..7, cols 0..2:
    //
    //     r0: a0 a1 a2
    //     r1: b0 b1 b2
    //     ...
    //     r7: h0 h1 h2
    //
    //   packed:
    //
    //     k0: a0 b0 c0 d0 e0 f0 g0 h0
    //     k1: a1 b1 c1 d1 e1 f1 g1 h1
    //     k2: a2 b2 c2 d2 e2 f2 g2 h2
    //
    // This keeps compatibility with older exported programs, but it is slower
    // than the offline-packed constant path above because the conversion is
    // paid during inference.
    const size_t packed_bytes =
        get_nt_n_packed_weight_bytes<ScalarT>(out_features, in_features);
    auto packed_or_error =
        context.allocate_temp(packed_bytes, kCortexMMveAlignment);
    if (!packed_or_error.ok()) {
      ET_LOG(
          Error,
          "%s: failed to allocate %zu packed-weight bytes",
          op_name,
          packed_bytes);
      context.fail(Error::MemoryAllocationFailed);
      return out;
    }
    pack_nt_t_weights_to_nt_n_packed(
        weight_ptr, out_features, in_features, (ScalarT*)packed_or_error.get());
    weight_ptr = static_cast<const ScalarT*>(packed_or_error.get());
    set_packed_cmsis_weight_format(fc_params);
  }
  const arm_cmsis_nn_status status = cmsis_fn(
      &cmsis_ctx,
      &fc_params,
      &input_dims,
      reinterpret_cast<const ScalarT*>(input.const_data_ptr()),
      &filter_dims,
      weight_ptr,
      &bias_dims,
      bias_ptr,
      &output_dims,
      reinterpret_cast<ScalarT*>(out.mutable_data_ptr()),
      layout);

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "%s: CMSIS-NN failed with status [%d]",
        op_name,
        static_cast<int>(status));
    context.fail(Error::Internal);
  }
  return out;
}

} // namespace

Tensor& linear_f32_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weights,
    const torch::executor::optional<Tensor>& bias,
    bool weight_is_packed,
    int64_t packed_out_features,
    double activation_min,
    double activation_max,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32)
  (void)input;
  (void)weights;
  (void)bias;
  (void)weight_is_packed;
  (void)packed_out_features;
  (void)activation_min;
  (void)activation_max;
  fail_cmsis_float_kernel_unavailable(context, "linear_f32_out", "float32");
  return out;
#else
  return linear_out_impl<float32_t, cmsis_nn_fc_params_f32>(
      context,
      "linear_f32_out",
      input,
      weights,
      bias,
      weight_is_packed,
      packed_out_features,
      activation_min,
      activation_max,
      out,
      arm_fully_connected_f32,
      arm_fully_connected_f32_get_buffer_size);
#endif
}

Tensor& linear_f16_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weights,
    const torch::executor::optional<Tensor>& bias,
    bool weight_is_packed,
    int64_t packed_out_features,
    double activation_min,
    double activation_max,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16)
  (void)input;
  (void)weights;
  (void)bias;
  (void)weight_is_packed;
  (void)packed_out_features;
  (void)activation_min;
  (void)activation_max;
  fail_cmsis_float_kernel_unavailable(context, "linear_f16_out", "float16");
  return out;
#else
  static_assert(
      sizeof(executorch::aten::Half) == sizeof(float16_t),
      "ExecuTorch Half and CMSIS float16_t must have identical storage");

  return linear_out_impl<float16_t, cmsis_nn_fc_params_f16>(
      context,
      "linear_f16_out",
      input,
      weights,
      bias,
      weight_is_packed,
      packed_out_features,
      activation_min,
      activation_max,
      out,
      arm_fully_connected_f16,
      arm_fully_connected_f16_get_buffer_size);
#endif
}

} // namespace native
} // namespace cortex_m
