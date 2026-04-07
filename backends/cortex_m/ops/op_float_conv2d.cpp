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

constexpr int64_t kConvDim = 4;

template <typename ScalarT>
bool validate_float_conv2d_arguments(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Tensor& output,
    const Int64ArrayRef& stride,
    const Int64ArrayRef& padding,
    const Int64ArrayRef& dilation,
    bool weight_is_packed,
    int64_t packed_output_channels,
    int64_t packed_kernel_height,
    int64_t packed_kernel_width,
    int64_t packed_kernel_input_channels,
    bool depthwise,
    int64_t depth_multiplier) {
  // Standard conv weights stay in OHWI form:
  //   [out_channels, kernel_height, kernel_width, in_channels]
  //
  // Offline-packed conv weights are rewritten earlier in the Cortex-M passes
  // and stored as a flat packed buffer:
  //   [block][k][lane]
  //
  // The packed tensor itself is rank-1, so the original logical conv shape is
  // carried separately through the packed_* metadata arguments.
  const auto dtype = cmsis_float_scalar_type<ScalarT>();
  if (input.dim() != kConvDim ||
      (!weight_is_packed && weight.dim() != kConvDim) ||
      (weight_is_packed && weight.dim() != 1) || output.dim() != kConvDim) {
    ET_LOG(Error, "%s: tensors must be 4-D", op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (input.scalar_type() != dtype || weight.scalar_type() != dtype ||
      output.scalar_type() != dtype ||
      (bias.has_value() && bias->scalar_type() != dtype)) {
    ET_LOG(
        Error,
        "%s: input/weight/bias/output must all be %s",
        op_name,
        cmsis_float_dtype_name<ScalarT>());
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (!is_channels_last_tensor(input) || !is_channels_last_tensor(output)) {
    ET_LOG(Error, "%s: input and output must be channels_last", op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (stride.size() != 2 || padding.size() != 2 || dilation.size() != 2) {
    ET_LOG(
        Error, "%s: stride, padding, and dilation must have length 2", op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (bias.has_value() &&
      (bias->dim() != 1 ||
       bias->size(0) !=
           (weight_is_packed ? packed_output_channels : output.size(1)))) {
    ET_LOG(
        Error,
        "%s: bias must have shape [out_channels] (expected %lld, got %lld)",
        op_name,
        static_cast<long long>(output.size(1)),
        static_cast<long long>(bias->dim() == 1 ? bias->size(0) : -1));
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (depthwise) {
    if (weight.size(0) != 1) {
      ET_LOG(Error, "%s: depthwise weight dim0 must be 1", op_name);
      context.fail(Error::InvalidArgument);
      return false;
    }

    const int64_t input_channels = input.size(1);
    const int64_t output_channels = output.size(1);
    if (output_channels != input_channels * depth_multiplier) {
      ET_LOG(
          Error,
          "%s: out_channels (%lld) must equal in_channels (%lld) * depth_multiplier (%lld)",
          op_name,
          static_cast<long long>(output_channels),
          static_cast<long long>(input_channels),
          static_cast<long long>(depth_multiplier));
      context.fail(Error::InvalidArgument);
      return false;
    }

    if (weight.size(3) != output_channels) {
      ET_LOG(
          Error,
          "%s: depthwise weight output channels mismatch (%lld vs %lld)",
          op_name,
          static_cast<long long>(weight.size(3)),
          static_cast<long long>(output_channels));
      context.fail(Error::InvalidArgument);
      return false;
    }
  } else {
    const int64_t input_channels = input.size(1);
    const int64_t output_channels = output.size(1);
    if (weight_is_packed) {
      if (packed_output_channels != output_channels) {
        ET_LOG(
            Error,
            "%s: packed output channels mismatch (%lld vs %lld)",
            op_name,
            static_cast<long long>(packed_output_channels),
            static_cast<long long>(output_channels));
        context.fail(Error::InvalidArgument);
        return false;
      }
      if (packed_kernel_height <= 0 || packed_kernel_width <= 0 ||
          packed_kernel_input_channels <= 0) {
        ET_LOG(
            Error,
            "%s: packed kernel metadata must be > 0 (got kh=%lld kw=%lld in=%lld)",
            op_name,
            static_cast<long long>(packed_kernel_height),
            static_cast<long long>(packed_kernel_width),
            static_cast<long long>(packed_kernel_input_channels));
        context.fail(Error::InvalidArgument);
        return false;
      }
      if (packed_kernel_input_channels != input_channels) {
        ET_LOG(
            Error,
            "%s: packed kernel input channels mismatch (%lld vs %lld)",
            op_name,
            static_cast<long long>(packed_kernel_input_channels),
            static_cast<long long>(input_channels));
        context.fail(Error::InvalidArgument);
        return false;
      }
    } else if (weight.size(0) != output_channels) {
      ET_LOG(
          Error,
          "%s: standard weight output channels mismatch (%lld vs %lld)",
          op_name,
          static_cast<long long>(weight.size(0)),
          static_cast<long long>(output_channels));
      context.fail(Error::InvalidArgument);
      return false;
    }
    if (!weight_is_packed && weight.size(3) != input_channels) {
      ET_LOG(
          Error,
          "%s: standard weight input channels mismatch (%lld vs %lld)",
          op_name,
          static_cast<long long>(weight.size(3)),
          static_cast<long long>(input_channels));
      context.fail(Error::InvalidArgument);
      return false;
    }
  }

  return true;
}

template <typename ScalarT, typename ConvParamsT>
void fill_float_conv_params(
    ConvParamsT& conv_params,
    const Int64ArrayRef& stride,
    const Int64ArrayRef& padding,
    const Int64ArrayRef& dilation,
    float activation_min,
    float activation_max) {
  conv_params.stride.h = static_cast<int32_t>(stride[0]);
  conv_params.stride.w = static_cast<int32_t>(stride[1]);
  conv_params.padding.h = static_cast<int32_t>(padding[0]);
  conv_params.padding.w = static_cast<int32_t>(padding[1]);
  conv_params.dilation.h = static_cast<int32_t>(dilation[0]);
  conv_params.dilation.w = static_cast<int32_t>(dilation[1]);
  conv_params.activation.min = static_cast<ScalarT>(activation_min);
  conv_params.activation.max = static_cast<ScalarT>(activation_max);
  // Start from the standard CMSIS weight layout. The packed runtime path below
  // upgrades this to NT_N_PACKED only when the weight buffer is already packed
  // offline, or when we explicitly repack it in a temporary buffer.
  set_standard_cmsis_weight_format(conv_params);
}

bool can_use_nt_n_packed_conv_weights(
    int32_t batch,
    int32_t input_height,
    int32_t kernel_height,
    int32_t kernel_width,
    int32_t input_channels,
    int32_t output_channels,
    int32_t output_height,
    int32_t output_width,
    const Int64ArrayRef& stride,
    const Int64ArrayRef& padding,
    const Int64ArrayRef& dilation) {
  // CMSIS-NN's float convolution family consumes prepacked weights in all
  // matmul-backed paths, but not in the final scalar/direct OHWI fallback:
  //
  //   1x1 conv                  -> arm_convolve_1x1_* -> packed matmul
  //   1x3/1x5 conv1d specials   -> packed conv1d helper
  //   generic KHxKW patch-GEMM  -> pack input patch -> packed matmul
  //   small fallback loop       -> standard OHWI only
  //
  // Only advertise packed weights when CMSIS is guaranteed to stay on one of
  // the packed-aware branches. Otherwise packed bytes would be consumed by the
  // direct OHWI loop and silently produce wrong numerics.
  //
  // Keep this predicate aligned with PackFloatConvWeightsPass. Dilation is
  // intentionally left out of the packed ET contract until we validate it on
  // real CMSIS-NN kernels.
  if (kernel_height <= 0 || kernel_width <= 0 || stride[0] <= 0 ||
      stride[1] <= 0 || padding[0] < 0 || padding[1] < 0 || dilation[0] != 1 ||
      dilation[1] != 1) {
    return false;
  }

  if (kernel_height == 1 && kernel_width == 1 && padding[0] == 0 &&
      padding[1] == 0) {
    return true;
  }

  if (batch == 1 && input_height == 1 && output_height == 1 &&
      kernel_height == 1 && (kernel_width == 3 || kernel_width == 5) &&
      stride[0] == 1 && stride[1] == 1 && padding[0] == 0 && padding[1] == 0) {
    return true;
  }

  const int32_t patch_len = kernel_height * kernel_width * input_channels;
  const int32_t output_positions = output_height * output_width;
  return patch_len >= 16 && output_channels >= 8 && output_positions >= 8;
}

template <typename ScalarT, typename ConvParamsT>
Tensor& conv2d_out_impl(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef dilation,
    bool weight_is_packed,
    int64_t packed_output_channels,
    int64_t packed_kernel_height,
    int64_t packed_kernel_width,
    int64_t packed_kernel_input_channels,
    double activation_min,
    double activation_max,
    Tensor& out,
    arm_cmsis_nn_status (*cmsis_fn)(
        const cmsis_nn_context*,
        const ConvParamsT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        ScalarT*),
    int32_t (*buffer_size_fn)(
        const ConvParamsT*,
        const cmsis_nn_dims*,
        const cmsis_nn_dims*,
        const cmsis_nn_dims*)) {
  if constexpr (!is_cmsis_float_enabled<ScalarT>()) {
    fail_cmsis_float_kernel_unavailable(
        context, op_name, cmsis_float_dtype_name<ScalarT>());
    return out;
  }

  if (!validate_float_conv2d_arguments<ScalarT>(
          context,
          op_name,
          input,
          weight,
          bias,
          out,
          stride,
          padding,
          dilation,
          weight_is_packed,
          packed_output_channels,
          packed_kernel_height,
          packed_kernel_width,
          packed_kernel_input_channels,
          /*depthwise=*/false,
          /*depth_multiplier=*/1)) {
    return out;
  }

  int32_t batch, input_channels, input_height, input_width, kernel_out_channels,
      kernel_height, kernel_width, kernel_in_channels, output_channels,
      output_height, output_width;
  if (!check_int32_within_range(
          context, op_name, input.size(0), "input batch", batch) ||
      !check_int32_within_range(
          context, op_name, input.size(1), "input channels", input_channels) ||
      !check_int32_within_range(
          context, op_name, input.size(2), "input height", input_height) ||
      !check_int32_within_range(
          context, op_name, input.size(3), "input width", input_width) ||
      !check_int32_within_range(
          context, op_name, out.size(1), "output channels", output_channels) ||
      !check_int32_within_range(
          context, op_name, out.size(2), "output height", output_height) ||
      !check_int32_within_range(
          context, op_name, out.size(3), "output width", output_width)) {
    return out;
  }

  if (weight_is_packed) {
    if (!check_int32_within_range(
            context,
            op_name,
            packed_output_channels,
            "packed output channels",
            kernel_out_channels)) {
      return out;
    }
    if (!check_int32_within_range(
            context,
            op_name,
            packed_kernel_height,
            "packed kernel height",
            kernel_height) ||
        !check_int32_within_range(
            context,
            op_name,
            packed_kernel_width,
            "packed kernel width",
            kernel_width) ||
        !check_int32_within_range(
            context,
            op_name,
            packed_kernel_input_channels,
            "packed kernel input channels",
            kernel_in_channels)) {
      return out;
    }
  } else if (
      !check_int32_within_range(
          context,
          op_name,
          weight.size(0),
          "weight out channels",
          kernel_out_channels) ||
      !check_int32_within_range(
          context, op_name, weight.size(1), "kernel height", kernel_height) ||
      !check_int32_within_range(
          context, op_name, weight.size(2), "kernel width", kernel_width) ||
      !check_int32_within_range(
          context,
          op_name,
          weight.size(3),
          "kernel in channels",
          kernel_in_channels)) {
    return out;
  }

  const cmsis_nn_dims input_dims{
      batch, input_height, input_width, input_channels};
  const cmsis_nn_dims filter_dims{
      kernel_out_channels, kernel_height, kernel_width, kernel_in_channels};
  const cmsis_nn_dims output_dims{
      batch, output_height, output_width, output_channels};
  const cmsis_nn_dims bias_dims{1, 1, 1, output_channels};

  // Logical conv view:
  //   input  : [N, H, W, C]  (channels_last)
  //   weight : [O, KH, KW, I]               standard
  //         or [flat packed buffer]         packed
  //   output : [N, OH, OW, O]
  //
  // Packed conv weights keep the same logical conv shape, but the actual bytes
  // are stored in CMSIS NT_N-packed order and the original {O, KH, KW, I}
  // dimensions are reconstructed from the explicit packed_* metadata above.

  ConvParamsT conv_params{};
  fill_float_conv_params<ScalarT>(
      conv_params,
      stride,
      padding,
      dilation,
      static_cast<float>(activation_min),
      static_cast<float>(activation_max));

  cmsis_nn_context cmsis_ctx{nullptr, 0};
  const int32_t buffer_size =
      buffer_size_fn(&conv_params, &input_dims, &filter_dims, &output_dims);
  if (buffer_size < 0) {
    ET_LOG(Error, "%s: CMSIS-NN buffer size calculation failed", op_name);
    context.fail(Error::Internal);
    return out;
  }
  if (buffer_size > 0) {
    auto buffer_or_error =
        context.allocate_temp(buffer_size, kCortexMMveAlignment);
    if (!buffer_or_error.ok()) {
      ET_LOG(
          Error,
          "%s: failed to allocate %d scratch bytes",
          op_name,
          buffer_size);
      context.fail(buffer_or_error.error());
      return out;
    }
    cmsis_ctx.buf = buffer_or_error.get();
    cmsis_ctx.size = buffer_size;
  }

  const ScalarT* bias_ptr =
      bias.has_value() ? cmsis_const_data_ptr<ScalarT>(*bias) : nullptr;
  const ScalarT* weight_ptr = cmsis_const_data_ptr<ScalarT>(weight);
  void* packed_weights = nullptr;
  const bool can_use_packed_weights =
      has_cmsis_weight_format<ConvParamsT>::value &&
      can_use_nt_n_packed_conv_weights(
          batch,
          input_height,
          kernel_height,
          kernel_width,
          input_channels,
          output_channels,
          output_height,
          output_width,
          stride,
          padding,
          dilation);
  if (weight_is_packed) {
    // Fast path: lowering already packed the constant conv weights offline and
    // stored them directly in the exported program.
    //
    //   OHWI constant --(offline pass)--> packed flat buffer
    //   runtime                        --> use packed bytes directly
    if (!can_use_packed_weights) {
      ET_LOG(
          Error,
          "%s: packed weights requested for unsupported convolution shape",
          op_name);
      context.fail(Error::InvalidArgument);
      return out;
    }
    set_packed_cmsis_weight_format(conv_params);
  } else if (can_use_packed_weights) {
    // Fallback path: weights still arrive in standard OHWI form.
    //
    // We flatten the logical conv kernel to [O, KH*KW*I], then pack that RHS
    // once into a temporary NT_N buffer before calling the CMSIS matmul-backed
    // conv kernel:
    //
    //   [O, KH, KW, I]
    //        |
    //        v
    //   [O, KH*KW*I]
    //        |
    //        v
    //   [block][k][lane]
    //
    // This is slower than the offline-packed path above, but still lets us use
    // the packed CMSIS inner kernel instead of the older NT_T variant.
    const int32_t flat_features =
        kernel_height * kernel_width * kernel_in_channels;
    const size_t packed_bytes =
        get_nt_n_packed_weight_bytes<ScalarT>(output_channels, flat_features);
    auto packed_or_error =
        context.allocate_temp(packed_bytes, kCortexMMveAlignment);
    if (!packed_or_error.ok()) {
      ET_LOG(
          Error,
          "%s: failed to allocate %zu packed-weight bytes",
          op_name,
          packed_bytes);
      context.fail(packed_or_error.error());
      return out;
    }
    packed_weights = packed_or_error.get();
    pack_nt_t_weights_to_nt_n_packed(
        weight_ptr, output_channels, flat_features, (ScalarT*)packed_weights);
    weight_ptr = static_cast<const ScalarT*>(packed_weights);
    set_packed_cmsis_weight_format(conv_params);
  }
  const arm_cmsis_nn_status status = cmsis_fn(
      &cmsis_ctx,
      &conv_params,
      &input_dims,
      cmsis_const_data_ptr<ScalarT>(input),
      &filter_dims,
      weight_ptr,
      &bias_dims,
      bias_ptr,
      &output_dims,
      cmsis_mutable_data_ptr<ScalarT>(out));
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

template <typename ScalarT, typename DwConvParamsT>
Tensor& depthwise_conv2d_out_impl(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef dilation,
    int64_t depth_multiplier,
    double activation_min,
    double activation_max,
    Tensor& out,
    arm_cmsis_nn_status (*cmsis_fn)(
        const cmsis_nn_context*,
        const DwConvParamsT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        ScalarT*),
    int32_t (*buffer_size_fn)(
        const DwConvParamsT*,
        const cmsis_nn_dims*,
        const cmsis_nn_dims*,
        const cmsis_nn_dims*)) {
  if constexpr (!is_cmsis_float_enabled<ScalarT>()) {
    fail_cmsis_float_kernel_unavailable(
        context, op_name, cmsis_float_dtype_name<ScalarT>());
    return out;
  }

  if (!validate_float_conv2d_arguments<ScalarT>(
          context,
          op_name,
          input,
          weight,
          bias,
          out,
          stride,
          padding,
          dilation,
          /*weight_is_packed=*/false,
          /*packed_output_channels=*/0,
          /*packed_kernel_height=*/0,
          /*packed_kernel_width=*/0,
          /*packed_kernel_input_channels=*/0,
          /*depthwise=*/true,
          depth_multiplier)) {
    return out;
  }

  int32_t batch, input_channels, input_height, input_width, kernel_height,
      kernel_width, output_channels, output_height, output_width,
      depth_multiplier_val;
  if (!check_int32_within_range(
          context, op_name, input.size(0), "input batch", batch) ||
      !check_int32_within_range(
          context, op_name, input.size(1), "input channels", input_channels) ||
      !check_int32_within_range(
          context, op_name, input.size(2), "input height", input_height) ||
      !check_int32_within_range(
          context, op_name, input.size(3), "input width", input_width) ||
      !check_int32_within_range(
          context, op_name, weight.size(1), "kernel height", kernel_height) ||
      !check_int32_within_range(
          context, op_name, weight.size(2), "kernel width", kernel_width) ||
      !check_int32_within_range(
          context, op_name, out.size(1), "output channels", output_channels) ||
      !check_int32_within_range(
          context, op_name, out.size(2), "output height", output_height) ||
      !check_int32_within_range(
          context, op_name, out.size(3), "output width", output_width) ||
      !check_int32_within_range(
          context,
          op_name,
          depth_multiplier,
          "depth multiplier",
          depth_multiplier_val)) {
    return out;
  }

  const cmsis_nn_dims input_dims{
      batch, input_height, input_width, input_channels};
  const cmsis_nn_dims filter_dims{
      1, kernel_height, kernel_width, output_channels};
  const cmsis_nn_dims output_dims{
      batch, output_height, output_width, output_channels};
  const cmsis_nn_dims bias_dims{1, 1, 1, output_channels};

  DwConvParamsT conv_params{};
  conv_params.ch_mult = depth_multiplier_val;
  fill_float_conv_params<ScalarT>(
      conv_params,
      stride,
      padding,
      dilation,
      static_cast<float>(activation_min),
      static_cast<float>(activation_max));

  cmsis_nn_context cmsis_ctx{nullptr, 0};
  const int32_t buffer_size =
      buffer_size_fn(&conv_params, &input_dims, &filter_dims, &output_dims);
  if (buffer_size < 0) {
    ET_LOG(Error, "%s: CMSIS-NN buffer size calculation failed", op_name);
    context.fail(Error::Internal);
    return out;
  }
  if (buffer_size > 0) {
    auto buffer_or_error =
        context.allocate_temp(buffer_size, kCortexMMveAlignment);
    if (!buffer_or_error.ok()) {
      ET_LOG(
          Error,
          "%s: failed to allocate %d scratch bytes",
          op_name,
          buffer_size);
      context.fail(buffer_or_error.error());
      return out;
    }
    cmsis_ctx.buf = buffer_or_error.get();
    cmsis_ctx.size = buffer_size;
  }

  const ScalarT* bias_ptr =
      bias.has_value() ? cmsis_const_data_ptr<ScalarT>(*bias) : nullptr;
  const arm_cmsis_nn_status status = cmsis_fn(
      &cmsis_ctx,
      &conv_params,
      &input_dims,
      cmsis_const_data_ptr<ScalarT>(input),
      &filter_dims,
      cmsis_const_data_ptr<ScalarT>(weight),
      &bias_dims,
      bias_ptr,
      &output_dims,
      cmsis_mutable_data_ptr<ScalarT>(out));
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

Tensor& conv2d_f32_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef dilation,
    bool weight_is_packed,
    int64_t packed_output_channels,
    int64_t packed_kernel_height,
    int64_t packed_kernel_width,
    int64_t packed_kernel_input_channels,
    double activation_min,
    double activation_max,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32)
  (void)input;
  (void)weight;
  (void)bias;
  (void)stride;
  (void)padding;
  (void)dilation;
  (void)weight_is_packed;
  (void)packed_output_channels;
  (void)packed_kernel_height;
  (void)packed_kernel_width;
  (void)packed_kernel_input_channels;
  (void)activation_min;
  (void)activation_max;
  fail_cmsis_float_kernel_unavailable(context, "conv2d_f32_out", "float32");
  return out;
#else
  return conv2d_out_impl<float32_t, cmsis_nn_conv_params_f32>(
      context,
      "conv2d_f32_out",
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
      out,
      arm_convolve_wrapper_f32,
      arm_convolve_wrapper_f32_get_buffer_size);
#endif
}

Tensor& conv2d_f16_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef dilation,
    bool weight_is_packed,
    int64_t packed_output_channels,
    int64_t packed_kernel_height,
    int64_t packed_kernel_width,
    int64_t packed_kernel_input_channels,
    double activation_min,
    double activation_max,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16)
  (void)input;
  (void)weight;
  (void)bias;
  (void)stride;
  (void)padding;
  (void)dilation;
  (void)weight_is_packed;
  (void)packed_output_channels;
  (void)packed_kernel_height;
  (void)packed_kernel_width;
  (void)packed_kernel_input_channels;
  (void)activation_min;
  (void)activation_max;
  fail_cmsis_float_kernel_unavailable(context, "conv2d_f16_out", "float16");
  return out;
#else
  return conv2d_out_impl<float16_t, cmsis_nn_conv_params_f16>(
      context,
      "conv2d_f16_out",
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
      out,
      arm_convolve_wrapper_f16,
      arm_convolve_wrapper_f16_get_buffer_size);
#endif
}

Tensor& depthwise_conv2d_f32_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef dilation,
    int64_t depth_multiplier,
    double activation_min,
    double activation_max,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32)
  (void)input;
  (void)weight;
  (void)bias;
  (void)stride;
  (void)padding;
  (void)dilation;
  (void)depth_multiplier;
  (void)activation_min;
  (void)activation_max;
  fail_cmsis_float_kernel_unavailable(
      context, "depthwise_conv2d_f32_out", "float32");
  return out;
#else
  return depthwise_conv2d_out_impl<float32_t, cmsis_nn_dw_conv_params_f32>(
      context,
      "depthwise_conv2d_f32_out",
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      depth_multiplier,
      activation_min,
      activation_max,
      out,
      arm_depthwise_conv_wrapper_f32,
      arm_depthwise_conv_wrapper_f32_get_buffer_size);
#endif
}

Tensor& depthwise_conv2d_f16_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef dilation,
    int64_t depth_multiplier,
    double activation_min,
    double activation_max,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16)
  (void)input;
  (void)weight;
  (void)bias;
  (void)stride;
  (void)padding;
  (void)dilation;
  (void)depth_multiplier;
  (void)activation_min;
  (void)activation_max;
  fail_cmsis_float_kernel_unavailable(
      context, "depthwise_conv2d_f16_out", "float16");
  return out;
#else
  return depthwise_conv2d_out_impl<float16_t, cmsis_nn_dw_conv_params_f16>(
      context,
      "depthwise_conv2d_f16_out",
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      depth_multiplier,
      activation_min,
      activation_max,
      out,
      arm_depthwise_conv_wrapper_f16,
      arm_depthwise_conv_wrapper_f16_get_buffer_size);
#endif
}

} // namespace native
} // namespace cortex_m
