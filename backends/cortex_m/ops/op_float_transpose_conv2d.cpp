/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "cortex_m_ops_common.h"

namespace cortex_m {
namespace native {

namespace {

constexpr int64_t kConvTransposeDim = 4;

template <typename ScalarT>
bool validate_float_transpose_conv2d_arguments(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Tensor& output,
    const Int64ArrayRef& stride,
    const Int64ArrayRef& padding,
    const Int64ArrayRef& output_padding,
    const Int64ArrayRef& dilation) {
  const auto dtype = cmsis_float_scalar_type<ScalarT>();
  if (input.dim() != kConvTransposeDim || weight.dim() != kConvTransposeDim ||
      output.dim() != kConvTransposeDim) {
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

  if (stride.size() != 2 || padding.size() != 2 || output_padding.size() != 2 ||
      dilation.size() != 2) {
    ET_LOG(
        Error,
        "%s: stride, padding, output_padding, and dilation must have length 2",
        op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (bias.has_value() &&
      (bias->dim() != 1 || bias->size(0) != output.size(1))) {
    ET_LOG(
        Error,
        "%s: bias must have shape [out_channels] (expected %lld, got %lld)",
        op_name,
        static_cast<long long>(output.size(1)),
        static_cast<long long>(bias->dim() == 1 ? bias->size(0) : -1));
    context.fail(Error::InvalidArgument);
    return false;
  }

  const int64_t input_channels = input.size(1);
  const int64_t output_channels = output.size(1);
  if (weight.size(0) != output_channels) {
    ET_LOG(
        Error,
        "%s: weight output channels mismatch (%lld vs %lld)",
        op_name,
        static_cast<long long>(weight.size(0)),
        static_cast<long long>(output_channels));
    context.fail(Error::InvalidArgument);
    return false;
  }
  if (weight.size(3) != input_channels) {
    ET_LOG(
        Error,
        "%s: weight input channels mismatch (%lld vs %lld)",
        op_name,
        static_cast<long long>(weight.size(3)),
        static_cast<long long>(input_channels));
    context.fail(Error::InvalidArgument);
    return false;
  }

  return true;
}

template <typename ScalarT, typename ParamsT>
void fill_float_transpose_conv_params(
    ParamsT& conv_params,
    const Int64ArrayRef& stride,
    const Int64ArrayRef& padding,
    const Int64ArrayRef& output_padding,
    const Int64ArrayRef& dilation,
    double activation_min,
    double activation_max) {
  conv_params.stride.h = static_cast<int32_t>(stride[0]);
  conv_params.stride.w = static_cast<int32_t>(stride[1]);
  conv_params.padding.h = static_cast<int32_t>(padding[0]);
  conv_params.padding.w = static_cast<int32_t>(padding[1]);
  conv_params.padding_offsets.h = static_cast<int32_t>(output_padding[0]);
  conv_params.padding_offsets.w = static_cast<int32_t>(output_padding[1]);
  conv_params.dilation.h = static_cast<int32_t>(dilation[0]);
  conv_params.dilation.w = static_cast<int32_t>(dilation[1]);
  conv_params.activation.min = static_cast<ScalarT>(activation_min);
  conv_params.activation.max = static_cast<ScalarT>(activation_max);
}

template <typename ScalarT, typename ParamsT>
Tensor& transpose_conv2d_out_impl(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef output_padding,
    const Int64ArrayRef dilation,
    double activation_min,
    double activation_max,
    Tensor& out,
    arm_cmsis_nn_status (*cmsis_fn)(
        const cmsis_nn_context*,
        const cmsis_nn_context*,
        const ParamsT*,
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
        const ParamsT*,
        const cmsis_nn_dims*,
        const cmsis_nn_dims*,
        const cmsis_nn_dims*),
    int32_t (*reverse_buffer_size_fn)(
        const ParamsT*,
        const cmsis_nn_dims*,
        const cmsis_nn_dims*)) {
  if constexpr (!is_cmsis_float_enabled<ScalarT>()) {
    fail_cmsis_float_kernel_unavailable(
        context, op_name, cmsis_float_dtype_name<ScalarT>());
    return out;
  }

  if (!validate_float_transpose_conv2d_arguments<ScalarT>(
          context,
          op_name,
          input,
          weight,
          bias,
          out,
          stride,
          padding,
          output_padding,
          dilation)) {
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
          context,
          op_name,
          weight.size(0),
          "weight out channels",
          kernel_out_channels) ||
      !check_int32_within_range(
          context, op_name, weight.size(1), "weight height", kernel_height) ||
      !check_int32_within_range(
          context, op_name, weight.size(2), "weight width", kernel_width) ||
      !check_int32_within_range(
          context,
          op_name,
          weight.size(3),
          "weight in channels",
          kernel_in_channels) ||
      !check_int32_within_range(
          context, op_name, out.size(1), "output channels", output_channels) ||
      !check_int32_within_range(
          context, op_name, out.size(2), "output height", output_height) ||
      !check_int32_within_range(
          context, op_name, out.size(3), "output width", output_width)) {
    return out;
  }

  const cmsis_nn_dims input_dims{
      batch, input_height, input_width, input_channels};
  const cmsis_nn_dims filter_dims{
      kernel_out_channels, kernel_height, kernel_width, kernel_in_channels};
  const cmsis_nn_dims output_dims{
      batch, output_height, output_width, output_channels};
  const cmsis_nn_dims bias_dims{1, 1, 1, output_channels};

  ParamsT conv_params{};
  fill_float_transpose_conv_params<ScalarT>(
      conv_params,
      stride,
      padding,
      output_padding,
      dilation,
      activation_min,
      activation_max);

  cmsis_nn_context cmsis_context{nullptr, 0};
  cmsis_nn_context output_context{nullptr, 0};

  const int32_t buffer_bytes =
      buffer_size_fn(&conv_params, &input_dims, &filter_dims, &output_dims);
  if (buffer_bytes > 0) {
    auto buffer_or_error = context.allocate_temp(
        static_cast<size_t>(buffer_bytes), kCortexMMveAlignment);
    if (!buffer_or_error.ok()) {
      ET_LOG(
          Error,
          "%s: failed to allocate scratch buffer (%d bytes, error %d)",
          op_name,
          buffer_bytes,
          static_cast<int>(buffer_or_error.error()));
      context.fail(buffer_or_error.error());
      return out;
    }
    cmsis_context.buf = buffer_or_error.get();
    cmsis_context.size = buffer_bytes;
  }

  const int32_t output_buffer_bytes =
      reverse_buffer_size_fn(&conv_params, &input_dims, &filter_dims);
  if (output_buffer_bytes > 0) {
    auto output_buffer_or_error = context.allocate_temp(
        static_cast<size_t>(output_buffer_bytes), kCortexMMveAlignment);
    if (!output_buffer_or_error.ok()) {
      ET_LOG(
          Error,
          "%s: failed to allocate output scratch buffer (%d bytes, error %d)",
          op_name,
          output_buffer_bytes,
          static_cast<int>(output_buffer_or_error.error()));
      context.fail(output_buffer_or_error.error());
      return out;
    }
    output_context.buf = output_buffer_or_error.get();
    output_context.size = output_buffer_bytes;
  }

  const arm_cmsis_nn_status status = cmsis_fn(
      &cmsis_context,
      &output_context,
      &conv_params,
      &input_dims,
      cmsis_const_data_ptr<ScalarT>(input),
      &filter_dims,
      cmsis_const_data_ptr<ScalarT>(weight),
      &bias_dims,
      bias.has_value() ? cmsis_const_data_ptr<ScalarT>(bias.value()) : nullptr,
      &output_dims,
      cmsis_mutable_data_ptr<ScalarT>(out),
      ARM_NN_LAYOUT_NHWC);

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "%s: arm_transpose_conv_wrapper failed with status %d",
        op_name,
        static_cast<int>(status));
    context.fail(Error::Internal);
  }

  return out;
}

} // namespace

Tensor& transpose_conv2d_f32_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef output_padding,
    const Int64ArrayRef dilation,
    double activation_min,
    double activation_max,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32)
  (void)input;
  (void)weight;
  (void)bias;
  (void)stride;
  (void)padding;
  (void)output_padding;
  (void)dilation;
  (void)activation_min;
  (void)activation_max;
  fail_cmsis_float_kernel_unavailable(
      context, "transpose_conv2d_f32_out", "float32");
  return out;
#else
  return transpose_conv2d_out_impl<
      float32_t,
      cmsis_nn_transpose_conv_params_f32>(
      context,
      "transpose_conv2d_f32_out",
      input,
      weight,
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      activation_min,
      activation_max,
      out,
      arm_transpose_conv_wrapper_f32,
      arm_transpose_conv_f32_get_buffer_size,
      arm_transpose_conv_f32_get_reverse_conv_buffer_size);
#endif
}

Tensor& transpose_conv2d_f16_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef output_padding,
    const Int64ArrayRef dilation,
    double activation_min,
    double activation_max,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16)
  (void)input;
  (void)weight;
  (void)bias;
  (void)stride;
  (void)padding;
  (void)output_padding;
  (void)dilation;
  (void)activation_min;
  (void)activation_max;
  fail_cmsis_float_kernel_unavailable(
      context, "transpose_conv2d_f16_out", "float16");
  return out;
#else
  return transpose_conv2d_out_impl<
      float16_t,
      cmsis_nn_transpose_conv_params_f16>(
      context,
      "transpose_conv2d_f16_out",
      input,
      weight,
      bias,
      stride,
      padding,
      output_padding,
      dilation,
      activation_min,
      activation_max,
      out,
      arm_transpose_conv_wrapper_f16,
      arm_transpose_conv_f16_get_buffer_size,
      arm_transpose_conv_f16_get_reverse_conv_buffer_size);
#endif
}

} // namespace native
} // namespace cortex_m
