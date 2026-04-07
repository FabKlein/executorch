/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "cortex_m_ops_common.h"

#include <cstring>

namespace cortex_m {
namespace native {

namespace {

template <typename ScalarT, typename ParamsT>
Tensor& svdf_out_impl(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Tensor& initial_state,
    const Tensor& weights_feature,
    const Tensor& weights_time,
    const Tensor& bias,
    bool time_major,
    int64_t rank,
    double input_activation_min,
    double input_activation_max,
    double output_activation_min,
    double output_activation_max,
    Tensor& out,
    arm_cmsis_nn_status (*cmsis_fn)(
        const cmsis_nn_context*,
        const cmsis_nn_context*,
        const cmsis_nn_context*,
        const ParamsT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        ScalarT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        ScalarT*)) {
  if constexpr (!is_cmsis_float_enabled<ScalarT>()) {
    fail_cmsis_float_kernel_unavailable(
        context, op_name, cmsis_float_dtype_name<ScalarT>());
    return out;
  }

  const ScalarType dtype = cmsis_float_scalar_type<ScalarT>();
  if (input.scalar_type() != dtype || initial_state.scalar_type() != dtype ||
      weights_feature.scalar_type() != dtype ||
      weights_time.scalar_type() != dtype || bias.scalar_type() != dtype ||
      out.scalar_type() != dtype) {
    ET_LOG(
        Error,
        "%s: all tensors must be %s",
        op_name,
        cmsis_float_dtype_name<ScalarT>());
    context.fail(Error::InvalidArgument);
    return out;
  }

  if (input.dim() != 3 || initial_state.dim() != 3 ||
      weights_feature.dim() != 2 || weights_time.dim() != 2 ||
      bias.dim() != 1 || out.dim() != 2) {
    ET_LOG(
        Error,
        "%s: expected input/state rank-3, weights rank-2, bias rank-1, out rank-2",
        op_name);
    context.fail(Error::InvalidArgument);
    return out;
  }

  int32_t time_steps = 0;
  int32_t batch_size = 0;
  int32_t input_size = 0;
  if (!check_int32_within_range(
          context,
          op_name,
          input.size(time_major ? 0 : 1),
          "time steps",
          time_steps) ||
      !check_int32_within_range(
          context,
          op_name,
          input.size(time_major ? 1 : 0),
          "batch size",
          batch_size) ||
      !check_int32_within_range(
          context, op_name, input.size(2), "input size", input_size)) {
    return out;
  }

  int32_t feature_batches = 0;
  int32_t time_batches = 0;
  int32_t unit_count = 0;
  int32_t rank_i32 = 0;
  if (!check_int32_within_range(
          context,
          op_name,
          weights_feature.size(0),
          "feature batches",
          feature_batches) ||
      !check_int32_within_range(
          context,
          op_name,
          weights_time.size(1),
          "time batches",
          time_batches) ||
      !check_int32_within_range(
          context, op_name, bias.size(0), "unit count", unit_count) ||
      !check_int32_within_range(context, op_name, rank, "rank", rank_i32)) {
    return out;
  }

  if (weights_feature.size(1) != input_size ||
      weights_time.size(0) != feature_batches ||
      initial_state.size(0) != batch_size ||
      initial_state.size(1) != feature_batches ||
      initial_state.size(2) != time_batches) {
    ET_LOG(Error, "%s: tensor shape mismatch", op_name);
    context.fail(Error::InvalidArgument);
    return out;
  }

  if (rank_i32 <= 0 || feature_batches % rank_i32 != 0 ||
      feature_batches / rank_i32 != unit_count) {
    ET_LOG(
        Error,
        "%s: invalid rank/feature_batches/unit_count combination",
        op_name);
    context.fail(Error::InvalidArgument);
    return out;
  }

  if (out.size(0) != batch_size || out.size(1) != unit_count) {
    ET_LOG(Error, "%s: output shape mismatch", op_name);
    context.fail(Error::InvalidArgument);
    return out;
  }

  const cmsis_nn_dims input_dims{batch_size, input_size, 1, 1};
  const cmsis_nn_dims state_dims{batch_size, feature_batches, time_batches, 1};
  const cmsis_nn_dims weights_feature_dims{feature_batches, input_size, 1, 1};
  const cmsis_nn_dims weights_time_dims{feature_batches, time_batches, 1, 1};
  const cmsis_nn_dims bias_dims{unit_count, 1, 1, 1};
  const cmsis_nn_dims output_dims{batch_size, unit_count, 1, 1};

  ParamsT params{};
  params.rank = rank_i32;
  params.input_activation.min = static_cast<ScalarT>(input_activation_min);
  params.input_activation.max = static_cast<ScalarT>(input_activation_max);
  params.output_activation.min = static_cast<ScalarT>(output_activation_min);
  params.output_activation.max = static_cast<ScalarT>(output_activation_max);

  const size_t state_bytes = static_cast<size_t>(initial_state.nbytes());
  auto state_or_error =
      context.allocate_temp(state_bytes, kCortexMMveAlignment);
  if (!state_or_error.ok()) {
    ET_LOG(
        Error,
        "%s: failed to allocate state buffer (%zu bytes, error %d)",
        op_name,
        state_bytes,
        static_cast<int>(state_or_error.error()));
    context.fail(state_or_error.error());
    return out;
  }
  auto* state_ptr = static_cast<ScalarT*>(state_or_error.get());
  std::memcpy(
      state_ptr, cmsis_const_data_ptr<ScalarT>(initial_state), state_bytes);

  const size_t input_ctx_bytes = static_cast<size_t>(batch_size) *
      static_cast<size_t>(feature_batches) * sizeof(ScalarT);
  auto input_ctx_or_error =
      context.allocate_temp(input_ctx_bytes, kCortexMMveAlignment);
  if (!input_ctx_or_error.ok()) {
    ET_LOG(
        Error,
        "%s: failed to allocate input scratch (%zu bytes, error %d)",
        op_name,
        input_ctx_bytes,
        static_cast<int>(input_ctx_or_error.error()));
    context.fail(input_ctx_or_error.error());
    return out;
  }

  const size_t output_ctx_bytes = static_cast<size_t>(batch_size) *
      static_cast<size_t>(unit_count) * sizeof(ScalarT);
  auto output_ctx_or_error =
      context.allocate_temp(output_ctx_bytes, kCortexMMveAlignment);
  if (!output_ctx_or_error.ok()) {
    ET_LOG(
        Error,
        "%s: failed to allocate output scratch (%zu bytes, error %d)",
        op_name,
        output_ctx_bytes,
        static_cast<int>(output_ctx_or_error.error()));
    context.fail(output_ctx_or_error.error());
    return out;
  }

  const size_t step_input_bytes = static_cast<size_t>(batch_size) *
      static_cast<size_t>(input_size) * sizeof(ScalarT);
  auto step_input_or_error =
      context.allocate_temp(step_input_bytes, kCortexMMveAlignment);
  if (!step_input_or_error.ok()) {
    ET_LOG(
        Error,
        "%s: failed to allocate step input buffer (%zu bytes, error %d)",
        op_name,
        step_input_bytes,
        static_cast<int>(step_input_or_error.error()));
    context.fail(step_input_or_error.error());
    return out;
  }
  auto* step_input_ptr = static_cast<ScalarT*>(step_input_or_error.get());

  const ScalarT* input_ptr = cmsis_const_data_ptr<ScalarT>(input);
  const ScalarT* weight_feature_ptr =
      cmsis_const_data_ptr<ScalarT>(weights_feature);
  const ScalarT* weight_time_ptr = cmsis_const_data_ptr<ScalarT>(weights_time);
  const ScalarT* bias_ptr = cmsis_const_data_ptr<ScalarT>(bias);
  ScalarT* out_ptr = cmsis_mutable_data_ptr<ScalarT>(out);

  cmsis_nn_context ctx{nullptr, 0};
  cmsis_nn_context input_ctx{
      input_ctx_or_error.get(), static_cast<int32_t>(input_ctx_bytes)};
  cmsis_nn_context output_ctx{
      output_ctx_or_error.get(), static_cast<int32_t>(output_ctx_bytes)};

  const int64_t input_batch_stride =
      static_cast<int64_t>(time_steps) * input_size;
  const int64_t input_step_stride =
      static_cast<int64_t>(batch_size) * input_size;

  for (int32_t step = 0; step < time_steps; ++step) {
    const ScalarT* step_ptr = nullptr;
    if (time_major) {
      step_ptr = input_ptr + static_cast<int64_t>(step) * input_step_stride;
    } else {
      for (int32_t batch = 0; batch < batch_size; ++batch) {
        const ScalarT* src = input_ptr +
            static_cast<int64_t>(batch) * input_batch_stride +
            static_cast<int64_t>(step) * input_size;
        std::memcpy(
            step_input_ptr + static_cast<int64_t>(batch) * input_size,
            src,
            static_cast<size_t>(input_size) * sizeof(ScalarT));
      }
      step_ptr = step_input_ptr;
    }

    const arm_cmsis_nn_status status = cmsis_fn(
        &ctx,
        &input_ctx,
        &output_ctx,
        &params,
        &input_dims,
        step_ptr,
        &state_dims,
        state_ptr,
        &weights_feature_dims,
        weight_feature_ptr,
        &weights_time_dims,
        weight_time_ptr,
        &bias_dims,
        bias_ptr,
        &output_dims,
        out_ptr);
    if (status != ARM_CMSIS_NN_SUCCESS) {
      ET_LOG(
          Error,
          "%s: arm_svdf failed with status %d at step %d",
          op_name,
          static_cast<int>(status),
          step);
      context.fail(Error::Internal);
      return out;
    }
  }

  return out;
}

} // namespace

Tensor& svdf_f32_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& initial_state,
    const Tensor& weights_feature,
    const Tensor& weights_time,
    const Tensor& bias,
    bool time_major,
    int64_t rank,
    double input_activation_min,
    double input_activation_max,
    double output_activation_min,
    double output_activation_max,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32)
  (void)input;
  (void)initial_state;
  (void)weights_feature;
  (void)weights_time;
  (void)bias;
  (void)time_major;
  (void)rank;
  (void)input_activation_min;
  (void)input_activation_max;
  (void)output_activation_min;
  (void)output_activation_max;
  fail_cmsis_float_kernel_unavailable(context, "svdf_f32_out", "float32");
  return out;
#else
  return svdf_out_impl<float32_t, cmsis_nn_svdf_params_f32>(
      context,
      "svdf_f32_out",
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
      out,
      arm_svdf_f32);
#endif
}

Tensor& svdf_f16_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& initial_state,
    const Tensor& weights_feature,
    const Tensor& weights_time,
    const Tensor& bias,
    bool time_major,
    int64_t rank,
    double input_activation_min,
    double input_activation_max,
    double output_activation_min,
    double output_activation_max,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16)
  (void)input;
  (void)initial_state;
  (void)weights_feature;
  (void)weights_time;
  (void)bias;
  (void)time_major;
  (void)rank;
  (void)input_activation_min;
  (void)input_activation_max;
  (void)output_activation_min;
  (void)output_activation_max;
  fail_cmsis_float_kernel_unavailable(context, "svdf_f16_out", "float16");
  return out;
#else
  return svdf_out_impl<float16_t, cmsis_nn_svdf_params_f16>(
      context,
      "svdf_f16_out",
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
      out,
      arm_svdf_f16);
#endif
}

} // namespace native
} // namespace cortex_m
