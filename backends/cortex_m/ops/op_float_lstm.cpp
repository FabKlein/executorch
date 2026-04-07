/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "cortex_m_ops_common.h"

#include <type_traits>

namespace cortex_m {
namespace native {

namespace {

template <
    typename ScalarT,
    typename LstmParamsT,
    typename LstmContextT,
    typename GateParamsT>
bool prepare_lstm_config(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Tensor& forget_input_weights,
    const Tensor& forget_hidden_weights,
    const Tensor& forget_bias,
    const Tensor& input_input_weights,
    const Tensor& input_hidden_weights,
    const Tensor& input_bias,
    const Tensor& cell_input_weights,
    const Tensor& cell_hidden_weights,
    const Tensor& cell_bias,
    const Tensor& output_input_weights,
    const Tensor& output_hidden_weights,
    const Tensor& output_bias,
    bool time_major,
    double cell_clip,
    Tensor& out,
    LstmParamsT& params,
    LstmContextT& buffers,
    int32_t& scratch_size) {
  const ScalarType expected_dtype =
      std::is_same_v<ScalarT, float32_t> ? ScalarType::Float : ScalarType::Half;
  const char* expected_dtype_name =
      expected_dtype == ScalarType::Float ? "float32" : "float16";

  auto check_gate = [&](const char* gate_name,
                        const Tensor& input_weights,
                        const Tensor& hidden_weights,
                        const Tensor& bias,
                        int32_t input_size,
                        int32_t hidden_size) -> bool {
    if (input_weights.scalar_type() != expected_dtype ||
        hidden_weights.scalar_type() != expected_dtype ||
        bias.scalar_type() != expected_dtype) {
      ET_LOG(
          Error,
          "%s: %s tensors must all be %s",
          op_name,
          gate_name,
          expected_dtype_name);
      context.fail(Error::InvalidArgument);
      return false;
    }
    if (input_weights.dim() != 2 || hidden_weights.dim() != 2 ||
        bias.dim() != 1) {
      ET_LOG(
          Error,
          "%s: %s expects rank-2 weights and rank-1 bias",
          op_name,
          gate_name);
      context.fail(Error::InvalidArgument);
      return false;
    }
    if (input_weights.size(0) != hidden_size ||
        input_weights.size(1) != input_size) {
      ET_LOG(Error, "%s: %s input weight shape mismatch", op_name, gate_name);
      context.fail(Error::InvalidArgument);
      return false;
    }
    if (hidden_weights.size(0) != hidden_size ||
        hidden_weights.size(1) != hidden_size) {
      ET_LOG(Error, "%s: %s hidden weight shape mismatch", op_name, gate_name);
      context.fail(Error::InvalidArgument);
      return false;
    }
    if (bias.size(0) != hidden_size) {
      ET_LOG(Error, "%s: %s bias shape mismatch", op_name, gate_name);
      context.fail(Error::InvalidArgument);
      return false;
    }
    return true;
  };

  if (input.scalar_type() != expected_dtype ||
      out.scalar_type() != expected_dtype) {
    ET_LOG(Error, "%s: input/output must be %s", op_name, expected_dtype_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (input.dim() != 3 || out.dim() != 3) {
    ET_LOG(Error, "%s: input/output must be rank-3", op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  int32_t time_steps = 0;
  int32_t batch_size = 0;
  int32_t input_size = 0;
  int32_t hidden_size = 0;
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
          context, op_name, input.size(2), "input size", input_size) ||
      !check_int32_within_range(
          context, op_name, forget_bias.size(0), "hidden size", hidden_size)) {
    return false;
  }

  if (!check_gate(
          "forget_gate",
          forget_input_weights,
          forget_hidden_weights,
          forget_bias,
          input_size,
          hidden_size) ||
      !check_gate(
          "input_gate",
          input_input_weights,
          input_hidden_weights,
          input_bias,
          input_size,
          hidden_size) ||
      !check_gate(
          "cell_gate",
          cell_input_weights,
          cell_hidden_weights,
          cell_bias,
          input_size,
          hidden_size) ||
      !check_gate(
          "output_gate",
          output_input_weights,
          output_hidden_weights,
          output_bias,
          input_size,
          hidden_size)) {
    return false;
  }

  if ((time_major &&
       (out.size(0) != time_steps || out.size(1) != batch_size ||
        out.size(2) != hidden_size)) ||
      (!time_major &&
       (out.size(0) != batch_size || out.size(1) != time_steps ||
        out.size(2) != hidden_size))) {
    ET_LOG(Error, "%s: output shape mismatch", op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  scratch_size = batch_size * hidden_size;
  params.time_major = time_major ? 1 : 0;
  params.batch_size = batch_size;
  params.time_steps = time_steps;
  params.input_size = input_size;
  params.hidden_size = hidden_size;
  params.cell_clip = static_cast<ScalarT>(cell_clip);

  auto fill_gate = [](GateParamsT& gate,
                      const Tensor& input_weights,
                      const Tensor& hidden_weights,
                      const Tensor& bias,
                      arm_nn_activation_type_flt activation) {
    gate.input_weights =
        reinterpret_cast<const ScalarT*>(input_weights.const_data_ptr());
    gate.hidden_weights =
        reinterpret_cast<const ScalarT*>(hidden_weights.const_data_ptr());
    gate.bias = reinterpret_cast<const ScalarT*>(bias.const_data_ptr());
    gate.activation_type = activation;
  };

  fill_gate(
      params.forget_gate,
      forget_input_weights,
      forget_hidden_weights,
      forget_bias,
      ARM_NN_FLT_ACT_SIGMOID);
  fill_gate(
      params.input_gate,
      input_input_weights,
      input_hidden_weights,
      input_bias,
      ARM_NN_FLT_ACT_SIGMOID);
  fill_gate(
      params.cell_gate,
      cell_input_weights,
      cell_hidden_weights,
      cell_bias,
      ARM_NN_FLT_ACT_TANH);
  fill_gate(
      params.output_gate,
      output_input_weights,
      output_hidden_weights,
      output_bias,
      ARM_NN_FLT_ACT_SIGMOID);

  buffers.temp1 = nullptr;
  buffers.temp2 = nullptr;
  buffers.cell_state = nullptr;
  return true;
}

template <
    typename ScalarT,
    typename LstmParamsT,
    typename LstmContextT,
    typename GateParamsT>
Tensor& lstm_out_impl(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Tensor& forget_input_weights,
    const Tensor& forget_hidden_weights,
    const Tensor& forget_bias,
    const Tensor& input_input_weights,
    const Tensor& input_hidden_weights,
    const Tensor& input_bias,
    const Tensor& cell_input_weights,
    const Tensor& cell_hidden_weights,
    const Tensor& cell_bias,
    const Tensor& output_input_weights,
    const Tensor& output_hidden_weights,
    const Tensor& output_bias,
    bool time_major,
    double cell_clip,
    Tensor& out,
    arm_cmsis_nn_status (*cmsis_fn)(
        const ScalarT*,
        ScalarT*,
        const LstmParamsT*,
        LstmContextT*)) {
  if constexpr (std::is_same_v<ScalarT, float32_t>) {
#if !(defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32)
    fail_cmsis_float_kernel_unavailable(context, op_name, "float32");
    return out;
#endif
  } else {
#if !(defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16)
    fail_cmsis_float_kernel_unavailable(context, op_name, "float16");
    return out;
#endif
  }

  LstmParamsT params{};
  LstmContextT buffers{};
  int32_t scratch_size = 0;
  if (!prepare_lstm_config<ScalarT, LstmParamsT, LstmContextT, GateParamsT>(
          context,
          op_name,
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
          out,
          params,
          buffers,
          scratch_size)) {
    return out;
  }

  const int32_t scratch_bytes =
      scratch_size * static_cast<int32_t>(sizeof(ScalarT));
  if (scratch_bytes > 0) {
    auto temp1_or_error = context.allocate_temp(scratch_bytes);
    if (!temp1_or_error.ok()) {
      ET_LOG(Error, "%s: failed to allocate temp1", op_name);
      context.fail(Error::MemoryAllocationFailed);
      return out;
    }
    auto temp2_or_error = context.allocate_temp(scratch_bytes);
    if (!temp2_or_error.ok()) {
      ET_LOG(Error, "%s: failed to allocate temp2", op_name);
      context.fail(Error::MemoryAllocationFailed);
      return out;
    }
    auto state_or_error = context.allocate_temp(scratch_bytes);
    if (!state_or_error.ok()) {
      ET_LOG(Error, "%s: failed to allocate cell_state", op_name);
      context.fail(Error::MemoryAllocationFailed);
      return out;
    }
    buffers.temp1 = reinterpret_cast<ScalarT*>(temp1_or_error.get());
    buffers.temp2 = reinterpret_cast<ScalarT*>(temp2_or_error.get());
    buffers.cell_state = reinterpret_cast<ScalarT*>(state_or_error.get());
  }

  const arm_cmsis_nn_status status = cmsis_fn(
      reinterpret_cast<const ScalarT*>(input.const_data_ptr()),
      reinterpret_cast<ScalarT*>(out.mutable_data_ptr()),
      &params,
      &buffers);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "%s: CMSIS-NN LSTM failed with status %d",
        op_name,
        static_cast<int>(status));
    context.fail(Error::Internal);
  }
  return out;
}

} // namespace

Tensor& lstm_unidirectional_f32_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& forget_input_weights,
    const Tensor& forget_hidden_weights,
    const Tensor& forget_bias,
    const Tensor& input_input_weights,
    const Tensor& input_hidden_weights,
    const Tensor& input_bias,
    const Tensor& cell_input_weights,
    const Tensor& cell_hidden_weights,
    const Tensor& cell_bias,
    const Tensor& output_input_weights,
    const Tensor& output_hidden_weights,
    const Tensor& output_bias,
    bool time_major,
    double cell_clip,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32)
  (void)input;
  (void)forget_input_weights;
  (void)forget_hidden_weights;
  (void)forget_bias;
  (void)input_input_weights;
  (void)input_hidden_weights;
  (void)input_bias;
  (void)cell_input_weights;
  (void)cell_hidden_weights;
  (void)cell_bias;
  (void)output_input_weights;
  (void)output_hidden_weights;
  (void)output_bias;
  (void)time_major;
  (void)cell_clip;
  fail_cmsis_float_kernel_unavailable(
      context, "lstm_unidirectional_f32_out", "float32");
  return out;
#else
  return lstm_out_impl<
      float32_t,
      cmsis_nn_lstm_params_f32,
      cmsis_nn_lstm_context_f32,
      cmsis_nn_lstm_gate_f32>(
      context,
      "lstm_unidirectional_f32_out",
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
      out,
      arm_lstm_unidirectional_f32);
#endif
}

Tensor& lstm_unidirectional_f16_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& forget_input_weights,
    const Tensor& forget_hidden_weights,
    const Tensor& forget_bias,
    const Tensor& input_input_weights,
    const Tensor& input_hidden_weights,
    const Tensor& input_bias,
    const Tensor& cell_input_weights,
    const Tensor& cell_hidden_weights,
    const Tensor& cell_bias,
    const Tensor& output_input_weights,
    const Tensor& output_hidden_weights,
    const Tensor& output_bias,
    bool time_major,
    double cell_clip,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16)
  (void)input;
  (void)forget_input_weights;
  (void)forget_hidden_weights;
  (void)forget_bias;
  (void)input_input_weights;
  (void)input_hidden_weights;
  (void)input_bias;
  (void)cell_input_weights;
  (void)cell_hidden_weights;
  (void)cell_bias;
  (void)output_input_weights;
  (void)output_hidden_weights;
  (void)output_bias;
  (void)time_major;
  (void)cell_clip;
  fail_cmsis_float_kernel_unavailable(
      context, "lstm_unidirectional_f16_out", "float16");
  return out;
#else
  return lstm_out_impl<
      float16_t,
      cmsis_nn_lstm_params_f16,
      cmsis_nn_lstm_context_f16,
      cmsis_nn_lstm_gate_f16>(
      context,
      "lstm_unidirectional_f16_out",
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
      out,
      arm_lstm_unidirectional_f16);
#endif
}

} // namespace native
} // namespace cortex_m
