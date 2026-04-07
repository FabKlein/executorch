/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "cortex_m_ops_common.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cortex_m {
namespace native {

namespace {

inline bool validate_activation_common(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Tensor& out,
    int32_t& size) {
  if (input.scalar_type() != out.scalar_type()) {
    ET_LOG(
        Error,
        "%s: input/output dtype mismatch (input=%d, out=%d)",
        op_name,
        static_cast<int>(input.scalar_type()),
        static_cast<int>(out.scalar_type()));
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (input.sizes() != out.sizes()) {
    ET_LOG(Error, "%s: input/output shapes must match", op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  const int64_t numel = input.numel();
  if (numel <= 0 || numel > std::numeric_limits<int32_t>::max()) {
    ET_LOG(
        Error,
        "%s: element count must fit in int32 (numel=%lld)",
        op_name,
        static_cast<long long>(numel));
    context.fail(Error::InvalidArgument);
    return false;
  }

  size = static_cast<int32_t>(numel);
  return true;
}

inline bool validate_activation_type(
    KernelRuntimeContext& context,
    const char* op_name,
    int64_t activation_type,
    int64_t& activation_type_out) {
  switch (activation_type) {
    case ARM_NN_FLT_ACT_NONE:
    case ARM_NN_FLT_ACT_SIGMOID:
    case ARM_NN_FLT_ACT_TANH:
    case ARM_NN_FLT_ACT_RELU:
    case ARM_NN_FLT_ACT_RELU6:
    case ARM_NN_FLT_ACT_HARDSWISH:
    case ARM_NN_FLT_ACT_LEAKY_RELU:
    // Allow custom activation types that are implemented locally below instead
    // of by CMSIS-NN.
    case kCortexMFloatActHardsigmoid:
    case kCortexMFloatActHardtanh:
      activation_type_out = activation_type;
      return true;
    default:
      ET_LOG(
          Error,
          "%s: unsupported activation type %lld",
          op_name,
          static_cast<long long>(activation_type));
      context.fail(Error::InvalidArgument);
      return false;
  }
}

template <typename ETScalarT>
Tensor& activation_out_impl(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    int64_t activation_type,
    double act_param,
    Tensor& out) {
  int32_t size = 0;
  if (!validate_activation_common(context, op_name, input, out, size)) {
    return out;
  }

  int64_t activation_type_checked = 0;
  if (!validate_activation_type(
          context, op_name, activation_type, activation_type_checked)) {
    return out;
  }

  arm_cmsis_nn_status status = ARM_CMSIS_NN_SUCCESS;

  constexpr float kInvSix = 1.0f / 6.0f;
  auto run_hardsigmoid = [&](const auto* input_ptr, auto* output_ptr) {
    using ScalarT = std::remove_pointer_t<decltype(input_ptr)>;
    for (int32_t i = 0; i < size; ++i) {
      const float x = static_cast<float>(input_ptr[i]);
      const float y = std::min(std::max(x + 3.0f, 0.0f), 6.0f) * kInvSix;
      output_ptr[i] = static_cast<ScalarT>(y);
    }
  };
  auto run_hardtanh = [&](const auto* input_ptr, auto* output_ptr) {
    using ScalarT = std::remove_pointer_t<decltype(input_ptr)>;
    for (int32_t i = 0; i < size; ++i) {
      const float x = static_cast<float>(input_ptr[i]);
      output_ptr[i] = static_cast<ScalarT>(std::min(std::max(x, -1.0f), 1.0f));
    }
  };

  if constexpr (std::is_same_v<ETScalarT, float>) {
#if !(defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32)
    (void)input;
    (void)act_param;
    fail_cmsis_float_kernel_unavailable(context, op_name, "float32");
    return out;
#else
    if (activation_type_checked == kCortexMFloatActHardsigmoid) {
      run_hardsigmoid(
          input.const_data_ptr<float>(), out.mutable_data_ptr<float>());
    } else if (activation_type_checked == kCortexMFloatActHardtanh) {
      run_hardtanh(
          input.const_data_ptr<float>(), out.mutable_data_ptr<float>());
    } else {
      status = arm_nn_activation_f32(
          input.const_data_ptr<float>(),
          out.mutable_data_ptr<float>(),
          size,
          static_cast<arm_nn_activation_type_flt>(activation_type_checked),
          static_cast<float32_t>(act_param));
    }
#endif
  } else {
#if !(defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16)
    (void)input;
    (void)act_param;
    fail_cmsis_float_kernel_unavailable(context, op_name, "float16");
    return out;
#else
    if (activation_type_checked == kCortexMFloatActHardsigmoid) {
      run_hardsigmoid(
          reinterpret_cast<const float16_t*>(
              input.const_data_ptr<executorch::aten::Half>()),
          reinterpret_cast<float16_t*>(
              out.mutable_data_ptr<executorch::aten::Half>()));
    } else if (activation_type_checked == kCortexMFloatActHardtanh) {
      run_hardtanh(
          reinterpret_cast<const float16_t*>(
              input.const_data_ptr<executorch::aten::Half>()),
          reinterpret_cast<float16_t*>(
              out.mutable_data_ptr<executorch::aten::Half>()));
    } else {
      status = arm_nn_activation_f16(
          reinterpret_cast<const float16_t*>(
              input.const_data_ptr<executorch::aten::Half>()),
          reinterpret_cast<float16_t*>(
              out.mutable_data_ptr<executorch::aten::Half>()),
          size,
          static_cast<arm_nn_activation_type_flt>(activation_type_checked),
          static_cast<float16_t>(act_param));
    }
#endif
  }

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "%s: CMSIS-NN activation failed with status %d",
        op_name,
        static_cast<int>(status));
    context.fail(Error::Internal);
  }

  return out;
}

} // namespace

Tensor& activation_f32_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    int64_t activation_type,
    double act_param,
    Tensor& out) {
  return activation_out_impl<float>(
      context, "activation_f32_out", input, activation_type, act_param, out);
}

Tensor& activation_f16_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    int64_t activation_type,
    double act_param,
    Tensor& out) {
  return activation_out_impl<executorch::aten::Half>(
      context, "activation_f16_out", input, activation_type, act_param, out);
}

} // namespace native
} // namespace cortex_m
