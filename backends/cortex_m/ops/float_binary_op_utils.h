/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "cortex_m_ops_common.h"

#include <type_traits>

namespace cortex_m {
namespace native {

template <typename ETScalarT, typename CmsisScalarT>
struct FloatTensorAccess;

template <>
struct FloatTensorAccess<float, float> {
  static constexpr ScalarType kScalarType = ScalarType::Float;

  static const float* input_ptr(const Tensor& tensor) {
    return tensor.const_data_ptr<float>();
  }

  static float* output_ptr(Tensor& tensor) {
    return tensor.mutable_data_ptr<float>();
  }
};

#if defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16
template <>
struct FloatTensorAccess<executorch::aten::Half, float16_t> {
  static constexpr ScalarType kScalarType = ScalarType::Half;

  static const float16_t* input_ptr(const Tensor& tensor) {
    static_assert(
        sizeof(executorch::aten::Half) == sizeof(float16_t),
        "ExecuTorch Half and CMSIS float16_t must have identical storage");
    return reinterpret_cast<const float16_t*>(
        tensor.const_data_ptr<executorch::aten::Half>());
  }

  static float16_t* output_ptr(Tensor& tensor) {
    static_assert(
        sizeof(executorch::aten::Half) == sizeof(float16_t),
        "ExecuTorch Half and CMSIS float16_t must have identical storage");
    return reinterpret_cast<float16_t*>(
        tensor.mutable_data_ptr<executorch::aten::Half>());
  }
};
#endif

template <typename ETScalarT>
constexpr bool is_cmsis_float_dtype_enabled();

template <>
constexpr bool is_cmsis_float_dtype_enabled<float>() {
  return kCmsisFloat32Enabled;
}

template <>
constexpr bool is_cmsis_float_dtype_enabled<executorch::aten::Half>() {
  return kCmsisFloat16Enabled;
}

template <
    typename ETScalarT,
    typename CmsisScalarT,
    arm_cmsis_nn_status (*CmsisFn)(
        const CmsisScalarT*,
        const CmsisScalarT*,
        CmsisScalarT*,
        CmsisScalarT,
        CmsisScalarT,
        int32_t)>
Tensor& run_float_binary_out(
    KernelRuntimeContext& context,
    const Tensor& input1,
    const Tensor& input2,
    const double activation_min,
    const double activation_max,
    Tensor& out,
    const char* op_name,
    const char* cmsis_name) {
  if constexpr (!is_cmsis_float_dtype_enabled<ETScalarT>()) {
    fail_cmsis_float_kernel_unavailable(
        context,
        op_name,
        std::is_same_v<ETScalarT, float> ? "float32" : "float16");
    return out;
  }

  const bool channel_broadcast = is_channel_broadcast(input1, input2);
  const ScalarType expected_dtype =
      FloatTensorAccess<ETScalarT, CmsisScalarT>::kScalarType;
  if (input1.scalar_type() != expected_dtype ||
      input2.scalar_type() != expected_dtype ||
      out.scalar_type() != expected_dtype) {
    ET_LOG(
        Error,
        "%s: input and output must have dtype %d",
        op_name,
        static_cast<int>(expected_dtype));
    context.fail(Error::InvalidArgument);
    return out;
  }
  if (!channel_broadcast &&
      (input1.sizes() != input2.sizes() || out.sizes() != input1.sizes())) {
    ET_LOG(
        Error, "%s: non-broadcast inputs must have identical sizes", op_name);
    context.fail(Error::InvalidArgument);
    return out;
  }

  if (activation_min > activation_max) {
    ET_LOG(
        Error,
        "%s: activation_min (%f) must be <= activation_max (%f)",
        op_name,
        activation_min,
        activation_max);
    context.fail(Error::InvalidArgument);
    return out;
  }

  const CmsisScalarT* input1_ptr =
      FloatTensorAccess<ETScalarT, CmsisScalarT>::input_ptr(input1);
  const CmsisScalarT* input2_ptr =
      FloatTensorAccess<ETScalarT, CmsisScalarT>::input_ptr(input2);
  CmsisScalarT* output_ptr =
      FloatTensorAccess<ETScalarT, CmsisScalarT>::output_ptr(out);

  const int32_t elems_per_loop = channel_broadcast
      ? static_cast<int32_t>(input1.size(1))
      : static_cast<int32_t>(out.numel());

  if (channel_broadcast && input1.numel() < input2.numel()) {
    std::swap(input1_ptr, input2_ptr);
  }

  for (int32_t offset = 0; offset < out.numel(); offset += elems_per_loop) {
    const arm_cmsis_nn_status status = CmsisFn(
        input1_ptr + offset,
        input2_ptr,
        output_ptr + offset,
        static_cast<CmsisScalarT>(activation_min),
        static_cast<CmsisScalarT>(activation_max),
        elems_per_loop);

    if (status != ARM_CMSIS_NN_SUCCESS) {
      ET_LOG(
          Error,
          "%s: %s failed with status [%d]",
          op_name,
          cmsis_name,
          static_cast<int>(status));
      context.fail(Error::Internal);
      return out;
    }
  }

  return out;
}

} // namespace native
} // namespace cortex_m
