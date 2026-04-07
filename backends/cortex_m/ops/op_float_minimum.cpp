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

arm_cmsis_nn_status float_minimum_impl(
    const Tensor& input1,
    const cmsis_nn_dims& input1_dims,
    const Tensor& input2,
    const cmsis_nn_dims& input2_dims,
    Tensor& out,
    const cmsis_nn_dims& output_dims) {
#if defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32
  if (input1.scalar_type() == ScalarType::Float) {
    const float* input1_data = input1.const_data_ptr<float>();
    const float* input2_data = input2.const_data_ptr<float>();
    float* output_data = out.mutable_data_ptr<float>();
    return arm_minimum_f32(
        nullptr,
        input1_data,
        &input1_dims,
        input2_data,
        &input2_dims,
        output_data,
        &output_dims);
  }
#endif
#if defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16
  if (input1.scalar_type() == ScalarType::Half) {
    static_assert(
        sizeof(executorch::aten::Half) == sizeof(float16_t),
        "ExecuTorch Half and CMSIS float16_t must have identical storage");
    const auto* input1_data = reinterpret_cast<const float16_t*>(
        input1.const_data_ptr<executorch::aten::Half>());
    const auto* input2_data = reinterpret_cast<const float16_t*>(
        input2.const_data_ptr<executorch::aten::Half>());
    auto* output_data = reinterpret_cast<float16_t*>(
        out.mutable_data_ptr<executorch::aten::Half>());
    return arm_minimum_f16(
        nullptr,
        input1_data,
        &input1_dims,
        input2_data,
        &input2_dims,
        output_data,
        &output_dims);
  }
#endif
  return ARM_CMSIS_NN_ARG_ERROR;
}

} // namespace native
} // namespace cortex_m
