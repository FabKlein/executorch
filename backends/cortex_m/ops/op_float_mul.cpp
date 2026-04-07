/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "float_binary_op_utils.h"

namespace cortex_m {
namespace native {

Tensor& mul_f32_out(
    KernelRuntimeContext& context,
    const Tensor& input1,
    const Tensor& input2,
    const double activation_min,
    const double activation_max,
    Tensor& out) {
#if defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32
  return run_float_binary_out<float, float, arm_elementwise_mul_f32>(
      context,
      input1,
      input2,
      activation_min,
      activation_max,
      out,
      "mul_f32_out",
      "arm_elementwise_mul_f32");
#else
  fail_cmsis_float_kernel_unavailable(context, "mul_f32_out", "float32");
  return out;
#endif
}

Tensor& mul_f16_out(
    KernelRuntimeContext& context,
    const Tensor& input1,
    const Tensor& input2,
    const double activation_min,
    const double activation_max,
    Tensor& out) {
#if defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16
  return run_float_binary_out<
      executorch::aten::Half,
      float16_t,
      arm_elementwise_mul_f16>(
      context,
      input1,
      input2,
      activation_min,
      activation_max,
      out,
      "mul_f16_out",
      "arm_elementwise_mul_f16");
#else
  fail_cmsis_float_kernel_unavailable(context, "mul_f16_out", "float16");
  return out;
#endif
}

} // namespace native
} // namespace cortex_m
