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

namespace {} // namespace

Tensor& max_pool2d_f32_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Int64ArrayRef kernel_size,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32)
  (void)input;
  (void)kernel_size;
  (void)stride;
  (void)padding;
  fail_cmsis_float_kernel_unavailable(context, "max_pool2d_f32_out", "float32");
  return out;
#else
  cmsis_nn_dims input_dims;
  cmsis_nn_dims filter_dims;
  cmsis_nn_dims output_dims;
  int32_t pad_h = 0, pad_w = 0, stride_h = 0, stride_w = 0;
  if (!prepare_float_pool2d_config<float32_t>(
          context,
          "max_pool2d_f32_out",
          input,
          out,
          kernel_size,
          stride,
          padding,
          input_dims,
          filter_dims,
          output_dims,
          pad_h,
          pad_w,
          stride_h,
          stride_w,
          /*require_positive_kernel_stride=*/true)) {
    return out;
  }

  cmsis_nn_pool_params_f32 pool_params;
  fill_float_pool_params<float32_t>(
      pool_params, pad_h, pad_w, stride_h, stride_w);
  cmsis_nn_context cmsis_ctx{nullptr, 0};
  const arm_cmsis_nn_status status = arm_max_pool_f32(
      &cmsis_ctx,
      &pool_params,
      &input_dims,
      input.const_data_ptr<float32_t>(),
      &filter_dims,
      &output_dims,
      out.mutable_data_ptr<float32_t>());
  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "max_pool2d_f32_out: arm_max_pool_f32 failed with status [%d]",
        static_cast<int>(status));
    context.fail(Error::Internal);
  }
  return out;
#endif
}

Tensor& max_pool2d_f16_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Int64ArrayRef kernel_size,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16)
  (void)input;
  (void)kernel_size;
  (void)stride;
  (void)padding;
  fail_cmsis_float_kernel_unavailable(context, "max_pool2d_f16_out", "float16");
  return out;
#else
  static_assert(
      sizeof(executorch::aten::Half) == sizeof(float16_t),
      "ExecuTorch Half and CMSIS float16_t must have identical storage");

  cmsis_nn_dims input_dims;
  cmsis_nn_dims filter_dims;
  cmsis_nn_dims output_dims;
  int32_t pad_h = 0, pad_w = 0, stride_h = 0, stride_w = 0;
  if (!prepare_float_pool2d_config<float16_t>(
          context,
          "max_pool2d_f16_out",
          input,
          out,
          kernel_size,
          stride,
          padding,
          input_dims,
          filter_dims,
          output_dims,
          pad_h,
          pad_w,
          stride_h,
          stride_w,
          /*require_positive_kernel_stride=*/true)) {
    return out;
  }

  cmsis_nn_pool_params_f16 pool_params;
  fill_float_pool_params<float16_t>(
      pool_params, pad_h, pad_w, stride_h, stride_w);
  cmsis_nn_context cmsis_ctx{nullptr, 0};
  const arm_cmsis_nn_status status = arm_max_pool_f16(
      &cmsis_ctx,
      &pool_params,
      &input_dims,
      reinterpret_cast<const float16_t*>(
          input.const_data_ptr<executorch::aten::Half>()),
      &filter_dims,
      &output_dims,
      reinterpret_cast<float16_t*>(
          out.mutable_data_ptr<executorch::aten::Half>()));
  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "max_pool2d_f16_out: arm_max_pool_f16 failed with status [%d]",
        static_cast<int>(status));
    context.fail(Error::Internal);
  }
  return out;
#endif
}

} // namespace native
} // namespace cortex_m
