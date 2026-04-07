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

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

namespace {

constexpr size_t kMaxSupportedDims = 4;

template <typename CppT>
void cmsis_pad_float_unavailable(
    KernelRuntimeContext& context,
    const char* op_name) {
  fail_cmsis_float_kernel_unavailable(
      context, op_name, std::is_same_v<CppT, float> ? "float32" : "float16");
}

template <typename CppT>
bool make_pad_dims(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Int64ArrayRef& pre_pad,
    const Int64ArrayRef& post_pad,
    cmsis_nn_dims& input_dims,
    cmsis_nn_dims& cmsis_pre_pad,
    cmsis_nn_dims& cmsis_post_pad) {
  if (pre_pad.size() != kMaxSupportedDims ||
      post_pad.size() != kMaxSupportedDims) {
    ET_LOG(
        Error,
        "%s: expected pre_pad/post_pad length %zu, got %zu/%zu",
        op_name,
        kMaxSupportedDims,
        pre_pad.size(),
        post_pad.size());
    context.fail(Error::InvalidArgument);
    return false;
  }

  const size_t rank = input.dim();
  if (rank == 0 || rank > kMaxSupportedDims) {
    ET_LOG(
        Error,
        "%s: expected tensor rank in [1, %zu], got %zu",
        op_name,
        kMaxSupportedDims,
        rank);
    context.fail(Error::InvalidArgument);
    return false;
  }

  constexpr size_t kNhwcDimOrder[] = {0, 2, 3, 1};
  const size_t offset = kMaxSupportedDims - rank;
  const bool nhwc = is_channels_last_tensor(input);

  int32_t dims[kMaxSupportedDims] = {1, 1, 1, 1};
  for (size_t i = 0; i < rank; ++i) {
    const size_t src = nhwc ? kNhwcDimOrder[offset + i] : i;
    if (!check_int32_within_range(
            context,
            op_name,
            input.size(src),
            "input size",
            dims[offset + i])) {
      return false;
    }
  }

  int32_t pre_pad_vals[kMaxSupportedDims];
  int32_t post_pad_vals[kMaxSupportedDims];
  for (size_t i = 0; i < kMaxSupportedDims; ++i) {
    if (!check_int32_within_range(
            context, op_name, pre_pad[i], "pre_pad", pre_pad_vals[i]) ||
        !check_int32_within_range(
            context, op_name, post_pad[i], "post_pad", post_pad_vals[i])) {
      return false;
    }
  }

  input_dims = {dims[0], dims[1], dims[2], dims[3]};
  cmsis_pre_pad = {
      pre_pad_vals[0], pre_pad_vals[1], pre_pad_vals[2], pre_pad_vals[3]};
  cmsis_post_pad = {
      post_pad_vals[0], post_pad_vals[1], post_pad_vals[2], post_pad_vals[3]};

  return true;
}

#if defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32
inline arm_cmsis_nn_status call_cmsis_pad_f32(
    const float* input_data,
    float* output_data,
    float pad_value,
    const cmsis_nn_dims* input_dims,
    const cmsis_nn_dims* cmsis_pre_pad,
    const cmsis_nn_dims* cmsis_post_pad) {
  return arm_pad_f32(
      input_data,
      output_data,
      static_cast<float32_t>(pad_value),
      input_dims,
      cmsis_pre_pad,
      cmsis_post_pad);
}
#endif

#if defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16
inline arm_cmsis_nn_status call_cmsis_pad_f16(
    const executorch::aten::Half* input_data,
    executorch::aten::Half* output_data,
    double pad_value,
    const cmsis_nn_dims* input_dims,
    const cmsis_nn_dims* cmsis_pre_pad,
    const cmsis_nn_dims* cmsis_post_pad) {
  return arm_pad_f16(
      reinterpret_cast<const float16_t*>(input_data),
      reinterpret_cast<float16_t*>(output_data),
      static_cast<float16_t>(pad_value),
      input_dims,
      cmsis_pre_pad,
      cmsis_post_pad);
}
#endif

template <typename CppT>
Tensor& pad_float_out_impl(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Int64ArrayRef pre_pad,
    const Int64ArrayRef post_pad,
    double pad_value,
    Tensor& out) {
  if constexpr (std::is_same_v<CppT, float>) {
    if constexpr (!kCmsisFloat32Enabled) {
      cmsis_pad_float_unavailable<CppT>(context, op_name);
      return out;
    }
  } else {
    if constexpr (!kCmsisFloat16Enabled) {
      cmsis_pad_float_unavailable<CppT>(context, op_name);
      return out;
    }
  }

  constexpr ScalarType kExpectedDtype =
      std::is_same_v<CppT, float> ? ScalarType::Float : ScalarType::Half;

  if (input.scalar_type() != kExpectedDtype ||
      out.scalar_type() != kExpectedDtype) {
    ET_LOG(
        Error,
        "%s: expected tensors to have dtype %d (input=%d, out=%d)",
        op_name,
        static_cast<int>(kExpectedDtype),
        static_cast<int>(input.scalar_type()),
        static_cast<int>(out.scalar_type()));
    context.fail(Error::InvalidArgument);
    return out;
  }

  cmsis_nn_dims input_dims;
  cmsis_nn_dims cmsis_pre_pad;
  cmsis_nn_dims cmsis_post_pad;
  if (!make_pad_dims<CppT>(
          context,
          op_name,
          input,
          pre_pad,
          post_pad,
          input_dims,
          cmsis_pre_pad,
          cmsis_post_pad)) {
    return out;
  }

  const CppT* input_data = input.const_data_ptr<CppT>();
  CppT* output_data = out.mutable_data_ptr<CppT>();

  arm_cmsis_nn_status status = ARM_CMSIS_NN_ARG_ERROR;
  if constexpr (std::is_same_v<CppT, float>) {
#if defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32
    status = call_cmsis_pad_f32(
        input_data,
        output_data,
        static_cast<float>(pad_value),
        &input_dims,
        &cmsis_pre_pad,
        &cmsis_post_pad);
#else
    ET_CHECK_MSG(
        false,
        "pad_float_out_impl<float> must not be instantiated when float32 "
        "support is disabled");
#endif
  } else {
#if defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16
    status = call_cmsis_pad_f16(
        input_data,
        output_data,
        pad_value,
        &input_dims,
        &cmsis_pre_pad,
        &cmsis_post_pad);
#else
    ET_CHECK_MSG(
        false,
        "pad_float_out_impl<Half> must not be instantiated when float16 "
        "support is disabled");
#endif
  }

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "%s: CMSIS-NN pad failed with status [%d]",
        op_name,
        static_cast<int>(status));
    context.fail(Error::Internal);
  }

  return out;
}

} // namespace

Tensor& pad_f32_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Int64ArrayRef pre_pad,
    const Int64ArrayRef post_pad,
    double pad_value,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32)
  (void)input;
  (void)pre_pad;
  (void)post_pad;
  (void)pad_value;
  fail_cmsis_float_kernel_unavailable(context, "pad_f32_out", "float32");
  return out;
#else
  return pad_float_out_impl<float>(
      context, "pad_f32_out", input, pre_pad, post_pad, pad_value, out);
#endif
}

Tensor& pad_f16_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Int64ArrayRef pre_pad,
    const Int64ArrayRef post_pad,
    double pad_value,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16)
  (void)input;
  (void)pre_pad;
  (void)post_pad;
  (void)pad_value;
  fail_cmsis_float_kernel_unavailable(context, "pad_f16_out", "float16");
  return out;
#else
  return pad_float_out_impl<executorch::aten::Half>(
      context, "pad_f16_out", input, pre_pad, post_pad, pad_value, out);
#endif
}

} // namespace native
} // namespace cortex_m
