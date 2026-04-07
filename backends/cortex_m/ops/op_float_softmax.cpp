/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "cortex_m_ops_common.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace cortex_m {
namespace native {

namespace {

inline bool is_last_dim(const Tensor& tensor, int64_t dim) {
  const auto rank = tensor.dim();
  const int64_t positive_dim = dim >= 0 ? dim : dim + rank;
  return positive_dim == static_cast<int64_t>(rank - 1);
}

inline int64_t normalize_dim(const Tensor& tensor, int64_t dim) {
  const auto rank = tensor.dim();
  const int64_t positive_dim = dim >= 0 ? dim : dim + rank;
  return positive_dim;
}

inline bool validate_softmax_shape(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Tensor& out,
    int64_t dim,
    int32_t& num_rows,
    int32_t& row_size) {
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

  if (!is_last_dim(input, dim)) {
    ET_LOG(
        Error,
        "%s: only last-dimension softmax is supported (dim=%lld, rank=%zu)",
        op_name,
        static_cast<long long>(dim),
        static_cast<size_t>(input.dim()));
    context.fail(Error::InvalidArgument);
    return false;
  }

  const auto positive_dim = normalize_dim(input, dim);
  const int64_t row_size64 = input.size(positive_dim);
  if (row_size64 <= 0 || row_size64 > std::numeric_limits<int32_t>::max()) {
    ET_LOG(
        Error,
        "%s: row size must fit in int32 (row_size=%lld)",
        op_name,
        static_cast<long long>(row_size64));
    context.fail(Error::InvalidArgument);
    return false;
  }
  row_size = static_cast<int32_t>(row_size64);

  const int64_t num_rows64 = input.numel() / row_size64;
  if (num_rows64 <= 0 || num_rows64 > std::numeric_limits<int32_t>::max()) {
    ET_LOG(
        Error,
        "%s: num_rows must fit in int32 (num_rows=%lld)",
        op_name,
        static_cast<long long>(num_rows64));
    context.fail(Error::InvalidArgument);
    return false;
  }
  num_rows = static_cast<int32_t>(num_rows64);
  return true;
}

} // namespace

Tensor& softmax_f32_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    int64_t dim,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32)
  (void)input;
  (void)dim;
  fail_cmsis_float_kernel_unavailable(context, "softmax_f32_out", "float32");
  return out;
#else
  if (input.scalar_type() != ScalarType::Float ||
      out.scalar_type() != ScalarType::Float) {
    ET_LOG(
        Error,
        "softmax_f32_out: only float32 tensors are supported (input=%d, out=%d)",
        static_cast<int>(input.scalar_type()),
        static_cast<int>(out.scalar_type()));
    context.fail(Error::InvalidArgument);
    return out;
  }

  int32_t num_rows = 0;
  int32_t row_size = 0;
  if (!validate_softmax_shape(
          context, "softmax_f32_out", input, out, dim, num_rows, row_size)) {
    return out;
  }

  const float32_t* input_data = input.const_data_ptr<float32_t>();
  float32_t* output_data = out.mutable_data_ptr<float32_t>();
  const arm_cmsis_nn_status status =
      arm_softmax_f32(input_data, num_rows, row_size, output_data);

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "softmax_f32_out: arm_softmax_f32 failed with status [%d]",
        static_cast<int>(status));
    context.fail(Error::Internal);
  }

  return out;
#endif
}

Tensor& softmax_f16_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    int64_t dim,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16)
  (void)input;
  (void)dim;
  fail_cmsis_float_kernel_unavailable(context, "softmax_f16_out", "float16");
  return out;
#else
  if (input.scalar_type() != ScalarType::Half ||
      out.scalar_type() != ScalarType::Half) {
    ET_LOG(
        Error,
        "softmax_f16_out: only float16 tensors are supported (input=%d, out=%d)",
        static_cast<int>(input.scalar_type()),
        static_cast<int>(out.scalar_type()));
    context.fail(Error::InvalidArgument);
    return out;
  }

  int32_t num_rows = 0;
  int32_t row_size = 0;
  if (!validate_softmax_shape(
          context, "softmax_f16_out", input, out, dim, num_rows, row_size)) {
    return out;
  }

  static_assert(
      sizeof(executorch::aten::Half) == sizeof(float16_t),
      "ExecuTorch Half and CMSIS float16_t must have identical storage");

  const auto* input_data = reinterpret_cast<const float16_t*>(
      input.const_data_ptr<executorch::aten::Half>());
  auto* output_data = reinterpret_cast<float16_t*>(
      out.mutable_data_ptr<executorch::aten::Half>());

  const arm_cmsis_nn_status status =
      arm_softmax_f16(input_data, num_rows, row_size, output_data);

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "softmax_f16_out: arm_softmax_f16 failed with status [%d]",
        static_cast<int>(status));
    context.fail(Error::Internal);
  }

  return out;
#endif
}

} // namespace native
} // namespace cortex_m
