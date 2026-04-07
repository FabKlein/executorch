/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

#include <array>
#include <limits>

namespace cortex_m {
namespace native {

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

namespace {

constexpr size_t kMaxSupportedDims = 4;

cmsis_nn_dims make_cmsis_dims(const Tensor& tensor) {
  const auto rank = tensor.dim();
  const auto sizes = tensor.sizes();
  return cmsis_nn_dims{
      static_cast<int32_t>(rank >= 1 ? sizes[0] : 1),
      static_cast<int32_t>(rank >= 2 ? sizes[1] : 1),
      static_cast<int32_t>(rank >= 3 ? sizes[2] : 1),
      static_cast<int32_t>(rank >= 4 ? sizes[3] : 1)};
}

} // namespace

// cppcheck-suppress unusedFunction
Tensor& transpose_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Int64ArrayRef perm,
    Tensor& out) {
  const ScalarType dtype = input.scalar_type();
  if (out.scalar_type() != dtype) {
    ET_LOG(
        Error,
        "transpose_out: input/output dtype mismatch (input=%d, out=%d)",
        static_cast<int>(dtype),
        static_cast<int>(out.scalar_type()));
    context.fail(Error::InvalidArgument);
    return out;
  }

  if (dtype != ScalarType::Char
#if defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32
      && dtype != ScalarType::Float
#endif
#if defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16
      && dtype != ScalarType::Half
#endif
  ) {
    ET_LOG(
        Error,
        "transpose_out: dtype %d is not supported by this build",
        static_cast<int>(dtype));
    context.fail(Error::InvalidArgument);
    return out;
  }

  const size_t rank = input.dim();
  if (rank == 0 || rank > kMaxSupportedDims) {
    ET_LOG(
        Error,
        "transpose_out: expected tensor rank in [1, %zu], got %zu",
        kMaxSupportedDims,
        rank);
    context.fail(Error::InvalidArgument);
    return out;
  }

  if (perm.size() != static_cast<int64_t>(rank)) {
    ET_LOG(
        Error,
        "transpose_out: permutation length %zd does not match tensor rank %zu",
        perm.size(),
        rank);
    context.fail(Error::InvalidArgument);
    return out;
  }

  std::array<bool, kMaxSupportedDims> seen_dims{false, false, false, false};
  std::array<int32_t, kMaxSupportedDims> perm_i32{0, 1, 2, 3};
  for (size_t i = 0; i < rank; ++i) {
    const auto in_size = input.size(i);
    const auto out_size = out.size(i);
    if (in_size > std::numeric_limits<int32_t>::max() ||
        out_size > std::numeric_limits<int32_t>::max()) {
      ET_LOG(
          Error,
          "transpose_out: dimension size exceeds int32_t range (input=%lld, output=%lld)",
          static_cast<long long>(in_size),
          static_cast<long long>(out_size));
      context.fail(Error::InvalidArgument);
      return out;
    }

    int32_t perm_value = 0;
    if (!check_int32_within_range(
            context, "transpose_out", perm[i], "perm", perm_value)) {
      return out;
    }
    if (perm_value < 0 || perm_value >= static_cast<int32_t>(rank)) {
      ET_LOG(
          Error,
          "transpose_out: permutation index %d at position %zu is out of range for rank %zu",
          perm_value,
          i,
          rank);
      context.fail(Error::InvalidArgument);
      return out;
    }
    if (seen_dims[perm_value]) {
      ET_LOG(
          Error, "transpose_out: duplicate permutation index %d", perm_value);
      context.fail(Error::InvalidArgument);
      return out;
    }
    seen_dims[perm_value] = true;
    perm_i32[i] = perm_value;

    if (out_size != input.size(static_cast<size_t>(perm_value))) {
      ET_LOG(
          Error,
          "transpose_out: output shape does not match permuted input shape at dim %zu (expected %lld, got %lld)",
          i,
          static_cast<long long>(input.size(static_cast<size_t>(perm_value))),
          static_cast<long long>(out_size));
      context.fail(Error::InvalidArgument);
      return out;
    }
  }

  const cmsis_nn_dims input_dims = make_cmsis_dims(input);
  const cmsis_nn_dims output_dims = make_cmsis_dims(out);

  arm_cmsis_nn_status status = ARM_CMSIS_NN_SUCCESS;
  if (dtype == ScalarType::Char) {
    std::array<uint32_t, kMaxSupportedDims> perm_buffer{0, 1, 2, 3};
    for (size_t i = 0; i < rank; ++i) {
      perm_buffer[i] = static_cast<uint32_t>(perm_i32[i]);
    }
    const cmsis_nn_transpose_params transpose_params{
        static_cast<int32_t>(rank), perm_buffer.data()};

    const int8_t* input_data = input.const_data_ptr<int8_t>();
    int8_t* output_data = out.mutable_data_ptr<int8_t>();
    status = arm_transpose_s8(
        input_data, output_data, &input_dims, &output_dims, &transpose_params);
  }
#if defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32
  else if (dtype == ScalarType::Float) {
    cmsis_nn_transpose_params_f32 transpose_params = {
        static_cast<int32_t>(rank), {0, 1, 2, 3}, ARM_NN_LAYOUT_NHWC};
    for (size_t i = 0; i < rank; ++i) {
      transpose_params.perm[i] = perm_i32[i];
    }

    const float32_t* input_data = input.const_data_ptr<float32_t>();
    float32_t* output_data = out.mutable_data_ptr<float32_t>();
    status = arm_transpose_f32(
        nullptr,
        &transpose_params,
        &input_dims,
        input_data,
        &output_dims,
        output_data);
  }
#endif
#if defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16
  else if (dtype == ScalarType::Half) {
    static_assert(
        sizeof(executorch::aten::Half) == sizeof(float16_t),
        "ExecuTorch Half and CMSIS float16_t must have identical storage");
    cmsis_nn_transpose_params_f16 transpose_params = {
        static_cast<int32_t>(rank), {0, 1, 2, 3}, ARM_NN_LAYOUT_NHWC};
    for (size_t i = 0; i < rank; ++i) {
      transpose_params.perm[i] = perm_i32[i];
    }

    const auto* input_data = reinterpret_cast<const float16_t*>(
        input.const_data_ptr<executorch::aten::Half>());
    auto* output_data = reinterpret_cast<float16_t*>(
        out.mutable_data_ptr<executorch::aten::Half>());
    status = arm_transpose_f16(
        nullptr,
        &transpose_params,
        &input_dims,
        input_data,
        &output_dims,
        output_data);
  }
#endif

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "transpose_out: CMSIS-NN transpose failed with status [%d]",
        static_cast<int>(status));
    context.fail(Error::Internal);
    return out;
  }

  return out;
}

} // namespace native
} // namespace cortex_m
