/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "cortex_m_ops_common.h"

#include <limits>

namespace cortex_m {
namespace native {

namespace {

template <typename ScalarT>
void log_sample_values(
    const char* op_name,
    const char* label,
    const ScalarT* data,
    int32_t count) {
  if (count <= 0 || data == nullptr) {
    return;
  }
  ET_LOG(
      Info,
      "%s: %s[0..%d] = {%f, %f, %f, %f, %f, %f, %f, %f}",
      op_name,
      label,
      count - 1,
      count > 0 ? static_cast<double>(data[0]) : 0.0,
      count > 1 ? static_cast<double>(data[1]) : 0.0,
      count > 2 ? static_cast<double>(data[2]) : 0.0,
      count > 3 ? static_cast<double>(data[3]) : 0.0,
      count > 4 ? static_cast<double>(data[4]) : 0.0,
      count > 5 ? static_cast<double>(data[5]) : 0.0,
      count > 6 ? static_cast<double>(data[6]) : 0.0,
      count > 7 ? static_cast<double>(data[7]) : 0.0);
}

template <typename ScalarT, typename BmmParamsT>
inline BmmParamsT make_float_bmm_params(
    float activation_min,
    float activation_max) {
  BmmParamsT params{
      false,
      false,
      {
          static_cast<ScalarT>(activation_min),
          static_cast<ScalarT>(activation_max),
      },
  };
  // Default to the ordinary RHS layout first. The runtime path below will flip
  // this to NT_N_PACKED when either:
  //   1. lowering already provided a packed constant RHS, or
  //   2. we repack a dynamic/non-packed RHS into a temporary buffer.
  set_standard_cmsis_rhs_format(params);
  return params;
}

template <typename ScalarT>
bool prepare_float_batch_matmul_config(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& lhs,
    const Tensor& rhs_transposed,
    bool rhs_is_packed,
    int64_t packed_rhs_cols,
    Tensor& out,
    cmsis_nn_dims& lhs_dims,
    cmsis_nn_dims& rhs_dims,
    cmsis_nn_dims& out_dims) {
  const ScalarType expected_dtype =
      std::is_same_v<ScalarT, float32_t> ? ScalarType::Float : ScalarType::Half;
  const char* expected_dtype_name =
      expected_dtype == ScalarType::Float ? "float32" : "float16";

  if (lhs.scalar_type() != expected_dtype ||
      rhs_transposed.scalar_type() != expected_dtype ||
      out.scalar_type() != expected_dtype) {
    ET_LOG(
        Error,
        "%s: lhs/rhs_transposed/out must all be %s",
        op_name,
        expected_dtype_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (lhs.dim() != 3 || out.dim() != 3 ||
      (!rhs_is_packed && rhs_transposed.dim() != 3) ||
      (rhs_is_packed && rhs_transposed.dim() != 1)) {
    ET_LOG(
        Error,
        "%s: expected rank-3 lhs/out and %s rhs_transposed",
        op_name,
        rhs_is_packed ? "rank-1 packed" : "rank-3");
    context.fail(Error::InvalidArgument);
    return false;
  }

  int32_t batch = 0;
  int32_t lhs_rows = 0;
  int32_t inner = 0;
  int32_t rhs_batch = 0;
  int32_t rhs_cols = 0;
  int32_t rhs_inner = 0;
  int32_t out_batch = 0;
  int32_t out_rows = 0;
  int32_t out_cols = 0;

  if (!check_int32_within_range(
          context, op_name, lhs.size(0), "lhs batch", batch) ||
      !check_int32_within_range(
          context, op_name, lhs.size(1), "lhs rows", lhs_rows) ||
      !check_int32_within_range(
          context, op_name, lhs.size(2), "lhs inner", inner) ||
      !check_int32_within_range(
          context, op_name, out.size(0), "out batch", out_batch) ||
      !check_int32_within_range(
          context, op_name, out.size(1), "out rows", out_rows) ||
      !check_int32_within_range(
          context, op_name, out.size(2), "out cols", out_cols)) {
    return false;
  }

  if (rhs_is_packed) {
    // Packed RHS arrives as one flat tensor. The original logical shape is not
    // recoverable from rhs_transposed.sizes(), so lowering passes the logical
    // N/output-column dimension separately in packed_rhs_cols. The K/inner
    // dimension comes from lhs.size(2).
    //
    // Unpacked path:
    //   lhs            = [B, M, K]
    //   rhs_transposed = [B, N, K]
    //
    // Packed path:
    //   lhs            = [B, M, K]
    //   rhs_transposed = [flat packed bytes interpreted as ScalarT]
    //   packed_rhs_cols = N
    rhs_batch = batch;
    rhs_inner = inner;
    if (!check_int32_within_range(
            context, op_name, packed_rhs_cols, "packed rhs cols", rhs_cols)) {
      return false;
    }
  } else if (
      !check_int32_within_range(
          context, op_name, rhs_transposed.size(0), "rhs batch", rhs_batch) ||
      !check_int32_within_range(
          context, op_name, rhs_transposed.size(1), "rhs cols", rhs_cols) ||
      !check_int32_within_range(
          context, op_name, rhs_transposed.size(2), "rhs inner", rhs_inner)) {
    return false;
  }

  if (batch != rhs_batch) {
    ET_LOG(Error, "%s: batch dims must match", op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (inner != rhs_inner) {
    ET_LOG(Error, "%s: inner dims must match", op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (out_batch != batch || out_rows != lhs_rows || out_cols != rhs_cols) {
    ET_LOG(Error, "%s: output shape mismatch", op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  // Match the CMSIS-NN batch-matmul contract used by the unit tests.
  // cmsis_nn_dims field order is {n, h, w, c}, while the batch-matmul kernels
  // interpret:
  //   c = rows
  //   w = cols
  // For ET's 3-D bmm, the extra height dimension is always 1.
  lhs_dims = cmsis_nn_dims{
      .n = batch,
      .h = 1,
      .w = inner,
      .c = lhs_rows,
  };
  rhs_dims = cmsis_nn_dims{
      .n = batch,
      .h = 1,
      .w = inner,
      .c = rhs_cols,
  };
  out_dims = cmsis_nn_dims{
      .n = batch,
      .h = 1,
      .w = rhs_cols,
      .c = lhs_rows,
  };
  return true;
}

template <typename ScalarT, typename BmmParamsT>
Tensor& batch_matmul_out_impl(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& lhs,
    const Tensor& rhs_transposed,
    bool rhs_is_packed,
    int64_t packed_rhs_cols,
    double activation_min,
    double activation_max,
    Tensor& out,
    arm_cmsis_nn_status (*cmsis_fn)(
        const cmsis_nn_context*,
        const BmmParamsT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        const ScalarT*,
        const cmsis_nn_dims*,
        ScalarT*),
    int32_t (*buffer_size_fn)(
        const BmmParamsT*,
        const cmsis_nn_dims*,
        const cmsis_nn_dims*,
        const cmsis_nn_dims*)) {
  cmsis_nn_dims lhs_dims;
  cmsis_nn_dims rhs_dims;
  cmsis_nn_dims out_dims;
  if (!prepare_float_batch_matmul_config<ScalarT>(
          context,
          op_name,
          lhs,
          rhs_transposed,
          rhs_is_packed,
          packed_rhs_cols,
          out,
          lhs_dims,
          rhs_dims,
          out_dims)) {
    return out;
  }

  BmmParamsT bmm_params = make_float_bmm_params<ScalarT, BmmParamsT>(
      static_cast<float>(activation_min), static_cast<float>(activation_max));

  ET_LOG(
      Info,
      "%s: lhs_dims={n=%d,h=%d,c=%d,w=%d} rhs_dims={n=%d,h=%d,c=%d,w=%d} out_dims={n=%d,h=%d,c=%d,w=%d}",
      op_name,
      lhs_dims.n,
      lhs_dims.h,
      lhs_dims.c,
      lhs_dims.w,
      rhs_dims.n,
      rhs_dims.h,
      rhs_dims.c,
      rhs_dims.w,
      out_dims.n,
      out_dims.h,
      out_dims.c,
      out_dims.w);
  log_sample_values(op_name, "lhs", lhs.const_data_ptr<ScalarT>(), 6);
  log_sample_values(
      op_name, "rhs_transposed", rhs_transposed.const_data_ptr<ScalarT>(), 12);

  const int32_t buf_size =
      buffer_size_fn(&bmm_params, &lhs_dims, &rhs_dims, &out_dims);

  cmsis_nn_context ctx{nullptr, 0};
  if (buf_size > 0) {
    auto buffer_or_error = context.allocate_temp(buf_size);
    if (!buffer_or_error.ok()) {
      ET_LOG(
          Error,
          "%s: failed to allocate scratch buffer (%d bytes)",
          op_name,
          buf_size);
      context.fail(buffer_or_error.error());
      return out;
    }
    ctx.buf = buffer_or_error.get();
    ctx.size = buf_size;
  }

  const ScalarT* rhs_ptr = rhs_transposed.const_data_ptr<ScalarT>();
  if (rhs_is_packed) {
    // Lowering already packed the constant RHS offline, so runtime can point
    // CMSIS directly at the flat packed buffer.
    set_packed_cmsis_rhs_format(bmm_params);
  } else if constexpr (has_cmsis_rhs_format<BmmParamsT>::value) {
    // Fallback path for non-packed RHS.
    //
    // Before:
    //   rhs_transposed = [B, N, K]
    //
    // After temporary packing:
    //   packed_rhs = [batch][block][k][lane]
    //
    // where each batch slice packs the logical [N, K] matrix into the CMSIS
    // NT_N layout expected by arm_batch_matmul_* when rhs_format is set to
    // ARM_NN_WEIGHT_FORMAT_NT_N_PACKED.
    const int32_t rhs_batch = rhs_dims.n;
    const int32_t rhs_rows = rhs_dims.c;
    const int32_t rhs_cols = rhs_dims.w;
    constexpr int32_t kBlockCols = get_cmsis_packed_n_block_cols<ScalarT>();
    const bool packed_batch_stride_matches_cmsis =
        rhs_batch == 1 || (rhs_rows % kBlockCols) == 0;
    if (!packed_batch_stride_matches_cmsis) {
      // CMSIS-NN advances between BMM batches using the logical unpacked RHS
      // stride, rhs_rows * rhs_cols. NT_N packing pads rhs_rows up to the block
      // size, so multi-batch packed RHS is only safe when no row padding is
      // needed. Otherwise keep the standard NT_T path for correctness.
    } else {
      const size_t packed_bytes = static_cast<size_t>(rhs_batch) *
          get_nt_n_packed_weight_bytes<ScalarT>(rhs_rows, rhs_cols);
      auto packed_or_error =
          context.allocate_temp(packed_bytes, kCortexMMveAlignment);
      if (!packed_or_error.ok()) {
        ET_LOG(
            Error,
            "%s: failed to allocate %zu packed RHS bytes",
            op_name,
            packed_bytes);
        context.fail(packed_or_error.error());
        return out;
      }
      auto* packed_base = static_cast<ScalarT*>(packed_or_error.get());
      const size_t packed_batch_stride =
          get_nt_n_packed_weight_bytes<ScalarT>(rhs_rows, rhs_cols) /
          sizeof(ScalarT);
      const size_t rhs_batch_stride =
          static_cast<size_t>(rhs_rows) * static_cast<size_t>(rhs_cols);
      for (int32_t b = 0; b < rhs_batch; ++b) {
        pack_nt_t_weights_to_nt_n_packed(
            rhs_ptr + static_cast<size_t>(b) * rhs_batch_stride,
            rhs_rows,
            rhs_cols,
            packed_base + static_cast<size_t>(b) * packed_batch_stride);
      }
      rhs_ptr = packed_base;
      set_packed_cmsis_rhs_format(bmm_params);
    }
  }

  const arm_cmsis_nn_status status = cmsis_fn(
      &ctx,
      &bmm_params,
      &lhs_dims,
      lhs.const_data_ptr<ScalarT>(),
      &rhs_dims,
      rhs_ptr,
      &out_dims,
      out.mutable_data_ptr<ScalarT>());

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "%s: CMSIS-NN batch matmul failed with status [%d]",
        op_name,
        status);
    context.fail(Error::Internal);
  }

  log_sample_values(op_name, "out", out.const_data_ptr<ScalarT>(), 8);

  return out;
}

} // namespace

Tensor& batch_matmul_f32_out(
    KernelRuntimeContext& context,
    const Tensor& lhs,
    const Tensor& rhs_transposed,
    bool rhs_is_packed,
    int64_t rhs_cols,
    double activation_min,
    double activation_max,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32)
  (void)lhs;
  (void)rhs_transposed;
  (void)rhs_is_packed;
  (void)rhs_cols;
  (void)activation_min;
  (void)activation_max;
  fail_cmsis_float_kernel_unavailable(
      context, "batch_matmul_f32_out", "float32");
  return out;
#else
  return batch_matmul_out_impl<float32_t, cmsis_nn_bmm_params_f32>(
      context,
      "batch_matmul_f32_out",
      lhs,
      rhs_transposed,
      rhs_is_packed,
      rhs_cols,
      activation_min,
      activation_max,
      out,
      arm_batch_matmul_f32,
      arm_batch_matmul_f32_get_buffer_size);
#endif
}

Tensor& batch_matmul_f16_out(
    KernelRuntimeContext& context,
    const Tensor& lhs,
    const Tensor& rhs_transposed,
    bool rhs_is_packed,
    int64_t rhs_cols,
    double activation_min,
    double activation_max,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16)
  (void)lhs;
  (void)rhs_transposed;
  (void)rhs_is_packed;
  (void)rhs_cols;
  (void)activation_min;
  (void)activation_max;
  fail_cmsis_float_kernel_unavailable(
      context, "batch_matmul_f16_out", "float16");
  return out;
#else
  return batch_matmul_out_impl<float16_t, cmsis_nn_bmm_params_f16>(
      context,
      "batch_matmul_f16_out",
      lhs,
      rhs_transposed,
      rhs_is_packed,
      rhs_cols,
      activation_min,
      activation_max,
      out,
      arm_batch_matmul_f16,
      arm_batch_matmul_f16_get_buffer_size);
#endif
}

} // namespace native
} // namespace cortex_m
