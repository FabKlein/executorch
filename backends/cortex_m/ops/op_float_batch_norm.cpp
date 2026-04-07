/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cmath>
#include <cstring>
#include "cortex_m_ops_common.h"

namespace cortex_m {
namespace native {

namespace {

template <typename ScalarT>
void apply_batch_norm_default_order(
    const Tensor& input,
    Tensor& out,
    const ScalarT* scale,
    const ScalarT* bias) {
  // Fallback implementation for default-order NCHW tensors.
  //
  // CMSIS float batch-norm APIs operate on NHWC / channels-last tensors. For
  // some graphs (for example after layout-sensitive GhostNet-style plumbing) we
  // now allow batch norm to stay in default order instead of forcing an
  // immediate layout clone. In that case we apply the same per-channel affine
  // transform directly with a simple scalar loop:
  //
  //   out[n, c, h, w] = input[n, c, h, w] * scale[c] + bias[c]
  //
  // Layout summary:
  //
  //   channels_last path: input[N,H,W,C] -> CMSIS arm_batch_norm_*
  //   default-order path: input[N,C,H,W] -> scalar fallback loop
  const int64_t batch = input.size(0);
  const int64_t channels = input.size(1);
  const int64_t height = input.size(2);
  const int64_t width = input.size(3);
  const auto* input_ptr = cmsis_const_data_ptr<ScalarT>(input);
  auto* out_ptr = cmsis_mutable_data_ptr<ScalarT>(out);

  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t c = 0; c < channels; ++c) {
      const ScalarT scale_value = scale[c];
      const ScalarT bias_value = bias[c];
      for (int64_t h = 0; h < height; ++h) {
        for (int64_t w = 0; w < width; ++w) {
          const int64_t offset = ((n * channels + c) * height + h) * width + w;
          out_ptr[offset] = static_cast<ScalarT>(
              static_cast<float>(input_ptr[offset]) *
                  static_cast<float>(scale_value) +
              static_cast<float>(bias_value));
        }
      }
    }
  }
}

template <typename ScalarT>
void apply_batch_norm_2d(
    const Tensor& input,
    Tensor& out,
    const ScalarT* scale,
    const ScalarT* bias) {
  // Dense-layer batch norm is exported as rank-2 [N, C]. CMSIS-NN's float
  // batch-norm entry point is NHWC-oriented and assumes an image-like tensor,
  // so there is no useful CMSIS call to make here. Apply the same inference
  // affine transform directly:
  //
  //   out[n, c] = input[n, c] * folded_scale[c] + folded_bias[c]
  //
  // This is mostly a robustness path for graphs where BN was not folded away
  // before a dense layer. The preferred optimized graph is still:
  //
  //   linear -> batch_norm    rewritten/folded ahead of runtime
  //
  // but if the BN survives, keep execution correct instead of rejecting the
  // model.
  const int64_t batch = input.size(0);
  const int64_t channels = input.size(1);
  const auto* input_ptr = cmsis_const_data_ptr<ScalarT>(input);
  auto* out_ptr = cmsis_mutable_data_ptr<ScalarT>(out);

  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t c = 0; c < channels; ++c) {
      const int64_t offset = n * channels + c;
      out_ptr[offset] = static_cast<ScalarT>(
          static_cast<float>(input_ptr[offset]) * static_cast<float>(scale[c]) +
          static_cast<float>(bias[c]));
    }
  }
}

template <typename ScalarT>
Tensor& batch_norm_out_impl(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& bias,
    Tensor& out,
    arm_cmsis_nn_status (*cmsis_fn)(
        const ScalarT*,
        ScalarT*,
        const ScalarT*,
        const ScalarT*,
        const cmsis_nn_dims*,
        arm_nn_tensor_layout)) {
  if constexpr (!is_cmsis_float_enabled<ScalarT>()) {
    fail_cmsis_float_kernel_unavailable(
        context, op_name, cmsis_float_dtype_name<ScalarT>());
    return out;
  }

  const ScalarType dtype = cmsis_float_scalar_type<ScalarT>();
  if (input.scalar_type() != dtype || scale.scalar_type() != dtype ||
      bias.scalar_type() != dtype || out.scalar_type() != dtype) {
    ET_LOG(
        Error,
        "%s: input/scale/bias/output must all be %s",
        op_name,
        cmsis_float_dtype_name<ScalarT>());
    context.fail(Error::InvalidArgument);
    return out;
  }

  if (input.dim() != 4 || out.dim() != 4) {
    ET_LOG(Error, "%s: input/output must be 4-D", op_name);
    context.fail(Error::InvalidArgument);
    return out;
  }

  const bool input_is_channels_last = is_channels_last_tensor(input);
  const bool out_is_channels_last = is_channels_last_tensor(out);
  const bool input_is_default = is_default_dim_order_tensor(input);
  const bool out_is_default = is_default_dim_order_tensor(out);

  // Batch norm is layout-tolerant now, but only if input and output use the
  // same layout family. We do not want to silently mix:
  //
  //   input [N,C,H,W]  -> output [N,H,W,C]
  //
  // inside this op; explicit dim-order conversion should happen elsewhere.
  if (!is_default_or_channels_last_tensor(input) ||
      !is_default_or_channels_last_tensor(out) ||
      input_is_channels_last != out_is_channels_last ||
      input_is_default != out_is_default) {
    ET_LOG(
        Error,
        "%s: input/output must use the same default or channels_last layout",
        op_name);
    context.fail(Error::InvalidArgument);
    return out;
  }

  if (input.sizes() != out.sizes()) {
    ET_LOG(Error, "%s: input/output shapes must match", op_name);
    context.fail(Error::InvalidArgument);
    return out;
  }

  if (scale.dim() != 1 || bias.dim() != 1 || scale.size(0) != input.size(1) ||
      bias.size(0) != input.size(1)) {
    ET_LOG(
        Error,
        "%s: scale and bias must be rank-1 with length equal to channels (%lld)",
        op_name,
        static_cast<long long>(input.size(1)));
    context.fail(Error::InvalidArgument);
    return out;
  }

  int32_t batch = 0;
  int32_t height = 0;
  int32_t width = 0;
  int32_t channels = 0;
  if (!check_int32_within_range(
          context, op_name, input.size(0), "batch", batch) ||
      !check_int32_within_range(
          context, op_name, input.size(2), "height", height) ||
      !check_int32_within_range(
          context, op_name, input.size(3), "width", width) ||
      !check_int32_within_range(
          context, op_name, input.size(1), "channels", channels)) {
    return out;
  }

  if (input_is_channels_last) {
    // Fast path: input/output already match the CMSIS NHWC contract.
    const cmsis_nn_dims input_dims{batch, height, width, channels};
    const arm_cmsis_nn_status status = cmsis_fn(
        cmsis_const_data_ptr<ScalarT>(input),
        cmsis_mutable_data_ptr<ScalarT>(out),
        cmsis_const_data_ptr<ScalarT>(scale),
        cmsis_const_data_ptr<ScalarT>(bias),
        &input_dims,
        ARM_NN_LAYOUT_NHWC);

    if (status != ARM_CMSIS_NN_SUCCESS) {
      ET_LOG(
          Error,
          "%s: arm_batch_norm failed with status %d",
          op_name,
          static_cast<int>(status));
      context.fail(Error::Internal);
    }
  } else {
    // Default-order path: preserve NCHW layout and avoid a forced clone to
    // channels-last. This is slower than the CMSIS fast path but better than
    // inserting extra layout materialization in some graphs.
    apply_batch_norm_default_order<ScalarT>(
        input,
        out,
        cmsis_const_data_ptr<ScalarT>(scale),
        cmsis_const_data_ptr<ScalarT>(bias));
  }

  return out;
}

} // namespace

Tensor& batch_norm_f32_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& bias,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32)
  (void)input;
  (void)scale;
  (void)bias;
  fail_cmsis_float_kernel_unavailable(context, "batch_norm_f32_out", "float32");
  return out;
#else
  return batch_norm_out_impl<float32_t>(
      context,
      "batch_norm_f32_out",
      input,
      scale,
      bias,
      out,
      arm_batch_norm_f32);
#endif
}

Tensor& batch_norm_f16_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& bias,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16)
  (void)input;
  (void)scale;
  (void)bias;
  fail_cmsis_float_kernel_unavailable(context, "batch_norm_f16_out", "float16");
  return out;
#else
  return batch_norm_out_impl<float16_t>(
      context,
      "batch_norm_f16_out",
      input,
      scale,
      bias,
      out,
      arm_batch_norm_f16);
#endif
}

Tensor& batch_norm_native_f32_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    double eps,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F32) && ARM_NN_ENABLE_F32)
  (void)input;
  (void)weight;
  (void)bias;
  (void)running_mean;
  (void)running_var;
  (void)eps;
  fail_cmsis_float_kernel_unavailable(
      context, "batch_norm_native_f32_out", "float32");
  return out;
#else

  const ScalarType dtype = cmsis_float_scalar_type<float32_t>();
  if (input.scalar_type() != dtype || weight.scalar_type() != dtype ||
      bias.scalar_type() != dtype || running_mean.scalar_type() != dtype ||
      running_var.scalar_type() != dtype || out.scalar_type() != dtype) {
    ET_LOG(
        Error,
        "%s: input/state tensors must all be float32",
        "batch_norm_native_f32_out");
    context.fail(Error::InvalidArgument);
    return out;
  }
  const bool input_is_4d = input.dim() == 4;
  const bool input_is_channels_last =
      input_is_4d && is_channels_last_tensor(input);
  const bool out_is_channels_last = input_is_4d && is_channels_last_tensor(out);
  const bool input_is_default = is_default_dim_order_tensor(input);
  const bool out_is_default = is_default_dim_order_tensor(out);

  // Native batch norm follows the same layout policy as batch_norm_out_impl:
  // either both tensors are NHWC/channels-last and we call CMSIS, or both are
  // default-order and we use the local fallback after folding mean/var. Rank-2
  // dense BN uses the same fallback because there is no NHWC spatial layout.
  //
  // Accepted forms:
  //
  //   [N,C]       dense BN fallback
  //   [N,H,W,C]   CMSIS-NN fast path
  //   [N,C,H,W]   local fallback, preserving default-order layout
  //
  // Mixed input/output layouts are rejected here; layout conversion must be an
  // explicit graph op so the pass pipeline remains visible and debuggable.
  if (input.dim() != out.dim() || (input.dim() != 2 && input.dim() != 4) ||
      (input_is_4d &&
       (!is_default_or_channels_last_tensor(input) ||
        !is_default_or_channels_last_tensor(out) ||
        input_is_channels_last != out_is_channels_last)) ||
      input_is_default != out_is_default || input.sizes() != out.sizes()) {
    ET_LOG(
        Error,
        "%s: input/output must be matching rank-2 or rank-4 tensors with the same layout",
        "batch_norm_native_f32_out");
    context.fail(Error::InvalidArgument);
    return out;
  }
  const int64_t channels64 = input.size(1);
  if (weight.dim() != 1 || bias.dim() != 1 || running_mean.dim() != 1 ||
      running_var.dim() != 1 || weight.size(0) != channels64 ||
      bias.size(0) != channels64 || running_mean.size(0) != channels64 ||
      running_var.size(0) != channels64) {
    ET_LOG(
        Error,
        "%s: parameter tensors must be rank-1 with channel length",
        "batch_norm_native_f32_out");
    context.fail(Error::InvalidArgument);
    return out;
  }
  int32_t batch = 0, height = 1, width = 1, channels = 0;
  if (!check_int32_within_range(
          context,
          "batch_norm_native_f32_out",
          input.size(0),
          "batch",
          batch) ||
      (input_is_4d &&
       (!check_int32_within_range(
            context,
            "batch_norm_native_f32_out",
            input.size(2),
            "height",
            height) ||
        !check_int32_within_range(
            context,
            "batch_norm_native_f32_out",
            input.size(3),
            "width",
            width))) ||
      !check_int32_within_range(
          context,
          "batch_norm_native_f32_out",
          channels64,
          "channels",
          channels)) {
    return out;
  }
  const size_t scratch_bytes =
      static_cast<size_t>(channels) * sizeof(float32_t) * 2;
  auto scratch_or_error =
      context.allocate_temp(scratch_bytes, kCortexMMveAlignment);
  if (scratch_or_error.error() != Error::Ok ||
      scratch_or_error.get() == nullptr) {
    context.fail(Error::MemoryAllocationFailed);
    return out;
  }
  auto* scratch = static_cast<uint8_t*>(scratch_or_error.get());
  auto* folded_scale = reinterpret_cast<float32_t*>(scratch);
  auto* folded_bias = reinterpret_cast<float32_t*>(
      scratch + static_cast<size_t>(channels) * sizeof(float32_t));
  const auto* weight_ptr = cmsis_const_data_ptr<float32_t>(weight);
  const auto* bias_ptr = cmsis_const_data_ptr<float32_t>(bias);
  const auto* mean_ptr = cmsis_const_data_ptr<float32_t>(running_mean);
  const auto* var_ptr = cmsis_const_data_ptr<float32_t>(running_var);
  // Convert native inference BN parameters to the simpler affine form consumed
  // by both CMSIS-NN and the local fallbacks:
  //
  //   y = (x - mean[c]) * weight[c] / sqrt(var[c] + eps) + bias[c]
  //
  // becomes:
  //
  //   folded_scale[c] = weight[c] / sqrt(var[c] + eps)
  //   folded_bias[c]  = bias[c] - mean[c] * folded_scale[c]
  //   y               = x * folded_scale[c] + folded_bias[c]
  //
  // The temporary arrays live in the ExecuTorch temp allocator and are released
  // after the kernel call returns.
  for (int32_t i = 0; i < channels; ++i) {
    const float scale = static_cast<float>(weight_ptr[i]) /
        std::sqrt(static_cast<float>(var_ptr[i]) + static_cast<float>(eps));
    folded_scale[i] = scale;
    folded_bias[i] = static_cast<float>(bias_ptr[i]) -
        static_cast<float>(mean_ptr[i]) * scale;
  }
  if (input.dim() == 2) {
    apply_batch_norm_2d<float32_t>(input, out, folded_scale, folded_bias);
  } else if (input_is_channels_last) {
    const cmsis_nn_dims input_dims{batch, height, width, channels};
    const auto status = arm_batch_norm_f32(
        cmsis_const_data_ptr<float32_t>(input),
        cmsis_mutable_data_ptr<float32_t>(out),
        folded_scale,
        folded_bias,
        &input_dims,
        ARM_NN_LAYOUT_NHWC);
    if (status != ARM_CMSIS_NN_SUCCESS) {
      ET_LOG(
          Error,
          "%s: arm_batch_norm failed with status %d",
          "batch_norm_native_f32_out",
          static_cast<int>(status));
      context.fail(Error::Internal);
    }
  } else {
    // Same affine transform as CMSIS, but applied directly in NCHW order:
    //
    //   folded_scale[c] = weight[c] / sqrt(var[c] + eps)
    //   folded_bias[c]  = bias[c] - mean[c] * folded_scale[c]
    //   out[n,c,h,w]    = input[n,c,h,w] * folded_scale[c] + folded_bias[c]
    apply_batch_norm_default_order<float32_t>(
        input, out, folded_scale, folded_bias);
  }
  return out;
#endif
}

Tensor& batch_norm_native_f16_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    double eps,
    Tensor& out) {
#if !(defined(ARM_NN_ENABLE_F16) && ARM_NN_ENABLE_F16)
  (void)input;
  (void)weight;
  (void)bias;
  (void)running_mean;
  (void)running_var;
  (void)eps;
  fail_cmsis_float_kernel_unavailable(
      context, "batch_norm_native_f16_out", "float16");
  return out;
#else

  const ScalarType dtype = cmsis_float_scalar_type<float16_t>();
  if (input.scalar_type() != dtype || weight.scalar_type() != dtype ||
      bias.scalar_type() != dtype || running_mean.scalar_type() != dtype ||
      running_var.scalar_type() != dtype || out.scalar_type() != dtype) {
    ET_LOG(
        Error,
        "%s: input/state tensors must all be float16",
        "batch_norm_native_f16_out");
    context.fail(Error::InvalidArgument);
    return out;
  }
  const bool input_is_4d = input.dim() == 4;
  const bool input_is_channels_last =
      input_is_4d && is_channels_last_tensor(input);
  const bool out_is_channels_last = input_is_4d && is_channels_last_tensor(out);
  const bool input_is_default = is_default_dim_order_tensor(input);
  const bool out_is_default = is_default_dim_order_tensor(out);

  // Same dual-layout handling as the f32 path, with rank-2 dense BN handled by
  // the local affine fallback.
  //
  // See batch_norm_native_f32_out above for the layout diagram and rationale.
  if (input.dim() != out.dim() || (input.dim() != 2 && input.dim() != 4) ||
      (input_is_4d &&
       (!is_default_or_channels_last_tensor(input) ||
        !is_default_or_channels_last_tensor(out) ||
        input_is_channels_last != out_is_channels_last)) ||
      input_is_default != out_is_default || input.sizes() != out.sizes()) {
    ET_LOG(
        Error,
        "%s: input/output must be matching rank-2 or rank-4 tensors with the same layout",
        "batch_norm_native_f16_out");
    context.fail(Error::InvalidArgument);
    return out;
  }
  const int64_t channels64 = input.size(1);
  if (weight.dim() != 1 || bias.dim() != 1 || running_mean.dim() != 1 ||
      running_var.dim() != 1 || weight.size(0) != channels64 ||
      bias.size(0) != channels64 || running_mean.size(0) != channels64 ||
      running_var.size(0) != channels64) {
    ET_LOG(
        Error,
        "%s: parameter tensors must be rank-1 with channel length",
        "batch_norm_native_f16_out");
    context.fail(Error::InvalidArgument);
    return out;
  }
  int32_t batch = 0, height = 1, width = 1, channels = 0;
  if (!check_int32_within_range(
          context,
          "batch_norm_native_f16_out",
          input.size(0),
          "batch",
          batch) ||
      (input_is_4d &&
       (!check_int32_within_range(
            context,
            "batch_norm_native_f16_out",
            input.size(2),
            "height",
            height) ||
        !check_int32_within_range(
            context,
            "batch_norm_native_f16_out",
            input.size(3),
            "width",
            width))) ||
      !check_int32_within_range(
          context,
          "batch_norm_native_f16_out",
          channels64,
          "channels",
          channels)) {
    return out;
  }
  const size_t scratch_bytes =
      static_cast<size_t>(channels) * sizeof(float16_t) * 2;
  auto scratch_or_error =
      context.allocate_temp(scratch_bytes, kCortexMMveAlignment);
  if (scratch_or_error.error() != Error::Ok ||
      scratch_or_error.get() == nullptr) {
    context.fail(Error::MemoryAllocationFailed);
    return out;
  }
  auto* scratch = static_cast<uint8_t*>(scratch_or_error.get());
  auto* folded_scale = reinterpret_cast<float16_t*>(scratch);
  auto* folded_bias = reinterpret_cast<float16_t*>(
      scratch + static_cast<size_t>(channels) * sizeof(float16_t));
  const auto* weight_ptr = cmsis_const_data_ptr<float16_t>(weight);
  const auto* bias_ptr = cmsis_const_data_ptr<float16_t>(bias);
  const auto* mean_ptr = cmsis_const_data_ptr<float16_t>(running_mean);
  const auto* var_ptr = cmsis_const_data_ptr<float16_t>(running_var);
  // Same native-BN-to-affine conversion as the f32 path. The computation uses
  // float intermediates to avoid doing sqrt and scale formation in half
  // precision, then stores the folded constants as float16_t because the CMSIS
  // f16 API and fallback loops consume half-precision tensors.
  for (int32_t i = 0; i < channels; ++i) {
    const float scale = static_cast<float>(weight_ptr[i]) /
        std::sqrt(static_cast<float>(var_ptr[i]) + static_cast<float>(eps));
    folded_scale[i] = static_cast<float16_t>(scale);
    folded_bias[i] = static_cast<float16_t>(
        static_cast<float>(bias_ptr[i]) -
        static_cast<float>(mean_ptr[i]) * scale);
  }
  if (input.dim() == 2) {
    apply_batch_norm_2d<float16_t>(input, out, folded_scale, folded_bias);
  } else if (input_is_channels_last) {
    const cmsis_nn_dims input_dims{batch, height, width, channels};
    const auto status = arm_batch_norm_f16(
        cmsis_const_data_ptr<float16_t>(input),
        cmsis_mutable_data_ptr<float16_t>(out),
        folded_scale,
        folded_bias,
        &input_dims,
        ARM_NN_LAYOUT_NHWC);
    if (status != ARM_CMSIS_NN_SUCCESS) {
      ET_LOG(
          Error,
          "%s: arm_batch_norm failed with status %d",
          "batch_norm_native_f16_out",
          static_cast<int>(status));
      context.fail(Error::Internal);
    }
  } else {
    // Same affine fallback as f32, but using float16_t storage.
    apply_batch_norm_default_order<float16_t>(
        input, out, folded_scale, folded_bias);
  }
  return out;
#endif
}

} // namespace native
} // namespace cortex_m
