# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch


def get_packed_n_block_cols(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 8
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported packed Cortex-M float dtype: {dtype}")


def pack_nt_t_weights_to_nt_n_packed(
    rhs_nt: torch.Tensor,
    rhs_rows: int,
    rhs_cols: int,
) -> torch.Tensor:
    """
    Pack a logical RHS matrix [rows, cols] into the CMSIS NT_N format:

      [rows, cols]
          ->
      [block][k][lane]

    where rows are grouped in blocks of 8 (f16) or 4 (f32).
    """

    block_cols = get_packed_n_block_cols(rhs_nt.dtype)
    rhs_blocks = (rhs_rows + block_cols - 1) // block_cols
    packed = torch.zeros(
        rhs_blocks * rhs_cols * block_cols,
        dtype=rhs_nt.dtype,
        device=rhs_nt.device,
    )
    rhs_nt_2d = rhs_nt.reshape(rhs_rows, rhs_cols)
    packed_3d = packed.view(rhs_blocks, rhs_cols, block_cols)
    for block in range(rhs_blocks):
        col_base = block * block_cols
        for lane in range(block_cols):
            col = col_base + lane
            if col < rhs_rows:
                packed_3d[block, :, lane].copy_(rhs_nt_2d[col, :])
    return packed.contiguous()


def unpack_nt_n_packed_weights(
    packed_weight: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """
    Undo the CMSIS NT_N packed layout back into a normal [O, I] matrix.
    """

    block_cols = get_packed_n_block_cols(packed_weight.dtype)
    rhs_blocks = (out_features + block_cols - 1) // block_cols
    packed_3d = packed_weight.reshape(rhs_blocks, in_features, block_cols)
    unpacked = torch.zeros(
        (out_features, in_features),
        dtype=packed_weight.dtype,
        device=packed_weight.device,
    )
    for block in range(rhs_blocks):
        col_base = block * block_cols
        for lane in range(block_cols):
            col = col_base + lane
            if col < out_features:
                unpacked[col, :].copy_(packed_3d[block, :, lane])
    return unpacked.contiguous()
