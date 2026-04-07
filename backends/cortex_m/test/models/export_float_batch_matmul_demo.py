#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import torch

import executorch.backends.cortex_m.ops.operators  # noqa: F401
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.exir import EdgeCompileConfig
from executorch.extension.export_util.utils import export_to_edge, save_pte_program


def ramp_tensor(
    start: float, end: float, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    return torch.linspace(start, end, steps=numel, dtype=dtype).reshape(shape)


class FloatBatchMatmulDemo(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("lhs", ramp_tensor(-1.0, 1.0, (1, 2, 3), dtype=dtype))
        self.register_buffer("rhs", ramp_tensor(-0.5, 1.5, (1, 3, 4), dtype=dtype))

    def forward(self) -> torch.Tensor:
        return torch.bmm(self.lhs, self.rhs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", default=".")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    model = FloatBatchMatmulDemo(dtype).eval()
    example_inputs: tuple[()] = ()

    edge_program = export_to_edge(
        model,
        example_inputs,
        edge_compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    edge_program._edge_programs["forward"] = CortexMPassManager(
        edge_program.exported_program()
    ).transform()
    program = edge_program.to_executorch()

    suffix = "f32" if dtype == torch.float32 else "f16"
    save_pte_program(program, f"batch_matmul_{suffix}_demo", args.output_dir)


if __name__ == "__main__":
    with torch.no_grad():
        main()
