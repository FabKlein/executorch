#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

import executorch.backends.cortex_m.ops.operators  # noqa: F401
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.exir import EdgeCompileConfig
from executorch.extension.export_util.utils import export_to_edge
from executorch.extension.export_util.utils import save_pte_program


def ramp_tensor(
    start: float, end: float, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    return torch.linspace(start, end, steps=numel, dtype=dtype).reshape(shape)


class FloatSoftmaxDemo(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, dim: int, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("x", ramp_tensor(-4.0, 4.0, shape, dtype=dtype))

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.x, dim=self.dim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        default=".",
        help="Directory where the .pte should be written",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float16",
        help="Floating-point dtype to export",
    )
    parser.add_argument(
        "--shape",
        default="16",
        help="Comma-separated tensor shape, e.g. '16' or '2,3,4'",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=-1,
        help="Softmax dimension. The Cortex-M float path currently supports only the last dimension.",
    )
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    shape = tuple(int(part) for part in args.shape.split(",") if part)

    model = FloatSoftmaxDemo(dtype=dtype, dim=args.dim, shape=shape).eval()
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
    save_pte_program(program, f"softmax_{suffix}_demo", args.output_dir)


if __name__ == "__main__":
    with torch.no_grad():
        main()
