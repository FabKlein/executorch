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


class FloatActivationDemo(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, kind: str):
        super().__init__()
        self.dtype = dtype
        self.kind = kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        if self.kind == "tanh":
            return torch.tanh(x)
        if self.kind == "hardswish":
            return torch.nn.functional.hardswish(x)
        raise AssertionError(f"Unsupported activation kind: {self.kind}")


def _example_input(dtype: torch.dtype, kind: str) -> tuple[torch.Tensor]:
    if kind == "tanh":
        return (torch.linspace(-4.0, 4.0, steps=16, dtype=dtype).reshape(4, 4),)
    if kind == "hardswish":
        x = torch.linspace(-5.0, 5.0, steps=1 * 2 * 3 * 4, dtype=dtype).reshape(
            1, 2, 3, 4
        )
        return (x,)
    raise AssertionError(f"Unsupported activation kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", default=".")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float16")
    parser.add_argument("--kind", choices=("tanh", "hardswish"), default="tanh")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    model = FloatActivationDemo(dtype, args.kind).eval()
    example_inputs = _example_input(dtype, args.kind)

    edge_program = export_to_edge(
        model,
        example_inputs,
        edge_compile_config=EdgeCompileConfig(
            preserve_ops=[torch.ops.aten.hardswish.default],
            _check_ir_validity=False,
        ),
    )
    edge_program._edge_programs["forward"] = CortexMPassManager(
        edge_program.exported_program()
    ).transform()
    program = edge_program.to_executorch()

    suffix = "f32" if dtype == torch.float32 else "f16"
    save_pte_program(program, f"activation_{args.kind}_{suffix}_demo", args.output_dir)


if __name__ == "__main__":
    with torch.no_grad():
        main()
