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


class FloatTransposeDemo(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2)


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
        default="float32",
        help="Floating-point dtype to export",
    )
    args = parser.parse_args()

    model = FloatTransposeDemo().eval()
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    example_inputs = (torch.randn((2, 3, 4, 2), dtype=dtype),)

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
    save_pte_program(program, f"transpose_{suffix}_demo", args.output_dir)


if __name__ == "__main__":
    with torch.no_grad():
        main()
