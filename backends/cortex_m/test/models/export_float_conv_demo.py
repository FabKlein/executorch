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


def ramp_tensor(start: float, end: float, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.linspace(
        start, end, steps=int(torch.tensor(shape).prod()), dtype=torch.float32
    ).reshape(shape)


class Conv2DFloatDemo(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, *args, bias: bool = False, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(*args, bias=bias, **kwargs)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.linspace(
                    -1.0,
                    1.0,
                    steps=self.conv.weight.numel(),
                    dtype=torch.float32,
                ).reshape_as(self.conv.weight)
            )
            if bias:
                self.conv.bias.copy_(
                    torch.linspace(
                        -0.25,
                        0.25,
                        steps=self.conv.bias.numel(),
                        dtype=torch.float32,
                    )
                )
        self.conv = self.conv.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv2DFloatX3Demo(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        for idx, conv in enumerate((self.conv1, self.conv2, self.conv3)):
            with torch.no_grad():
                conv.weight.copy_(
                    torch.linspace(
                        -1.0 + idx * 0.2,
                        1.0 + idx * 0.2,
                        steps=conv.weight.numel(),
                        dtype=torch.float32,
                    ).reshape_as(conv.weight)
                )
        self.conv1 = self.conv1.to(dtype)
        self.conv2 = self.conv2.to(dtype)
        self.conv3 = self.conv3.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DepthwiseConv2DFloatDemo(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, *args, bias: bool = False, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(*args, bias=bias, **kwargs)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.linspace(
                    -1.0,
                    1.0,
                    steps=self.conv.weight.numel(),
                    dtype=torch.float32,
                ).reshape_as(self.conv.weight)
            )
            if bias:
                self.conv.bias.copy_(
                    torch.linspace(
                        -0.25,
                        0.25,
                        steps=self.conv.bias.numel(),
                        dtype=torch.float32,
                    )
                )
        self.conv = self.conv.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def build_variant(variant: str, dtype: torch.dtype, input_scale: float = 1.0):
    if variant == "conv_x3":
        model = Conv2DFloatX3Demo(dtype).eval()
        example_inputs = (
            ramp_tensor(-1 * input_scale, 1 * input_scale, (1, 3, 8, 8))
            .to(dtype)
            .contiguous(memory_format=torch.channels_last),
        )
        pte_base = "conv_x3"
    elif variant == "conv_bias":
        model = Conv2DFloatDemo(dtype, 5, 4, (1, 2), bias=True).eval()
        example_inputs = (
            ramp_tensor(-3 * input_scale, 3 * input_scale, (1, 5, 10, 10))
            .to(dtype)
            .contiguous(memory_format=torch.channels_last),
        )
        pte_base = "conv_bias"
    elif variant == "depthwise_padding":
        model = DepthwiseConv2DFloatDemo(dtype, 2, 2, 5, padding=2, groups=2).eval()
        example_inputs = (
            ramp_tensor(-2 * input_scale, 2 * input_scale, (1, 2, 5, 5))
            .to(dtype)
            .contiguous(memory_format=torch.channels_last),
        )
        pte_base = "depthwise_padding"
    elif variant == "depthwise_bias":
        model = DepthwiseConv2DFloatDemo(
            dtype, 3, 3, 3, padding=1, groups=3, bias=True
        ).eval()
        example_inputs = (
            ramp_tensor(-3 * input_scale, 3 * input_scale, (1, 3, 6, 6))
            .to(dtype)
            .contiguous(memory_format=torch.channels_last),
        )
        pte_base = "depthwise_bias"
    else:
        raise ValueError(f"Unsupported variant: {variant}")
    return model, example_inputs, pte_base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", default=".")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float16")
    parser.add_argument(
        "--variant",
        choices=("conv_x3", "conv_bias", "depthwise_padding", "depthwise_bias"),
        default="conv_x3",
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        default=1.0,
        help="Scale factor applied to generated input sample values.",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    model, example_inputs, pte_base = build_variant(
        args.variant, dtype, input_scale=args.input_scale
    )

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
    save_pte_program(program, f"{pte_base}_{suffix}_demo", args.output_dir)


if __name__ == "__main__":
    with torch.no_grad():
        main()
