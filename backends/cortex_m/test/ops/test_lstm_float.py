# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize, XfailIfNoCorstone300
from executorch.backends.cortex_m.test.tester import (
    CortexMTester,
    McuTestCase,
    ramp_tensor,
)


def _gate_tensors(
    dtype: torch.dtype, hidden_size: int, input_size: int, gate_offset: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_weights = (
        torch.linspace(
            -0.45 + gate_offset,
            0.45 + gate_offset,
            steps=hidden_size * input_size,
            dtype=torch.float32,
        )
        .reshape(hidden_size, input_size)
        .to(dtype)
        .contiguous()
    )
    hidden_weights = (
        torch.linspace(
            -0.35 + gate_offset,
            0.35 + gate_offset,
            steps=hidden_size * hidden_size,
            dtype=torch.float32,
        )
        .reshape(hidden_size, hidden_size)
        .to(dtype)
        .contiguous()
    )
    bias = (
        torch.linspace(
            -0.15 + gate_offset,
            0.15 + gate_offset,
            steps=hidden_size,
            dtype=torch.float32,
        )
        .to(dtype)
        .contiguous()
    )
    return input_weights, hidden_weights, bias


class CortexMLSTMFloat(torch.nn.Module):
    def __init__(
        self, dtype: torch.dtype, input_size: int, hidden_size: int, time_major: bool
    ):
        super().__init__()
        self.dtype = dtype
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_major = time_major
        self.cell_clip = 0.0

        gate_offsets = {
            "forget": -0.05,
            "input": 0.05,
            "cell": 0.1,
            "output": -0.1,
        }
        for gate_name, gate_offset in gate_offsets.items():
            input_weights, hidden_weights, bias = _gate_tensors(
                dtype, hidden_size, input_size, gate_offset
            )
            self.register_buffer(f"{gate_name}_input_weights", input_weights)
            self.register_buffer(f"{gate_name}_hidden_weights", hidden_weights)
            self.register_buffer(f"{gate_name}_bias", bias)

        suffix = "f32" if dtype == torch.float32 else "f16"
        op_name = f"executorch_exir_dialects_edge__ops_cortex_m_lstm_unidirectional_{suffix}_default"
        self.ops_before_transforms = {op_name: 1}
        self.ops_after_transforms = {op_name: 1}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        if self.dtype == torch.float32:
            return torch.ops.cortex_m.lstm_unidirectional_f32.default(
                x,
                self.forget_input_weights,
                self.forget_hidden_weights,
                self.forget_bias,
                self.input_input_weights,
                self.input_hidden_weights,
                self.input_bias,
                self.cell_input_weights,
                self.cell_hidden_weights,
                self.cell_bias,
                self.output_input_weights,
                self.output_hidden_weights,
                self.output_bias,
                self.time_major,
                self.cell_clip,
            )
        return torch.ops.cortex_m.lstm_unidirectional_f16.default(
            x,
            self.forget_input_weights,
            self.forget_hidden_weights,
            self.forget_bias,
            self.input_input_weights,
            self.input_hidden_weights,
            self.input_bias,
            self.cell_input_weights,
            self.cell_hidden_weights,
            self.cell_bias,
            self.output_input_weights,
            self.output_hidden_weights,
            self.output_bias,
            self.time_major,
            self.cell_clip,
        )


test_cases = {
    "f32_time_major_small": McuTestCase(
        CortexMLSTMFloat(torch.float32, input_size=4, hidden_size=3, time_major=True),
        (ramp_tensor(-0.5, 0.5, (3, 1, 4)).to(torch.float32),),
    ),
    "f32_batch_major_batch2": McuTestCase(
        CortexMLSTMFloat(torch.float32, input_size=4, hidden_size=3, time_major=False),
        (ramp_tensor(-0.4, 0.4, (2, 3, 4)).to(torch.float32),),
    ),
    "f16_time_major_small": McuTestCase(
        CortexMLSTMFloat(torch.float16, input_size=4, hidden_size=3, time_major=True),
        (ramp_tensor(-0.5, 0.5, (3, 1, 4)).to(torch.float16),),
    ),
    "f16_batch_major_batch2": McuTestCase(
        CortexMLSTMFloat(torch.float16, input_size=4, hidden_size=3, time_major=False),
        (ramp_tensor(-0.4, 0.4, (2, 3, 4)).to(torch.float16),),
    ),
}


def _float_compare_kwargs(test_case):
    if test_case.model.dtype == torch.float16:
        return {"atol": 1.2e-1, "rtol": 5e-2}
    return {"atol": 5e-5, "rtol": 5e-5}


@parametrize("test_case", test_cases)
def test_dialect_lstm_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.check_count(test_case.model.ops_before_transforms)
    tester.run_passes()
    tester.check_count(test_case.model.ops_after_transforms)
    tester.run_method_and_compare_outputs(
        inputs=test_case.example_inputs,
        **_float_compare_kwargs(test_case),
    )


@XfailIfNoCorstone300
@parametrize("test_case", test_cases)
def test_implementation_lstm_float(test_case):
    tester = CortexMTester(test_case.model, test_case.example_inputs)
    tester.export()
    tester.to_edge()
    tester.run_passes()
    tester.to_executorch()
    tester.serialize()
    tester.run_method_and_compare_outputs(
        inputs=test_case.example_inputs,
        **_float_compare_kwargs(test_case),
    )
