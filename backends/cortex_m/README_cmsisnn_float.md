<!--
 Copyright (c) Meta Platforms, Inc. and affiliates.
 All rights reserved.
 Copyright 2025-2026 Arm Limited and/or its affiliates.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree.
-->

# CMSIS-NN Float Notes

This note summarizes the current Cortex-M float (`float32` / `float16`)
bring-up flow for ExecuTorch.

## Scope

The Cortex-M ET float path exists and is usable, but it is still more
experimental than the quantized path:

- operator coverage is narrower
- lowering is more sensitive to export graph shape
- backend capability configuration must stay aligned between build time and
  Python lowering time

It also requires an up-to-date CMSIS-NN checkout with float support enabled.
At the time of writing, this float support is not available from the upstream
CMSIS-NN main branch used by default in many builds, so a local source checkout
must be provided explicitly through `--cmsis_nn_local_path=...`.

## Initial setup

The usual Arm/FVP dependencies still come from the Arm setup flow described in
the main documentation. Once the toolchains and simulator are available, a
typical float build uses:

- GCC: recent Arm GNU Toolchain, **GCC 14 or newer**
- preferred Arm toolchains:
  - Arm Compiler 6, for example **6.24**
  - Arm Toolchain for Embedded (LLVM), for example **LLVM 22**

Example backend build with GCC and local CMSIS-NN:

```bash
cd /path/to/executorch
source venv/bin/activate
export PATH=/path/to/arm-gnu-toolchain/bin:$PATH

bash backends/arm/scripts/build_executorch.sh \
  --toolchain=arm-none-eabi-gcc \
  --et_build_root=/tmp/et_cortexm_f16 \
  --cmsis_nn_local_path=/path/to/CMSIS-NN \
  --cmsis_nn_enable_f16
```

For `float32`, replace `--cmsis_nn_enable_f16` with `--cmsis_nn_enable_f32`.
For builds that need both dtypes, pass both flags.

## Notebook walkthrough

For a step-by-step interactive walkthrough of the full MobileNetV3 float16 flow
on Corstone-300, see:

- [`examples/arm/cortex_m_mv3_f16_example.ipynb`](../../examples/arm/cortex_m_mv3_f16_example.ipynb)

The notebook covers sample visualization, PyTorch inference, Cortex-M lowering
inspection, `.pte` export, semihosting runner build, direct FVP execution, and
PyTorch vs FVP output comparison.

## Quick start

For a fast environment sanity-check before running a full model, start with one
of the small float operator tests from the local ExecuTorch virtual environment:

```bash
cd /path/to/executorch
source venv/bin/activate
pytest backends/cortex_m/test/ops/test_add_float.py -q
```

This is a good first check when validating a new CMSIS-NN checkout, toolchain,
or Cortex-M float build tree.

## Build-time capability propagation

The Cortex-M backend uses a generated capability artifact so Python
lowering does not silently drift out of sync with the compiled backend.

### Build side

`backends/cortex_m/CMakeLists.txt` derives:

- `EXECUTORCH_CORTEX_M_ENABLE_FLOAT32`
- `EXECUTORCH_CORTEX_M_ENABLE_FLOAT16`

from:

- `ARM_NN_ENABLE_F32`
- `ARM_NN_ENABLE_F16`

and writes a generated artifact under the build tree, for example:

- `arm_test/cmake-out/backends/cortex_m/float_capabilities.json`

Example artifact:

```json
{
  "enable_f32": false,
  "enable_f16": true
}
```

### Python pass side

The Cortex-M pass manager reads this generated JSON through:

- `backends/cortex_m/passes/float_capabilities.py`

and uses it to gate float lowering.

This prevents cases like:

- building only `f16`
- but accidentally lowering a model to `cortex_m::*_f32`

The intended flow is:

1. build backend with the desired float dtype(s)
2. point the export/lowering step at the generated JSON artifact
3. lower only to the float ops that were actually compiled

## Toolchain notes

### GCC

For CMSIS-NN float, **do not use old GCC releases**.

- GCC 13 and older are not supported for this path
- use **GCC 14+**

Symptoms of older GCC toolchains include:

- `__ARM_undef`
- invalid codegen
- broken float kernel builds or runtime behavior

### Preferred toolchains

For the best experience on Cortex-M float, prefer:

- **Arm Compiler 6** (for example 6.24)
- **Arm Toolchain for Embedded / LLVM** (for example LLVM 22)

## Currently supported float operators

The current Cortex-M float path is strongest on CNN-style models. The practical
set of float operators currently exercised by tests and model flows includes:

- `conv2d`
- `depthwise_conv2d`
- `transpose_conv2d`
- `linear`
- `batch_matmul`
- `avg_pool2d`
- `max_pool2d`
- elementwise `add`, `mul`, `minimum`, `maximum`
- activation-style ops such as `relu`, `relu6`, `hardtanh`, `hardswish`,
  `hardsigmoid`, `sigmoid`, and `tanh`
- `softmax`
- `batch_norm` / `batch_norm_native`
- `pad`

Support is still graph-shape dependent. A model can use only operators from the
list above and still leave residual `aten::*` nodes if layout or decomposition
patterns are not yet normalized the way the backend expects.



## Manual operator testing example

The focused operator tests under:

- `backends/cortex_m/test/ops/`

are useful when narrowing backend issues without running a full model.

### Example: float16 clamped addition

Use:

- `backends/cortex_m/test/ops/test_add_float.py`

This test can:

- export a tiny add model to PTE
- prepare semihosting inputs and reference outputs
- build the runner
- run the operator on FVP

Example wrapper-style invocation depends on the current test helper flow, but
the manual workflow is always:

1. export/generate the operator test assets
2. build the runner for the chosen toolchain
3. run FVP directly in semihosting mode
4. compare `out-0.bin` with the generated reference output

### Manual FVP command shape

Typical direct semihosting command:

```bash
cd /path/to/executorch
source examples/arm/arm-scratch/setup_path.sh

FVP_Corstone_SSE-300_Ethos-U55 \
  -C ethosu.num_macs=128 \
  -C mps3_board.visualisation.disable-visualisation=1 \
  -C mps3_board.telnetterminal0.start_telnet=0 \
  -C mps3_board.uart0.out_file='-' \
  -C cpu0.semihosting-enable=1 \
  -C cpu0.semihosting-stack_base=0 \
  -C cpu0.semihosting-heap_limit=0 \
  -C cpu0.semihosting-cwd=/path/to/run_dir \
  -C "ethosu.extra_args=--fast" \
  -C "cpu0.semihosting-cmd_line=executor_runner -m demo.pte -o out -i i0.bin -i i1.bin" \
  -a /path/to/arm_executor_runner \
```

### Output verification

To verify concordance:

1. compare the FVP-produced `out-0.bin` with the reference output generated
   during export
2. check:
   - `max_abs`
   - `mean_abs`
   - expected class or expected tensor values
3. confirm the operator count / lowering path if needed with the graph
   inspection helpers

For simple operator tests, the most useful success criterion is:

- exact or near-exact numeric agreement with the generated reference

This is usually enough to confirm that:

- export
- lowering
- runtime kernel
- semihosting runner

are all aligned for that operator.

## Troubleshooting

- `EXECUTORCH_CORTEX_M_FLOAT_CAPABILITIES_FILE` unset or stale:
  the lowering passes will either disable float lowering entirely or read an
  old build artifact. Rebuild the Cortex-M backend first, then point export to
  the matching `float_capabilities.json` in the build tree.
- Float dtype disabled in the backend build:
  if the model asks for `float16` or `float32` lowering but the matching
  backend capability is off, lowering should fail early. Check the generated
  capability file and the `--cmsis_nn_enable_f16` / `--cmsis_nn_enable_f32`
  build flags.
- Changed CMSIS-NN kernels but no effect in the final ELF:
  rebuild the main ExecuTorch Arm build tree first, then rebuild or relink the
  final runner. Rebuilding only the runner directory is often not enough.
- Residual `_clone_dim_order` or `aten::*` nodes:
  this usually means the model still needs an extra layout/decomposition
  normalization step. The semihosting wrappers print lowered Cortex-M operators
  and residual `aten` / `dim_order` ops to make this easier to spot.
- Semihosting file I/O issues on non-FVP simulators:
  some local simulators do not behave like FVP for `stdio`-based semihosting.
  In that case, use a bundled `.bpte` runner instead of relying on host file
  access at runtime.
- FVP performance expectations:
  FVP is useful for functionality and relative flow validation, but not for
  cycle-accurate CPU performance conclusions. Use FPGA or real hardware for
  meaningful float kernel timing.
