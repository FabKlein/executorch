#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
from pathlib import Path


# This helper is used at CMake configure time to strip Cortex-M float operator
# schemas from operators.yaml when the corresponding f32/f16 backend support is
# disabled. This keeps selected-op registration aligned with the actual backend
# build and avoids advertising unsupported float kernels.


def _is_enabled(func_line: str, enable_f32: bool, enable_f16: bool) -> bool:
    if "_f32" in func_line:
        return enable_f32
    if "_f16" in func_line:
        return enable_f16
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--enable-f32", choices=["ON", "OFF"], required=True)
    parser.add_argument("--enable-f16", choices=["ON", "OFF"], required=True)
    args = parser.parse_args()

    enable_f32 = args.enable_f32 == "ON"
    enable_f16 = args.enable_f16 == "ON"

    text = Path(args.input).read_text()
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.strip() == "" and current:
            blocks.append("".join(current))
            current = []
        else:
            current.append(line)
    if current:
        blocks.append("".join(current))

    kept_blocks: list[str] = []
    for block in blocks:
        func_line = None
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("- func:"):
                func_line = stripped
                break
        if func_line is None or _is_enabled(func_line, enable_f32, enable_f16):
            kept_blocks.append(block.rstrip())

    Path(args.output).write_text("\n\n".join(kept_blocks) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
