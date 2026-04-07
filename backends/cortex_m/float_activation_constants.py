# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared Cortex-M float activation IDs.

These IDs mirror the CMSIS-NN float activation enum values used by the
Cortex-M float activation op wrappers. Keeping them in one module avoids
repeating undocumented numeric tags across operators and passes.
"""

CMSIS_FLOAT_ACT_NONE = 32
CMSIS_FLOAT_ACT_SIGMOID = 33
CMSIS_FLOAT_ACT_TANH = 34
CMSIS_FLOAT_ACT_RELU = 35
CMSIS_FLOAT_ACT_RELU6 = 36
CMSIS_FLOAT_ACT_HARDSWISH = 37
CMSIS_FLOAT_ACT_LEAKY_RELU = 38

# Deliberately outside the upstream CMSIS-NN enum range; these synthetic tags
# are used only to describe small activation forms in ET passes/operators.
CMSIS_FLOAT_ACT_HARDSIGMOID = 0x100
CMSIS_FLOAT_ACT_HARDTANH = 0x101
