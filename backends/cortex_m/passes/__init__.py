# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .cortex_m_pass import CortexMPass  # noqa  # usort: skip
from .activation_fusion_pass import ActivationFusionPass  # noqa
from .aten_to_cortex_m_pass import AtenToCortexMPass  # noqa
from .clamp_hardswish_pass import ClampHardswishPass  # noqa
from .collapse_float_activation_decomposition_pass import (  # noqa
    CollapseFloatActivationDecompositionPass,
)
from .convert_to_cortex_m_pass import ConvertToCortexMPass  # noqa
from .decompose_hardswish_pass import DecomposeHardswishPass  # noqa
from .decompose_mean_pass import DecomposeMeanPass  # noqa
from .float_activation_rewrite_pass import FloatActivationRewritePass  # noqa
from .float_op_rewrite_pass import FloatOpRewritePass  # noqa
from .float_pool_rewrite_pass import FloatPoolRewritePass  # noqa
from .fold_batch_norm_into_conv_pass import FoldBatchNormIntoConvPass  # noqa
from .fold_batch_norm_into_linear_pass import FoldBatchNormIntoLinearPass  # noqa
from .pack_float_conv_weights_pass import PackFloatConvWeightsPass  # noqa
from .quantized_clamp_activation_pass import QuantizedClampActivationPass  # noqa
from .remove_unused_constant_placeholders_pass import (  # noqa
    RemoveUnusedConstantPlaceholdersPass,
)
from .replace_quant_nodes_pass import ReplaceQuantNodesPass  # noqa
from .cortex_m_pass_manager import CortexMPassManager  # noqa  # usort: skip
