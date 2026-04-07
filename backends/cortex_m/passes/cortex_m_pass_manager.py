# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Any, Optional, Type

from executorch.backends.arm._passes import (
    DeduplicateGetAttrPass,
    FoldAndAnnotateQParamsPass,
    ScalarsToAttributePass,
)
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.backends.transforms.addmm_mm_to_linear import AddmmToLinearTransform
from executorch.backends.transforms.convert_conv1d_to_conv2d_pass import (
    ConvertConv1dToConv2dPass,
)
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.backends.transforms.replace_squeeze_unsqueeze_with_view import (
    ReplaceSqueezeAndUnsqueezeWithViewPass,
)
from executorch.exir.pass_base import ExportPass
from executorch.exir.pass_manager import PassManager
from executorch.exir.program._program import _transform, lift_constant_tensor_pass
from torch.export import ExportedProgram

from .activation_fusion_pass import ActivationFusionPass
from .aten_to_cortex_m_pass import AtenToCortexMPass
from .bypass_flatten_for_linear_pass import BypassFlattenForLinearPass
from .clamp_hardswish_pass import ClampHardswishPass
from .collapse_float_activation_decomposition_pass import (
    CollapseFloatActivationDecompositionPass,
)
from .convert_to_cortex_m_pass import ConvertToCortexMPass
from .decompose_hardswish_pass import DecomposeHardswishPass
from .decompose_mean_pass import DecomposeMeanPass
from .explicit_layout_pass import (
    CortexMCanonicalizeViewCopyPermutePass,
    CortexMReplaceOpsWithChannelsLastVariants,
    ValidateCortexMExplicitLayoutPass,
)
from .float_activation_rewrite_pass import FloatActivationRewritePass
from .float_capabilities import (
    CortexMFloatCapabilities,
    get_cortex_m_float_capabilities,
)
from .float_op_rewrite_pass import FloatOpRewritePass
from .float_pool_rewrite_pass import FloatPoolRewritePass
from .fold_batch_norm_into_conv_pass import FoldBatchNormIntoConvPass
from .fold_batch_norm_into_linear_pass import FoldBatchNormIntoLinearPass
from .matmul_to_bmm_pass import MatmulToBmmPass
from .normalize_dim_order_pass import NormalizeDimOrderPass
from .pack_float_conv_weights_pass import PackFloatConvWeightsPass
from .quantized_clamp_activation_pass import QuantizedClampActivationPass
from .remove_redundant_clone_dim_order_pass import RemoveRedundantCloneDimOrderPass
from .remove_unused_constant_placeholders_pass import (
    RemoveUnusedConstantPlaceholdersPass,
)
from .replace_quant_nodes_pass import ReplaceQuantNodesPass

PassClass = Type[ExportPass]


class CortexMPassManager(PassManager):
    legacy_pass_list: list[PassClass] = [
        # Preserve the current upstream quantized lowering sequence.
        RemoveGetItemPass,
        FoldAndAnnotateQParamsPass,
        ReplaceScalarWithTensorArgPass,
        ReplaceQuantNodesPass,
        ActivationFusionPass,
        QuantizedClampActivationPass,
        DecomposeHardswishPass,
        AtenToCortexMPass,
        # Float canonicalization starts after quantized aten nodes are lowered.
        AddmmToLinearTransform,
        DecomposeMeanPass,
        NormalizeDimOrderPass,
        CollapseFloatActivationDecompositionPass,
        # Rewrite simple float ops and tag activations that can be fused.
        FloatOpRewritePass,
        # Fold dense BN while linear weights still have their ordinary rank-2
        # [out, in] shape, before ConvertToCortexMPass packs them.
        FoldBatchNormIntoLinearPass,
        NormalizeDimOrderPass,
        BypassFlattenForLinearPass,
        FloatPoolRewritePass,
        FloatActivationRewritePass,
        CollapseFloatActivationDecompositionPass,
        # Conv, transpose-conv, BMM and linear packing require graph surgery.
        ConvertToCortexMPass,
        # Folding may expose a conv that earlier metadata/layout checks could
        # not lower, so normalize and run the idempotent conversion once more.
        FoldBatchNormIntoConvPass,
        NormalizeDimOrderPass,
        ConvertToCortexMPass,
        # Packing must follow BN folding, which requires unpacked OHWI weights.
        PackFloatConvWeightsPass,
        RemoveUnusedConstantPlaceholdersPass,
        BypassFlattenForLinearPass,
        RemoveRedundantCloneDimOrderPass,
    ]

    explicit_layout_pass_list: list[PassClass] = [
        RemoveGetItemPass,
        FoldAndAnnotateQParamsPass,
        ReplaceScalarWithTensorArgPass,
        ActivationFusionPass,
        QuantizedClampActivationPass,
        DecomposeHardswishPass,
        ConvertConv1dToConv2dPass,
        CortexMReplaceOpsWithChannelsLastVariants,
        ReplaceSqueezeAndUnsqueezeWithViewPass,
        CortexMCanonicalizeViewCopyPermutePass,
        RemovePermutesAroundElementwiseOps,
        CortexMCanonicalizeViewCopyPermutePass,
        ValidateCortexMExplicitLayoutPass,
        ReplaceQuantNodesPass,
        AtenToCortexMPass,
    ]

    pass_list = legacy_pass_list

    pass_list_transform_for_annotation: list[PassClass] = [
        ScalarsToAttributePass,
        ReplaceScalarWithTensorArgPass,
        ClampHardswishPass,
        DecomposeMeanPass,
        MatmulToBmmPass,
        DeduplicateGetAttrPass,
    ]

    def __init__(
        self,
        exported_program: ExportedProgram | None,
        passes: Optional[list[PassClass]] = None,
        target_config: Optional[CortexMTargetConfig] = None,
        use_explicit_layout: bool = False,
        capabilities: Optional[CortexMFloatCapabilities] = None,
    ) -> None:
        """Initialize the Cortex-M pass manager.

        The explicit-layout sequence remains the upstream quantized pipeline.
        Float lowering currently uses the legacy dim-order representation.
        """
        super().__init__(passes=[])
        self.exported_program = exported_program
        default_passes = (
            self.explicit_layout_pass_list
            if use_explicit_layout
            else self.legacy_pass_list
        )
        self.passes: list[PassClass] = (  # type: ignore[assignment]
            passes if passes is not None else default_passes
        )
        self.target_config = target_config or CortexMTargetConfig(cpu=CortexM.M55)
        self.capabilities = capabilities or get_cortex_m_float_capabilities()

    def transform_for_annotation(self, model):
        for pass_cls in self.pass_list_transform_for_annotation:
            model = pass_cls().call(model).graph_module
        return model

    def transform(self) -> ExportedProgram:
        exported_program = self.exported_program
        if not isinstance(exported_program, ExportedProgram):
            raise ValueError(
                f"{type(self).__name__}.transform() needs a real ExportedProgram, "
                f"got {exported_program!r}"
            )

        for pass_cls in self.passes:
            if not isinstance(pass_cls, type):
                raise ValueError(
                    f"{type(self).__name__} expects pass classes, not instances; "
                    f"got {pass_cls!r}"
                )

            signature = inspect.signature(pass_cls)
            kwargs: dict[str, Any] = {}
            if "exported_program" in signature.parameters:
                kwargs["exported_program"] = exported_program
            if "target_config" in signature.parameters:
                kwargs["target_config"] = self.target_config
            if "capabilities" in signature.parameters:
                kwargs["capabilities"] = self.capabilities

            exported_program = _transform(exported_program, pass_cls(**kwargs))

        # Float packing and folding can create new constants.
        return lift_constant_tensor_pass(exported_program)
