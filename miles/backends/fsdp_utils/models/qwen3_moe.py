"""Qwen3-MoE experts for the qwen3_moe true-on-policy contract: SGLang's Triton fused-MoE kernels at
its deterministic config, with Triton backward kernels. The router stays transformers'
``Qwen3MoeTopKRouter``, which already matches SGLang's on-policy routing. Quantized experts (block
FP8, MXFP8, MXFP4, NVFP4) run the sampler's kernels with a dequantized backward; see
examples/infra_features/true_on_policy/README.md."""

import logging

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from miles.backends.fsdp_utils.quantization import is_quantized_module
from miles.kernels.moe import mx_fused_moe
from miles.kernels.moe.fused_experts import fused_experts, quantize_experts_fp8

logger = logging.getLogger(__name__)


class Qwen3MoeTrueOnPolicyExperts(Qwen3MoeExperts):
    """Batched experts (``gate_up_proj`` [E, 2I, H] with gate rows first, ``down_proj`` [E, H, I])
    computed with the same kernels and layout as SGLang's Triton MoE runner."""

    # Set by apply_qwen3_moe_true_on_policy_patch when the rollout serves block-FP8 experts.
    fp8_block_shape: list[int] | None = None
    # Set by apply_qwen3_moe_true_on_policy_patch for microscaling W+A experts.
    mx_format: str | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        w13, w2 = self.gate_up_proj, self.down_proj
        # the kernels run with filter_expert=False, as SGLang's runner does, so they do not skip
        # invalid ids; an id outside [0, E) would read past the expert weights
        torch._assert_async(((top_k_index >= 0) & (top_k_index < w13.shape[0])).all())
        fp8 = {}
        if self.mx_format is not None:
            fp8 = dict(
                w1_mx=mx_fused_moe.quantize_experts(w13, self.mx_format),
                w2_mx=mx_fused_moe.quantize_experts(w2, self.mx_format),
            )
        elif self.fp8_block_shape is not None:
            fp8 = dict(
                w1_fp8=quantize_experts_fp8(w13, self.fp8_block_shape),
                w2_fp8=quantize_experts_fp8(w2, self.fp8_block_shape),
            )
        return fused_experts(
            hidden_states.to(torch.bfloat16).contiguous(),
            w13,
            w2,
            top_k_weights,
            top_k_index,
            **fp8,
        )


def apply_qwen3_moe_true_on_policy_patch(
    model, quantization_config: dict | None = None, mx_format: str | None = None
) -> int:
    """Swap every ``Qwen3MoeExperts`` in ``model`` for the kernel-aligned version; returns the count.

    ``quantization_config`` is the rollout checkpoint's: experts it stores in block FP8 compute in
    block FP8 here too. ``mx_format`` runs every expert in that microscaling format instead."""
    if quantization_config is not None and mx_format is not None:
        raise ValueError("Use either an FP8 rollout checkpoint or MILES_MOE_QUANT, not both")
    patched = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen3MoeExperts):
            module.__class__ = Qwen3MoeTrueOnPolicyExperts
            module.mx_format = mx_format
            if quantization_config is not None and is_quantized_module(quantization_config, f"{name}.0.gate_proj"):
                module.fp8_block_shape = list(quantization_config["weight_block_size"])
            patched += 1
    if patched == 0:
        raise TypeError("Expected a Qwen3 MoE model with Qwen3MoeExperts modules")
    if mx_format is not None:
        logger.info(f"Qwen3-MoE true-on-policy experts run the {mx_format} microscaling kernels")
    return patched
