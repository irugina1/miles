"""Qwen3-MoE experts for the qwen3_moe true-on-policy contract: SGLang's Triton fused-MoE kernels at
its deterministic config, with Triton backward kernels. The router stays transformers'
``Qwen3MoeTopKRouter``, which already matches SGLang's on-policy routing."""

import torch
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from miles.kernels.moe.fused_experts import fused_experts


class Qwen3MoeTrueOnPolicyExperts(Qwen3MoeExperts):
    """Batched experts (``gate_up_proj`` [E, 2I, H] with gate rows first, ``down_proj`` [E, H, I])
    computed with the same kernels and layout as SGLang's Triton MoE runner."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # the kernels run with filter_expert=False, as SGLang's runner does, so they do not skip
        # invalid ids; an id outside [0, E) would read past the expert weights
        num_experts = self.gate_up_proj.shape[0]
        torch._assert_async(((top_k_index >= 0) & (top_k_index < num_experts)).all())
        return fused_experts(
            hidden_states.to(torch.bfloat16).contiguous(),
            self.gate_up_proj,
            self.down_proj,
            top_k_weights,
            top_k_index,
        )


def apply_qwen3_moe_true_on_policy_patch(model) -> int:
    """Swap every ``Qwen3MoeExperts`` in ``model`` for the kernel-aligned version; returns the count."""
    patched = 0
    for module in model.modules():
        if isinstance(module, Qwen3MoeExperts):
            module.__class__ = Qwen3MoeTrueOnPolicyExperts
            patched += 1
    if patched == 0:
        raise TypeError("Expected a Qwen3 MoE model with Qwen3MoeExperts modules")
    return patched
