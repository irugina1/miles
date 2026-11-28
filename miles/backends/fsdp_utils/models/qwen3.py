"""Qwen3 dense model math required by the formal true-on-policy contract."""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


class Qwen3FinalRMSNorm(Qwen3RMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(torch.bfloat16) * hidden_states.to(torch.bfloat16)


_FP32_SYNC_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)


def resolve_qwen3_dense_sync_dtype(name: str, checkpoint_dtype: torch.dtype) -> torch.dtype:
    if name == "model.embed_tokens.weight" or name.endswith(_FP32_SYNC_SUFFIXES):
        return torch.float32
    return checkpoint_dtype


def apply_qwen3_dense_true_on_policy_patch(model) -> bool:
    base_model = getattr(
        model, "model", model
    )  # config-access-exempt: HF wrappers differ in base-model and final-norm layout
    final_norm = getattr(
        base_model, "norm", None
    )  # config-access-exempt: HF wrappers differ in base-model and final-norm layout
    if isinstance(final_norm, Qwen3FinalRMSNorm):
        return False
    if not isinstance(final_norm, Qwen3RMSNorm):
        raise TypeError("Expected a Qwen3 model with a final Qwen3RMSNorm")
    final_norm.__class__ = Qwen3FinalRMSNorm
    return True


# every projection of a decoder layer; embeddings, lm_head and norms stay bf16
QUANTIZED_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def apply_qwen3_dense_linear_quant(model, fmt: str) -> int:
    """Swap every decoder layer's projections to Miles' quantized linear in format ``fmt``, the kernels the
    SGLang plugin runs on the engines (miles/kernels/quant/mx_linear.py). Returns the count."""
    from miles.kernels.quant.mx_linear import swap_to_mx_linear

    base_model = getattr(model, "model", model)  # config-access-exempt: HF wrappers differ in base-model layout
    swapped = 0
    for layer in base_model.layers:
        for block in (layer.self_attn, layer.mlp):
            for name in QUANTIZED_PROJECTIONS:
                linear = getattr(block, name, None)  # config-access-exempt: q/k/v/o live in self_attn, the rest in mlp
                if linear is not None:
                    swap_to_mx_linear(linear, fmt)
                    swapped += 1
    expected = len(base_model.layers) * len(QUANTIZED_PROJECTIONS)
    if swapped != expected:
        raise RuntimeError(f"quantized {swapped} Qwen3 projections, expected {expected}")
    return swapped
