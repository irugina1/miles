"""qwen3_moe: unfuse transformers>=5.6 batched experts into the per-expert names SGLang expects (via
HF's own reverse conversion), the true-on-policy experts patch, and the rollout-routing-replay hook."""

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.backends.fsdp_utils.adaptations.routing_replay import RoutingReplayAdapter
from miles.backends.fsdp_utils.adaptations.weight_bridge import is_batched_experts_param, unfuse_batched_experts
from miles.backends.fsdp_utils.models.replay_routers import install_qwen3_router_replay


class Qwen3MoeAdapter(ArchAdapter):
    model_types = frozenset({"qwen3_moe"})
    verified = True
    passes_attention_packing_kwargs = True
    routing_replay = RoutingReplayAdapter(
        name="qwen3_moe", module_cls_name="Qwen3MoeTopKRouter", install=install_qwen3_router_replay
    )

    def patch_model(self, model, args):
        from miles.kernels.moe.mx_fused_moe import moe_mx_format

        if not args.true_on_policy_mode:
            if moe_mx_format() is not None:
                # the SGLang plugin would quantize the sampler's experts while the trainer computes bf16
                raise ValueError("MILES_MOE_QUANT requires --true-on-policy-mode")
            return
        # Lazy: pulls sglang fused-MoE kernels that don't exist for every sglang build.
        from miles.backends.fsdp_utils.models.qwen3_moe import apply_qwen3_moe_true_on_policy_patch
        from miles.backends.fsdp_utils.quantization import rollout_quantization_config

        apply_qwen3_moe_true_on_policy_patch(model, rollout_quantization_config(args), moe_mx_format())

    def param_transform(self, name, param):
        return unfuse_batched_experts if is_batched_experts_param(name, param) else None
