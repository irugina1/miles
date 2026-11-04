"""SGLang hooks for the Qwen3-MoE true-on-policy contract.

SGLang loads this module through the ``sglang.srt.plugins`` entry point in every engine process. With
``MILES_MOE_QUANT`` set it also runs the experts with Miles' microscaling kernels. See
``examples/infra_features/true_on_policy/README.md`` for what each hook is for.
"""

import logging
import re

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

_QWEN3_MOE_BLOCK_INIT = "sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock.__init__"
_QWEN3_MOE_FORWARD_NORMAL = "sglang.srt.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward_normal"
_QWEN3_MOE_LOAD_WEIGHTS = "sglang.srt.models.qwen3_moe.Qwen3MoeForCausalLM.load_weights"
_UNQUANT_MOE_FORWARD_CUDA = "sglang.srt.layers.quantization.unquant.UnquantizedFusedMoEMethod.forward_cuda"
_EXPERT_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.")
_CACHE_ATTR = "_miles_mx_experts"

logger = logging.getLogger(__name__)


def _set_qwen3_moe_top_k(result, self, *args, **kwargs):
    # The block's true-on-policy routing branch (forward_normal) reads ``self.top_k``, which
    # SGLang only sets on the DeepEP path, so on the standard path it raises AttributeError.
    if not hasattr(self, "top_k"):  # config-access-exempt: SGLang sets top_k only on the DeepEP path
        config = kwargs["config"] if "config" in kwargs else args[1]
        self.top_k = config.num_experts_per_tok
    return result


def _qwen3_moe_forward_normal(original, self, hidden_states, *args, **kwargs):
    """SGLang's true-on-policy routing branch with the routed-experts capture that R3 needs.

    SGLang's branch skips the capture. This copy also renormalizes only when the model sets
    ``norm_topk_prob``, as the trainer does; SGLang's always renormalizes."""
    from sglang.srt.true_on_policy import is_true_on_policy_enabled

    if not is_true_on_policy_enabled() or hidden_states.shape[0] == 0:
        return original(self, hidden_states, *args, **kwargs)
    if args or kwargs:
        # an argument this copy does not handle; passing through would silently drop R3's routing
        raise TypeError(f"Qwen3MoeSparseMoeBlock.forward_normal got unexpected arguments {args} {list(kwargs)}")
    import torch.nn.functional as F
    from sglang.srt.layers.moe.topk import StandardTopKOutput, capture_routed_experts_if_allowed

    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits, _ = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.topk.topk_config.renormalize:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    capture_routed_experts_if_allowed(self.topk.topk_config, self.layer_id, selected_experts)
    topk_output = StandardTopKOutput(
        topk_weights=routing_weights, topk_ids=selected_experts, router_logits=router_logits
    )
    final_hidden_states = self.experts(hidden_states, topk_output)
    return final_hidden_states.view(num_tokens, hidden_dim)


def _refresh_mx_experts(model, layer_ids, fmt: str) -> None:
    """Re-quantize the experts of ``layer_ids`` into the cached buffers, in place, so captured CUDA
    graphs keep reading valid weights."""
    from miles.kernels.moe.mx_fused_moe import quantize_experts

    for layer_id in sorted(layer_ids):
        experts = model.model.layers[layer_id].mlp.experts
        cached = getattr(experts, _CACHE_ATTR, None)  # config-access-exempt: set below on first load
        w13 = quantize_experts(experts.w13_weight, fmt, out=None if cached is None else cached[0])
        w2 = quantize_experts(experts.w2_weight, fmt, out=None if cached is None else cached[1])
        setattr(experts, _CACHE_ATTR, (w13, w2))  # config-access-exempt: plugin-owned cache attribute


def _make_load_weights_hook(fmt: str):
    def _load_weights(original, self, weights, *args, **kwargs):
        touched = set()

        def recorded():
            for name, tensor in weights:
                if match := _EXPERT_LAYER.search(name):
                    touched.add(int(match.group(1)))
                yield name, tensor

        result = original(self, recorded(), *args, **kwargs)
        _refresh_mx_experts(self, touched, fmt)
        return result

    return _load_weights


def _mx_moe_forward_cuda(original, self, layer, dispatch_output, *args, **kwargs):
    cached = getattr(layer, _CACHE_ATTR, None)  # config-access-exempt: only Qwen3-MoE experts carry it
    if cached is None:
        return original(self, layer, dispatch_output, *args, **kwargs)
    from miles.kernels.moe.mx_fused_moe import mx_fused_experts
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    topk_output = dispatch_output.topk_output
    out = mx_fused_experts(
        dispatch_output.hidden_states, cached[0], cached[1], topk_output.topk_weights, topk_output.topk_ids
    )
    return StandardCombineInput(hidden_states=out)


def _register_moe_contract() -> None:
    """Add Miles' Qwen3-MoE contract to SGLang's contract table, which only ships the dense one.
    SGLang's runtime policy does not depend on the model, so the MoE contract gets the same switches;
    the MoE-specific parts are the Triton runner flag and the hooks below."""
    from dataclasses import fields

    from sglang.srt.true_on_policy import contracts
    from sglang.srt.true_on_policy.schema import TrueOnPolicyContractSchema

    from miles.true_on_policy.schema import QWEN3_MOE_TRUE_ON_POLICY_V1_SCHEMA as moe

    schema = TrueOnPolicyContractSchema(**{f.name: getattr(moe, f.name) for f in fields(TrueOnPolicyContractSchema)})
    contracts._CONTRACT_BY_NAME.setdefault(schema.name, contracts.SGLangTrueOnPolicyContract(schema=schema))


def register() -> None:
    _register_moe_contract()
    HookRegistry.register(_QWEN3_MOE_BLOCK_INIT, _set_qwen3_moe_top_k, HookType.AFTER)
    HookRegistry.register(_QWEN3_MOE_FORWARD_NORMAL, _qwen3_moe_forward_normal, HookType.AROUND)

    from miles.kernels.moe.mx_fused_moe import moe_mx_format

    if (fmt := moe_mx_format()) is not None:
        HookRegistry.register(_QWEN3_MOE_LOAD_WEIGHTS, _make_load_weights_hook(fmt), HookType.AROUND)
        HookRegistry.register(_UNQUANT_MOE_FORWARD_CUDA, _mx_moe_forward_cuda, HookType.AROUND)
        logger.warning(f"Miles plugin: Qwen3-MoE experts run the {fmt} microscaling kernels")
