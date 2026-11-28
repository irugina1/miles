"""SGLang hooks for Miles' true-on-policy contracts.

SGLang loads this module through the ``sglang.srt.plugins`` entry point in every engine process. SGLang
ships only ``qwen3_dense_true_on_policy_v1`` (Qwen3 dense on Hopper); this adds every Miles contract to
its contract table under Miles' name (the Hopper one as ``qwen3_dense_hopper_bf16_true_on_policy_v1``).
Without it the engines reject the contract name at startup.

Under a quantized contract (``linear_quant`` set, e.g. ``qwen3_dense_blackwell_nvfp4_true_on_policy_v1``)
it also runs the Qwen3 dense projections with Miles' quantized linear kernels
(miles/kernels/quant/mx_linear.py), the ones the trainer runs. The engines keep the bf16 weights that
weight sync writes; after every load the quantized copy is refreshed in place, so captured CUDA graphs
keep reading valid weights.
"""

import functools
import logging
import re

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

_UNQUANT_LINEAR = "sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod"
_QWEN3_INIT = "sglang.srt.models.qwen3.Qwen3ForCausalLM.__init__"
# SGLang's (possibly merged) Qwen3 projections; every one is quantized under a quantized contract
_PROJECTION = re.compile(r"(?:^|\.)layers\.\d+\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.(?:gate_up_proj|down_proj))$")
_CACHE_ATTR = "_miles_mx_weight"
# SGLang sets ``prefix`` on some linear classes only, so the model tags each module with its name
_NAME_ATTR = "_miles_module_name"

logger = logging.getLogger(__name__)


def _register_contracts() -> None:
    """SGLang's runtime policy does not depend on the contract, so each Miles contract gets the same
    switches as SGLang's own; the attention backend reaches SGLang as a server flag."""
    from dataclasses import fields

    from sglang.srt.true_on_policy import contracts
    from sglang.srt.true_on_policy.schema import TrueOnPolicyContractSchema

    from miles.true_on_policy.contracts import all_true_on_policy_contracts

    for contract in all_true_on_policy_contracts():
        schema = TrueOnPolicyContractSchema(
            **{f.name: getattr(contract.schema, f.name) for f in fields(TrueOnPolicyContractSchema)}
        )
        contracts._CONTRACT_BY_NAME.setdefault(schema.name, contracts.SGLangTrueOnPolicyContract(schema=schema))


@functools.cache
def _linear_quant() -> str | None:
    """The engine's quantization format, from its true-on-policy contract (None: bf16 linears)."""
    from sglang.srt.runtime_context import get_server_args

    from miles.true_on_policy.contracts import all_true_on_policy_contracts

    name = getattr(get_server_args(), "true_on_policy_contract", None)
    contract = {c.name: c for c in all_true_on_policy_contracts()}.get(name)
    fmt = contract.linear_quant if contract is not None else None
    if fmt is not None:
        logger.warning(f"Miles plugin: Qwen3 dense projections run the {fmt} W+A kernels ({name})")
    return fmt


def _is_quantized_projection(name: str) -> bool:
    return _PROJECTION.search(name) is not None


def _tag_module_names(result, self, *args, **kwargs):
    for name, module in self.named_modules():
        setattr(module, _NAME_ATTR, name)  # config-access-exempt: plugin-owned tag
    return result


def _refresh_quantized_weight(result, self, layer, *args, **kwargs):
    """Re-quantize the layer's bf16 weight after a load, into the cached buffers when they exist."""
    name = getattr(layer, _NAME_ATTR, None)  # config-access-exempt: set on Qwen3 models only
    if name is None or not _is_quantized_projection(name) or (fmt := _linear_quant()) is None:
        return result
    from miles.kernels.quant.mx_linear import quantize_weight

    if getattr(layer, "bias", None) is not None:  # config-access-exempt: Qwen3 projections have no bias
        raise ValueError(f"{name}: quantized linears support bias-free layers only")
    weight = layer.weight
    shard_sizes = tuple(getattr(layer, "output_sizes", None) or (weight.shape[0],))  # config-access-exempt: only merged linears have output_sizes
    if sum(shard_sizes) != weight.shape[0]:
        raise ValueError(f"{name}: shards {shard_sizes} do not cover {tuple(weight.shape)} (TP > 1?)")
    cached = getattr(layer, _CACHE_ATTR, None)  # config-access-exempt: plugin-owned cache, set below
    setattr(layer, _CACHE_ATTR, quantize_weight(weight.data, fmt, shard_sizes, out=cached))  # config-access-exempt: plugin-owned cache
    return result


def _mx_linear_apply(original, self, layer, x, bias=None, *args, **kwargs):
    cached = getattr(layer, _CACHE_ATTR, None)  # config-access-exempt: only quantized projections carry it
    if cached is None:
        return original(self, layer, x, bias, *args, **kwargs)
    from miles.kernels.quant.mx_linear import mx_linear_inference

    assert bias is None and not args and not kwargs, (bias, args, kwargs)
    return mx_linear_inference(x, cached, _linear_quant())


def register() -> None:
    _register_contracts()
    # the contract is known only once SGLang has parsed its arguments, so the hooks always go in and act
    # under a quantized contract only (a bf16 engine never caches a quantized weight, so its linears keep
    # SGLang's own path)
    HookRegistry.register(_QWEN3_INIT, _tag_module_names, HookType.AFTER)
    HookRegistry.register(f"{_UNQUANT_LINEAR}.process_weights_after_loading", _refresh_quantized_weight, HookType.AFTER)
    HookRegistry.register(f"{_UNQUANT_LINEAR}.apply", _mx_linear_apply, HookType.AROUND)
