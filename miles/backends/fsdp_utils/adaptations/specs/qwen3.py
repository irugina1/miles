"""Qwen3 dense: the formal true-on-policy precision contract (fp32 params under bf16 autocast, selected
params synced at fp32), its final-norm rounding order, and the quantized linears of the quantized
contracts."""

import logging
from dataclasses import replace

import torch

from miles.backends.fsdp_utils.adaptations.arch_adapter import ArchAdapter
from miles.true_on_policy.contracts import all_true_on_policy_contracts, get_true_on_policy_contract

logger = logging.getLogger(__name__)

# the Qwen3 dense contracts share the precision contract and model math below; they differ in attention
# (Hopper / Blackwell) and in the linear layers (bf16 / quantized)
_DENSE_CONTRACTS = frozenset(c.name for c in all_true_on_policy_contracts() if c.model_family == "qwen3_dense")


def _uses_formal_contract(args) -> bool:
    return (
        args.true_on_policy_mode
        and args.sglang.common_value("true_on_policy_contract") in _DENSE_CONTRACTS
    )


def _resolve_sync_dtype(name, checkpoint_dtype):
    from miles.backends.fsdp_utils.models.qwen3 import resolve_qwen3_dense_sync_dtype

    return resolve_qwen3_dense_sync_dtype(name, checkpoint_dtype)


class Qwen3Adapter(ArchAdapter):
    model_types = frozenset({"qwen3"})
    verified = True
    passes_attention_packing_kwargs = True

    def resolve_precision(self, base, args):
        if not _uses_formal_contract(args):
            return base
        contract = args.sglang.common_value("true_on_policy_contract")
        if args.backend.fp16:
            raise ValueError(f"{contract} requires bf16 training")
        if not base.keep_fp32_master:
            raise ValueError(f"{contract} requires fp32 master weights")
        return replace(
            base, param_dtype=torch.float32, autocast_dtype=torch.bfloat16, sync_dtype_resolver=_resolve_sync_dtype
        )

    def patch_model(self, model, args):
        if not _uses_formal_contract(args) or args.backend.fp16:
            return
        from miles.backends.fsdp_utils.models.qwen3 import (
            apply_qwen3_dense_linear_quant,
            apply_qwen3_dense_true_on_policy_patch,
        )

        apply_qwen3_dense_true_on_policy_patch(model)
        contract = get_true_on_policy_contract(args.sglang.common_value("true_on_policy_contract"))
        if contract.linear_quant is not None:
            n = apply_qwen3_dense_linear_quant(model, contract.linear_quant)
            logger.warning(f"Qwen3 dense: {n} projections run the {contract.linear_quant} W+A kernels ({contract.name})")
