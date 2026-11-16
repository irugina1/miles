from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TrueOnPolicyContractName = Literal["qwen3_dense_true_on_policy_v1", "qwen3_dense_blackwell_true_on_policy_v1"]
ModelFamily = Literal["qwen3_dense", "qwen3_moe", "qwen3_next"]
KernelContract = Literal["qwen3_dense_sglang_math"]
# Where the rollout logprobs the trainer must match come from: an SGLang prefill that re-scores each
# finished rollout, or the decode pass that sampled it (the sampler itself).
LogprobContract = Literal["sglang_prefill", "sglang_decode"]


@dataclass(frozen=True)
class TrueOnPolicyContractSchema:
    """Declarative cross-repo identity for a true-on-policy parity contract."""

    name: TrueOnPolicyContractName
    model_family: ModelFamily
    required_kernel_contracts: tuple[KernelContract, ...]
    logprob_contract: LogprobContract
    sglang_attention_backend: str
    fsdp_attention_implementation: str
    disable_megatron_sequence_parallel: bool


# A true-on-policy contract fixes the kernels on both sides, so it holds for one (model, hardware) pair.
# This one is Qwen3 dense on Hopper: FlashAttention-3 runs on SM80-90 only (SGLang refuses it on
# Blackwell). The trainer matches an SGLang prefill that re-scores each finished rollout.
QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA = TrueOnPolicyContractSchema(
    name="qwen3_dense_true_on_policy_v1",
    model_family="qwen3_dense",
    required_kernel_contracts=("qwen3_dense_sglang_math",),
    logprob_contract="sglang_prefill",
    sglang_attention_backend="fa3",
    fsdp_attention_implementation="flash_attention_3",
    disable_megatron_sequence_parallel=True,
)


# Qwen3 dense on Blackwell (SM100), where the contract above cannot run. Attention is SGLang's vendored
# FlashAttention-4 on both sides (batch-invariant, one KV split), and the trainer calls it the way
# SGLang's decode does, so the FSDP trainer is bitwise equal to the sampler's decode: the sampler's own
# logprobs are used and nothing is re-scored. Everything else follows the contract above. FSDP with
# single-GPU rollout engines only; verified on Qwen3-4B-Instruct-2507. SGLang does not ship this
# contract; Miles' SGLang plugin registers it there with the dense contract's runtime policy.
QWEN3_DENSE_BLACKWELL_TRUE_ON_POLICY_V1_SCHEMA = TrueOnPolicyContractSchema(
    name="qwen3_dense_blackwell_true_on_policy_v1",
    model_family="qwen3_dense",
    required_kernel_contracts=("qwen3_dense_sglang_math",),
    logprob_contract="sglang_decode",
    sglang_attention_backend="fa4",
    fsdp_attention_implementation="fa4",
    disable_megatron_sequence_parallel=True,
)
