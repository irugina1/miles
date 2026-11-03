from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TrueOnPolicyContractName = Literal["qwen3_dense_true_on_policy_v1", "qwen3_moe_true_on_policy_v1"]
ModelFamily = Literal["qwen3_dense", "qwen3_moe", "qwen3_next"]
KernelContract = Literal["qwen3_dense_sglang_math", "qwen3_moe_sglang_math"]
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
    # SGLang MoE runner whose kernels the trainer mirrors; None for dense models.
    sglang_moe_runner_backend: str | None = None


QWEN3_DENSE_TRUE_ON_POLICY_V1_SCHEMA = TrueOnPolicyContractSchema(
    name="qwen3_dense_true_on_policy_v1",
    model_family="qwen3_dense",
    required_kernel_contracts=("qwen3_dense_sglang_math",),
    logprob_contract="sglang_prefill",
    sglang_attention_backend="fa3",
    fsdp_attention_implementation="flash_attention_3",
    disable_megatron_sequence_parallel=True,
)


# Qwen3 MoE: the trainer is bitwise equal to SGLang's decode (FA4 attention and Triton fused-MoE on
# both sides). SGLang does not ship this contract; Miles' SGLang plugin registers it there with the
# dense contract's runtime policy, which is model-agnostic. See
# examples/infra_features/true_on_policy/README.md.
QWEN3_MOE_TRUE_ON_POLICY_V1_SCHEMA = TrueOnPolicyContractSchema(
    name="qwen3_moe_true_on_policy_v1",
    model_family="qwen3_moe",
    required_kernel_contracts=("qwen3_moe_sglang_math",),
    logprob_contract="sglang_decode",
    sglang_attention_backend="fa4",
    fsdp_attention_implementation="fa4",
    disable_megatron_sequence_parallel=True,
    sglang_moe_runner_backend="triton",
)
