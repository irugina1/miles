"""SGLang hooks for Miles' true-on-policy contracts.

SGLang loads this module through the ``sglang.srt.plugins`` entry point in every engine process. SGLang
only ships ``qwen3_dense_true_on_policy_v1`` (Qwen3 dense on Hopper); this adds Miles' Blackwell
contract, ``qwen3_dense_blackwell_true_on_policy_v1``, to its contract table. Without it the engines reject the contract name at startup.
"""


def _register_dense_blackwell_contract() -> None:
    """SGLang's runtime policy does not depend on the contract, so the Blackwell contract gets the same
    switches as the Hopper one; its attention backend (fa4) reaches SGLang as a server flag."""
    from dataclasses import fields

    from sglang.srt.true_on_policy import contracts
    from sglang.srt.true_on_policy.schema import TrueOnPolicyContractSchema

    from miles.true_on_policy.schema import QWEN3_DENSE_BLACKWELL_TRUE_ON_POLICY_V1_SCHEMA as blackwell

    schema = TrueOnPolicyContractSchema(**{f.name: getattr(blackwell, f.name) for f in fields(TrueOnPolicyContractSchema)})
    contracts._CONTRACT_BY_NAME.setdefault(schema.name, contracts.SGLangTrueOnPolicyContract(schema=schema))


def register() -> None:
    _register_dense_blackwell_contract()
