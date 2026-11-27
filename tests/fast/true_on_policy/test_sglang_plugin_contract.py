from dataclasses import replace
from types import SimpleNamespace

import pytest


def test_plugin_registers_every_miles_contract_with_sglang_runtime_policy():
    pytest.importorskip("sglang")
    from sglang.srt.true_on_policy import contracts

    from miles.true_on_policy.contracts import all_true_on_policy_contracts
    from miles_plugins.sglang.true_on_policy import _register_contracts

    _register_contracts()
    server_args = SimpleNamespace(tp_size=1)
    sglang_own = contracts.get_true_on_policy_contract("qwen3_dense_true_on_policy_v1").policy_for(server_args)
    for contract in all_true_on_policy_contracts():
        registered = contracts.get_true_on_policy_contract(contract.name).policy_for(server_args)
        assert registered == replace(sglang_own, contract_name=contract.name)


@pytest.mark.parametrize("fmt", ["mxfp8", "mxfp4", "nvfp4"])
def test_quantized_contracts_are_the_blackwell_contract_with_a_format(fmt):
    from miles.true_on_policy.contracts import QWEN3_DENSE_BLACKWELL_BF16_TRUE_ON_POLICY_V1, get_true_on_policy_contract

    quantized = get_true_on_policy_contract(f"qwen3_dense_blackwell_{fmt}_true_on_policy_v1")
    assert quantized.linear_quant == fmt
    assert QWEN3_DENSE_BLACKWELL_BF16_TRUE_ON_POLICY_V1.linear_quant is None
    assert replace(quantized.schema, name=QWEN3_DENSE_BLACKWELL_BF16_TRUE_ON_POLICY_V1.name, linear_quant=None) == (
        QWEN3_DENSE_BLACKWELL_BF16_TRUE_ON_POLICY_V1.schema
    )


@pytest.mark.parametrize(
    "prefix,expected",
    [
        ("model.layers.3.self_attn.qkv_proj", True),
        ("model.layers.3.self_attn.o_proj", True),
        ("model.layers.3.mlp.gate_up_proj", True),
        ("model.layers.35.mlp.down_proj", True),
        ("model.layers.3.self_attn.q_norm", False),
        ("lm_head", False),
        ("model.embed_tokens", False),
        ("", False),
    ],
)
def test_plugin_quantizes_every_sglang_qwen3_projection_and_nothing_else(prefix, expected):
    pytest.importorskip("sglang")
    from miles_plugins.sglang.true_on_policy import _is_quantized_projection

    assert _is_quantized_projection(prefix) is expected
