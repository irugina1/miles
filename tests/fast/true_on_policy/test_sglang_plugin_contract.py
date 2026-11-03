from dataclasses import replace
from types import SimpleNamespace

import pytest


def test_plugin_registers_the_moe_contract_with_sglang_runtime_policy():
    pytest.importorskip("sglang")
    from sglang.srt.true_on_policy import contracts

    from miles_plugins.sglang.true_on_policy import _register_moe_contract

    _register_moe_contract()
    server_args = SimpleNamespace(tp_size=1)
    moe = contracts.get_true_on_policy_contract("qwen3_moe_true_on_policy_v1").policy_for(server_args)
    dense = contracts.get_true_on_policy_contract("qwen3_dense_true_on_policy_v1").policy_for(server_args)

    assert moe == replace(dense, contract_name="qwen3_moe_true_on_policy_v1")
