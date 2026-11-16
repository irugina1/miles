from dataclasses import replace
from types import SimpleNamespace

import pytest


def test_plugin_registers_the_dense_blackwell_contract_with_sglang_runtime_policy():
    pytest.importorskip("sglang")
    from sglang.srt.true_on_policy import contracts

    from miles_plugins.sglang.true_on_policy import _register_dense_blackwell_contract

    _register_dense_blackwell_contract()
    server_args = SimpleNamespace(tp_size=1)
    blackwell = contracts.get_true_on_policy_contract("qwen3_dense_blackwell_true_on_policy_v1").policy_for(server_args)
    hopper = contracts.get_true_on_policy_contract("qwen3_dense_true_on_policy_v1").policy_for(server_args)

    assert blackwell == replace(hopper, contract_name="qwen3_dense_blackwell_true_on_policy_v1")
