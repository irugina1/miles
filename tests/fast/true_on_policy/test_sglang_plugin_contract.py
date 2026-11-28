from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


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
        ("model.layers.3.self_attn.qkv_proj", "nvfp4"),
        ("model.layers.3.self_attn.o_proj", "nvfp4"),
        ("model.layers.3.mlp.gate_up_proj", "nvfp4"),
        ("model.layers.35.mlp.down_proj", "nvfp4"),
        ("model.layers.3.self_attn.q_norm", None),
        ("lm_head", None),
        ("model.embed_tokens", None),
        ("", None),
    ],
)
def test_plugin_quantizes_every_sglang_qwen3_projection_and_nothing_else(prefix, expected, monkeypatch):
    pytest.importorskip("sglang")
    import miles_plugins.sglang.true_on_policy as plugin
    from miles.true_on_policy.contracts import QWEN3_DENSE_BLACKWELL_NVFP4_TRUE_ON_POLICY_V1

    monkeypatch.setattr(plugin, "_quantized_contract", lambda: QWEN3_DENSE_BLACKWELL_NVFP4_TRUE_ON_POLICY_V1)
    assert plugin._projection_format(prefix, 36) == expected


def test_mxfp8_tail_contract_keeps_the_last_sixth_of_the_layers_in_mxfp8():
    from miles.true_on_policy.contracts import QWEN3_DENSE_BLACKWELL_NVFP4_TRUE_ON_POLICY_V1, get_true_on_policy_contract

    contract = get_true_on_policy_contract("qwen3_dense_blackwell_nvfp4_mxfp8_tail_true_on_policy_v1")
    assert [contract.linear_quant_of_layer(i, 36) for i in range(36)] == ["nvfp4"] * 30 + ["mxfp8"] * 6
    assert replace(contract.schema, name="x", linear_quant_tail=None, linear_quant_tail_fraction=0.0) == replace(
        QWEN3_DENSE_BLACKWELL_NVFP4_TRUE_ON_POLICY_V1.schema, name="x"
    )
    assert contract.describe_linear_quant(36) == "nvfp4: layers 0-29, mxfp8: layers 30-35"
    assert contract.describe_linear_quant(28) == "nvfp4: layers 0-22, mxfp8: layers 23-27"  # 28 / 6 = 4.67 -> 5
    assert QWEN3_DENSE_BLACKWELL_NVFP4_TRUE_ON_POLICY_V1.describe_linear_quant(36) == "nvfp4: layers 0-35"


def test_a_tail_needs_a_base_format_and_a_fraction():
    from miles.true_on_policy.schema import QWEN3_DENSE_BLACKWELL_BF16_TRUE_ON_POLICY_V1_SCHEMA as bf16

    with pytest.raises(ValueError):
        replace(bf16, linear_quant_tail="mxfp8", linear_quant_tail_fraction=1 / 6)
    with pytest.raises(ValueError):
        replace(bf16, linear_quant="nvfp4", linear_quant_tail="mxfp8")
    with pytest.raises(ValueError):
        replace(bf16, linear_quant="nvfp4", linear_quant_tail_fraction=0.5)


def test_plugin_tags_each_projection_with_its_layer_format(monkeypatch):
    pytest.importorskip("sglang")
    import miles_plugins.sglang.true_on_policy as plugin
    from miles.true_on_policy.contracts import get_true_on_policy_contract

    contract = get_true_on_policy_contract("qwen3_dense_blackwell_nvfp4_mxfp8_tail_true_on_policy_v1")
    monkeypatch.setattr(plugin, "_quantized_contract", lambda: contract)
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_hidden_layers=36)
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    for _ in range(36):
        layer = torch.nn.Module()
        layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
        layer.self_attn.qkv_proj, layer.mlp.down_proj, layer.input_layernorm = (torch.nn.Module() for _ in range(3))
        model.model.layers.append(layer)
    plugin._tag_module_names(None, model)
    tag = lambda m: getattr(m, plugin._FORMAT_ATTR)  # noqa: E731
    layers = model.model.layers
    assert [tag(layer.self_attn.qkv_proj) for layer in layers] == ["nvfp4"] * 30 + ["mxfp8"] * 6
    assert [tag(layer.mlp.down_proj) for layer in layers] == ["nvfp4"] * 30 + ["mxfp8"] * 6
    assert {tag(layer.input_layernorm) for layer in layers} == {None}
    monkeypatch.setattr(plugin, "_quantized_contract", lambda: None)
    plugin._tag_module_names(None, model)
    assert {tag(layer.self_attn.qkv_proj) for layer in layers} == {None}


def test_trainer_swaps_each_layer_to_its_contract_format():
    pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from miles.backends.fsdp_utils.models.qwen3 import QUANTIZED_PROJECTIONS, apply_qwen3_dense_linear_quant
    from miles.true_on_policy.contracts import get_true_on_policy_contract

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=12,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    model = Qwen3ForCausalLM(config)
    contract = get_true_on_policy_contract("qwen3_dense_blackwell_nvfp4_mxfp8_tail_true_on_policy_v1")
    assert apply_qwen3_dense_linear_quant(model, contract) == 12 * len(QUANTIZED_PROJECTIONS)

    def layer_formats(layer):
        blocks = (layer.self_attn, layer.mlp)
        return {getattr(b, name).mx_fmt for b in blocks for name in QUANTIZED_PROJECTIONS if hasattr(b, name)}

    assert [layer_formats(layer) for layer in model.model.layers] == [{"nvfp4"}] * 10 + [{"mxfp8"}] * 2
