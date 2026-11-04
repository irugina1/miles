"""FSDP quantized rollout: which weights the sync quantizes, and where the quantization config comes from."""

from types import SimpleNamespace

import pytest

from miles.backends.fsdp_utils import quantization
from miles.backends.fsdp_utils.quantization import (
    is_quantized_module,
    rollout_model_path,
    rollout_quantization_config,
    validate_fp8_block_config,
)
from miles.backends.sglang_utils.sglang_config import WorkerType

FP8_EXPERTS = {
    "quant_method": "fp8",
    "fmt": "e4m3",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
    "modules_to_not_convert": [
        "model.embed_tokens",
        "lm_head",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.gate",
        "model.layers.0.input_layernorm",
    ],
}


def test_experts_quantized_and_listed_modules_kept():
    assert is_quantized_module(FP8_EXPERTS, "model.layers.0.mlp.experts.3.down_proj")
    assert not is_quantized_module(FP8_EXPERTS, "model.layers.0.self_attn.q_proj")
    assert not is_quantized_module(FP8_EXPERTS, "model.layers.0.mlp.gate")
    assert not is_quantized_module(FP8_EXPERTS, "lm_head")
    assert not is_quantized_module(FP8_EXPERTS, "model.embed_tokens")


def test_sync_refuses_to_quantize_anything_but_the_experts():
    import torch

    from miles.backends.fsdp_utils.quantization import quantize_hf_tensor

    # k_proj is missing from modules_to_not_convert, so the checkpoint would serve it in FP8
    with pytest.raises(ValueError, match="only routed MoE experts"):
        quantize_hf_tensor(FP8_EXPERTS, "model.layers.0.self_attn.k_proj.weight", torch.zeros(128, 128))
    kept = quantize_hf_tensor(FP8_EXPERTS, "model.layers.0.self_attn.q_proj.weight", torch.zeros(128, 128))
    assert [name for name, _ in kept] == ["model.layers.0.self_attn.q_proj.weight"]


def test_an_excluded_first_expert_keeps_the_whole_fused_moe_unquantized():
    config = dict(FP8_EXPERTS, modules_to_not_convert=["model.layers.1.mlp.experts.0.gate_proj"])
    assert not is_quantized_module(config, "model.layers.1.mlp.experts.7.up_proj")
    assert is_quantized_module(config, "model.layers.2.mlp.experts.7.up_proj")


@pytest.mark.parametrize(
    "config",
    [
        {"quant_method": "fp8", "activation_scheme": "dynamic"},  # per-tensor
        {"quant_method": "mxfp8", "weight_block_size": [1, 32]},
        dict(FP8_EXPERTS, scale_fmt="ue8m0"),
        dict(FP8_EXPERTS, activation_scheme="static"),
    ],
)
def test_unsupported_rollout_quantization_fails_early(config):
    with pytest.raises(NotImplementedError):
        validate_fp8_block_config(config)


def _args(model_path: str, *, update_weights: bool = True):
    group = SimpleNamespace(worker_type=WorkerType.REGULAR, model_path=model_path)
    model = SimpleNamespace(update_weights=update_weights, server_groups=[group])
    return SimpleNamespace(hf_checkpoint="/models/bf16", sglang=SimpleNamespace(models=[model]))


def test_rollout_model_path_follows_the_weight_synced_engines():
    assert rollout_model_path(_args("/models/fp8")) == "/models/fp8"
    # a frozen model (e.g. a reward model) never receives the trainer's weights
    assert rollout_model_path(_args("/models/other", update_weights=False)) == "/models/bf16"


def test_rollout_quantization_config_reads_the_served_checkpoint(monkeypatch):
    configs = {"/models/bf16": None, "/models/fp8": FP8_EXPERTS}
    monkeypatch.setattr(quantization, "_load_quantization_config", configs.__getitem__)
    assert rollout_quantization_config(_args("/models/bf16")) is None
    assert rollout_quantization_config(_args("/models/fp8")) == FP8_EXPERTS


def test_the_trainers_own_checkpoint_is_synced_as_trained(monkeypatch):
    # whatever quantization metadata the trainer's checkpoint carries, its weights go out unchanged
    monkeypatch.setattr(quantization, "_load_quantization_config", lambda path: {"quant_method": "mxfp4"})
    assert rollout_quantization_config(_args("/models/bf16")) is None
