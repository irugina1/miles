"""Block-FP8 rollout checkpoints for the FSDP backend.

The trainer loads the bf16 ``--hf-checkpoint``. When the rollout engines serve a different, block-FP8
checkpoint, its ``quantization_config`` decides which synced weights are quantized; only the routed
experts may be. The trainer and the weight sync quantize with the same kernel.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

from miles.kernels.quant.fp8_blockwise import fp8_blockwise_cast


@functools.cache
def _load_quantization_config(model_path: str) -> dict | None:
    from miles.utils.hf_utils.config import load_hf_config

    return getattr(
        load_hf_config(model_path), "quantization_config", None
    )  # config-access-exempt: unquantized HF checkpoints omit quantization_config


def rollout_model_path(args) -> str:
    """The checkpoint the weight-synced rollout engines serve (``--hf-checkpoint`` unless overridden)."""
    from miles.backends.sglang_utils.sglang_config import WorkerType

    paths = {
        group.model_path
        for model in args.sglang.models
        if model.update_weights
        for group in model.server_groups
        if group.worker_type != WorkerType.PLACEHOLDER
    }
    if not paths:
        return args.hf_checkpoint
    if len(paths) > 1:
        raise ValueError(f"Weight-synced rollout engines serve different checkpoints: {sorted(paths)}")
    return paths.pop()


def rollout_quantization_config(args) -> dict | None:
    """The block-FP8 config of a separate rollout checkpoint, or None when the engines serve the
    trainer's own checkpoint (its weights are then synced as trained)."""
    path = rollout_model_path(args)
    if path == args.hf_checkpoint:
        return None
    config = _load_quantization_config(path)
    if config is None:
        return None
    config = dict(config)
    validate_fp8_block_config(config)
    return config


def validate_fp8_block_config(config: dict) -> None:
    if config.get("quant_method") != "fp8" or not config.get("weight_block_size"):
        raise NotImplementedError(
            f"The FSDP backend syncs only block-wise FP8 rollout checkpoints, got quantization_config={config}"
        )
    if config.get("fmt", "e4m3") != "e4m3" or config.get("activation_scheme", "dynamic") != "dynamic":
        raise NotImplementedError(f"Only e4m3 weights with dynamic activation scales are supported: {config}")
    if config.get("scale_fmt") not in (None, "fp32"):
        raise NotImplementedError(f"Only fp32 block scales are supported: {config}")


def _ignored_modules(config: dict) -> frozenset[str]:
    names = config.get("modules_to_not_convert") or config.get("ignored_layers") or ()
    return frozenset(
        name for raw in names for name in (raw.removeprefix("model."), f"model.{raw.removeprefix('model.')}")
    )


def is_quantized_module(config: dict, module_name: str) -> bool:
    """Whether the rollout checkpoint stores ``module_name``'s weight in FP8.

    SGLang picks one precision for a whole FusedMoE module, so every expert of a layer follows the
    first expert's entry (checkpoint converters list per-expert names).
    """
    ignored = _ignored_modules(config)
    if module_name in ignored:
        return False
    if ".experts." in module_name:
        experts_prefix = module_name.split(".experts.", 1)[0] + ".experts"
        return experts_prefix not in ignored and f"{experts_prefix}.0.gate_proj" not in ignored
    return True


def fp8_block_quantize(weight: torch.Tensor, block_size) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP8 of a 2-D bf16 weight: (e4m3 weight, fp32 per-block scale), as the sync sends it."""
    return fp8_blockwise_cast(weight.contiguous(), list(block_size))


def quantize_hf_tensor(config: dict, name: str, tensor: torch.Tensor) -> list[tuple[str, Any]]:
    """The HF tensors the rollout engine expects in place of ``name``: the tensor itself, or an FP8
    weight plus its ``weight_scale_inv``."""
    if not name.endswith(".weight") or tensor.ndim != 2 or not is_quantized_module(config, name[: -len(".weight")]):
        return [(name, tensor)]
    if ".experts." not in name:
        # The trainer mirrors FP8 numerics for the routed experts only; anything else would run in
        # FP8 on the rollout side and in bf16 here.
        raise ValueError(
            f"the rollout checkpoint quantizes {name}, but only routed MoE experts may be quantized; "
            "list every other module in its modules_to_not_convert"
        )
    qweight, scale = fp8_block_quantize(tensor, config["weight_block_size"])
    return [(name, qweight), (name.replace(".weight", ".weight_scale_inv"), scale)]
