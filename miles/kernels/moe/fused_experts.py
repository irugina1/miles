from __future__ import annotations

from dataclasses import dataclass

import torch
import triton.language as tl
from sgl_kernel import moe_sum_reduce

from miles.kernels.moe import mx_fused_moe as mx
from miles.kernels.moe.fused_experts_bwd import invoke_fused_moe_backward_kernel
from miles.kernels.moe.mx_fused_moe import MXTensor
from miles.kernels.quant.fp8_blockwise import fp8_blockwise_cast
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.fused_moe_triton_kernels import invoke_fused_moe_kernel
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size


@dataclass(frozen=True)
class FP8Weight:
    """Block-FP8 expert weights for the forward: e4m3 [E, N, K] and fp32 scales [E, N/bn, K/bk]."""

    weight: torch.Tensor
    scale: torch.Tensor
    block_shape: list[int]


def quantize_experts_fp8(weight: torch.Tensor, block_shape: list[int]) -> FP8Weight:
    """[E, N, K] bf16 -> block FP8. Blocks never straddle experts (N is a multiple of the block),
    so this equals quantizing each expert's matrices one by one, as the weight sync does."""
    num_experts, n, k = weight.shape
    bn, bk = block_shape
    assert n % bn == 0 and k % bk == 0, (weight.shape, block_shape)
    with torch.no_grad():
        qweight, scale = fp8_blockwise_cast(weight.detach().reshape(num_experts * n, k).contiguous(), [bn, bk])
    return FP8Weight(
        weight=qweight.view(num_experts, n, k), scale=scale.view(num_experts, n // bn, k // bk), block_shape=[bn, bk]
    )


def _quant_spec(fp8: FP8Weight | None, mx_weight: MXTensor | None):
    """What the backward needs to rebuild the forward's quantized operands (never the quantized
    weights themselves: keeping them for every layer until backward would cost a bf16-sized copy)."""
    if mx_weight is not None:
        return ("mx", mx_weight.fmt)
    if fp8 is not None:
        return ("fp8", list(fp8.block_shape))
    return None


def _dequantized_backward_operands(spec, activations: torch.Tensor, weight: torch.Tensor):
    """dequant(Q(X)) and dequant(Q(W)): the operands the forward multiplied, rebuilt from the saved
    bf16 tensors (the quantizers are deterministic) for the bf16 backward GEMMs."""
    if spec is None:
        return activations, weight
    kind, arg = spec
    with torch.no_grad():
        if kind == "mx":
            x = mx.dequantize(mx.quantize_rows(activations, arg))
            w = mx.dequantize(mx.quantize_experts(weight, arg))
            return x, w
        bn, bk = arg
        # activations: per-token groups of bk, quantized exactly as SGLang's FP8 MoE launcher does
        xq, xs = sglang_per_token_group_quant_fp8(activations.contiguous(), bk)
        x = (xq.float().view(xq.shape[0], -1, bk) * xs.float()[..., None]).view(xq.shape).to(torch.bfloat16)
        # weights: the forward's own cast
        num_experts, n, k = weight.shape
        wf = quantize_experts_fp8(weight, [bn, bk])
        wq = wf.weight.view(num_experts, n // bn, bn, k // bk, bk)
        ws = wf.scale.view(num_experts, n // bn, 1, k // bk, 1)
        w = torch.empty(num_experts, n, k, device=weight.device, dtype=torch.bfloat16)
        for e0 in range(0, num_experts, 16):  # chunks of experts bound the fp32 temporaries
            block = wq[e0 : e0 + 16].float() * ws[e0 : e0 + 16].float()
            w[e0 : e0 + 16] = block.view(-1, n, k).to(torch.bfloat16)
        return x, w


class GateUpProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w1_fp8: FP8Weight | None = None,
        w1_mx: MXTensor | None = None,
    ):
        num_tokens, _ = hidden_states.shape
        E, N, _ = w1.shape
        CHUNK_SIZE = 64 * 1024

        # default deterministic config
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }

        topk = topk_ids.shape[1]

        if w1_mx is not None:
            # microscaling W+A: the kernels the rollout engine runs (see mx_fused_moe.py)
            intermediate_cache1 = mx.gate_up(hidden_states, w1_mx, topk_weights, topk_ids, mx.align(topk_ids, E))
        else:
            intermediate_cache1 = torch.empty(
                (num_tokens * topk, N),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            for chunk in range((num_tokens // CHUNK_SIZE) + 1):
                begin_chunk_idx, end_chunk_idx = (
                    chunk * CHUNK_SIZE,
                    min((chunk + 1) * CHUNK_SIZE, num_tokens),
                )
                curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
                cur_intermediate_cache1 = intermediate_cache1[begin_chunk_idx * topk : end_chunk_idx * topk]

                curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
                curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

                sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                    curr_topk_ids, config["BLOCK_SIZE_M"], E
                )

                invoke_fused_moe_kernel(
                    curr_hidden_states,
                    w1 if w1_fp8 is None else w1_fp8.weight,
                    None,
                    cur_intermediate_cache1,
                    None,
                    None if w1_fp8 is None else w1_fp8.scale,
                    None,
                    curr_topk_weights,
                    curr_topk_ids,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    False,
                    topk_ids.shape[1],
                    config,
                    compute_type=tl.bfloat16,
                    use_fp8_w8a8=w1_fp8 is not None,
                    use_int8_w8a8=False,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    per_channel_quant=False,
                    block_shape=None if w1_fp8 is None else w1_fp8.block_shape,
                    # SGLang enables TMA only for tuned configs that ask for it; the deterministic config does not
                    a_use_tma=False,
                    b_use_tma=False,
                    c_sorted=False,
                    # SGLang's runner passes False when every expert is local (no EP)
                    filter_expert=False,
                )

        ctx.save_for_backward(hidden_states, w1, topk_weights, topk_ids)
        ctx.quant = _quant_spec(w1_fp8, w1_mx)
        ctx.config = config
        ctx.num_tokens = num_tokens
        ctx.topk = topk

        return intermediate_cache1

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for GateUpProjFunction using Triton kernels.

        Args:
            grad_output: shape (num_tokens * topk, N)

        Returns:
            (grad_hidden_states, grad_w1, grad_topk_weights, None)
        """

        hidden_states, w1, topk_weights, topk_ids = ctx.saved_tensors
        config = ctx.config
        num_tokens = ctx.num_tokens
        topk = ctx.topk
        bwd_hidden_states, bwd_w1 = _dequantized_backward_operands(ctx.quant, hidden_states, w1)

        E, N, D_in = w1.shape
        CHUNK_SIZE = 64 * 1024

        # Initialize gradient tensors
        grad_hidden_states = torch.zeros_like(hidden_states)
        grad_w1 = torch.zeros_like(w1)
        # GateUpProj stage doesn't need topk_weights gradient
        grad_topk_weights = torch.zeros_like(topk_weights)

        # Process in chunks to match forward pass
        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )

            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            curr_hidden_states = bwd_hidden_states[begin_chunk_idx:end_chunk_idx]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx * topk : end_chunk_idx * topk]

            # Get aligned metadata
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            # Prepare gradient buffer for this chunk
            curr_grad_hidden_states = torch.zeros_like(curr_hidden_states)
            curr_grad_w1 = torch.zeros_like(w1)

            # Call Triton backward kernel with MUL_ROUTED_WEIGHT=False
            # Use chunk of hidden_states to match sorted_token_ids indices
            invoke_fused_moe_backward_kernel(
                grad_output=curr_grad_output,
                input=curr_hidden_states,  # Use chunk of hidden_states to match sorted_token_ids
                weight=bwd_w1,
                grad_input=curr_grad_hidden_states,
                grad_weight=curr_grad_w1,
                grad_topk_weights=None,  # Not needed for GateUpProj
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False,
                top_k=topk,
                config=config,
                compute_type=tl.bfloat16,
            )

            # Accumulate gradients
            grad_hidden_states[begin_chunk_idx:end_chunk_idx] += curr_grad_hidden_states
            grad_w1 += curr_grad_w1

        return grad_hidden_states, grad_w1, grad_topk_weights, None, None, None


class SiluAndMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, intermediate_cache1: torch.Tensor):
        num_tokens, N = intermediate_cache1.shape
        intermediate_cache2 = torch.empty(
            (num_tokens, N // 2),
            device=intermediate_cache1.device,
            dtype=intermediate_cache1.dtype,
        )
        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)

        ctx.save_for_backward(intermediate_cache1)
        return intermediate_cache2

    @staticmethod
    def backward(ctx, grad_output):
        (intermediate_cache1,) = ctx.saved_tensors
        N = intermediate_cache1.shape[-1]
        x1, x2 = intermediate_cache1.view(-1, N).chunk(2, dim=-1)
        silu_x1 = torch.nn.functional.silu(x1)

        sig = torch.sigmoid(x1)
        dsilu_dx1 = sig + x1 * sig * (1 - sig)
        grad_x1 = grad_output * x2 * dsilu_dx1
        grad_x2 = grad_output * silu_x1
        grad_input = torch.cat([grad_x1, grad_x2], dim=-1)

        return grad_input.view_as(intermediate_cache1)


class DownProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        intermediate_cache2: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        w2_fp8: FP8Weight | None = None,
        w2_mx: MXTensor | None = None,
    ):
        num_tokens, _ = intermediate_cache2.shape
        topk = topk_ids.shape[1]
        num_tokens //= topk
        E, _, _ = w2.shape
        # We execute the fused_moe kernel in chunks to circumvent this issue:
        # https://github.com/vllm-project/vllm/issues/5938
        CHUNK_SIZE = 64 * 1024

        # default deterministic config
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }

        if w2_mx is not None:
            intermediate_cache3 = mx.down(intermediate_cache2, w2_mx, topk_weights, topk_ids, mx.align(topk_ids, E))
        else:
            intermediate_cache3 = torch.empty(
                (num_tokens, topk, w2.shape[1]),
                device=intermediate_cache2.device,
                dtype=intermediate_cache2.dtype,
            )
            for chunk in range((num_tokens // CHUNK_SIZE) + 1):
                begin_chunk_idx, end_chunk_idx = (
                    chunk * CHUNK_SIZE,
                    min((chunk + 1) * CHUNK_SIZE, num_tokens),
                )
                cur_intermediate_cache2 = intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk]
                cur_intermediate_cache3 = intermediate_cache3[begin_chunk_idx:end_chunk_idx]

                curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
                curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

                sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                    curr_topk_ids, config["BLOCK_SIZE_M"], E
                )
                invoke_fused_moe_kernel(
                    cur_intermediate_cache2,
                    w2 if w2_fp8 is None else w2_fp8.weight,
                    None,
                    cur_intermediate_cache3,
                    None,
                    None if w2_fp8 is None else w2_fp8.scale,
                    None,
                    curr_topk_weights,
                    curr_topk_ids,
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                    True,
                    1,
                    config,
                    compute_type=tl.bfloat16,
                    use_fp8_w8a8=w2_fp8 is not None,
                    use_int8_w8a8=False,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    per_channel_quant=False,
                    block_shape=None if w2_fp8 is None else w2_fp8.block_shape,
                    a_use_tma=False,
                    b_use_tma=False,
                    filter_expert=False,
                    router_topk=topk,
                )

        ctx.save_for_backward(intermediate_cache2, w2, topk_weights, topk_ids)
        ctx.quant = _quant_spec(w2_fp8, w2_mx)
        ctx.config = config
        ctx.num_tokens = num_tokens
        ctx.topk = topk

        return intermediate_cache3

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for DownProjFunction using Triton kernels.

        Args:
            grad_output: shape (num_tokens, topk, hidden_size)

        Returns:
            (grad_intermediate_cache2, grad_w2, grad_topk_weights, None)
        """
        intermediate_cache2, w2, topk_weights, topk_ids = ctx.saved_tensors
        config = ctx.config
        num_tokens = ctx.num_tokens
        topk = ctx.topk
        bwd_intermediate_cache2, bwd_w2 = _dequantized_backward_operands(ctx.quant, intermediate_cache2, w2)

        E, hidden_size, intermediate_size = w2.shape
        CHUNK_SIZE = 64 * 1024

        # Initialize gradient tensors
        grad_intermediate_cache2 = torch.zeros_like(intermediate_cache2)
        grad_w2 = torch.zeros_like(w2)
        grad_topk_weights = torch.zeros_like(topk_weights)

        # Process in chunks to match forward pass
        for chunk in range((num_tokens // CHUNK_SIZE) + 1):
            begin_chunk_idx, end_chunk_idx = (
                chunk * CHUNK_SIZE,
                min((chunk + 1) * CHUNK_SIZE, num_tokens),
            )

            curr_num_tokens = end_chunk_idx - begin_chunk_idx
            if curr_num_tokens == 0:
                continue

            curr_intermediate_cache2 = bwd_intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk]
            curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
            curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
            curr_grad_output = grad_output[begin_chunk_idx:end_chunk_idx]

            # Get aligned metadata
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids, config["BLOCK_SIZE_M"], E
            )

            # Prepare gradient buffers for this chunk
            curr_grad_intermediate_cache2 = torch.zeros_like(curr_intermediate_cache2)
            curr_grad_w2 = torch.zeros_like(w2)
            curr_grad_topk_weights = torch.zeros_like(curr_topk_weights)

            # Call Triton backward kernel with MUL_ROUTED_WEIGHT=True
            # Note: Use top_k=1 to match forward pass indexing
            invoke_fused_moe_backward_kernel(
                grad_output=curr_grad_output,
                input=curr_intermediate_cache2,
                weight=bwd_w2,
                grad_input=curr_grad_intermediate_cache2,
                grad_weight=curr_grad_w2,
                grad_topk_weights=curr_grad_topk_weights,
                topk_weights=curr_topk_weights,
                topk_ids=curr_topk_ids,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=True,
                top_k=1,
                config=config,
                compute_type=tl.bfloat16,
            )

            # Accumulate gradients
            grad_intermediate_cache2[begin_chunk_idx * topk : end_chunk_idx * topk] = curr_grad_intermediate_cache2
            grad_w2 += curr_grad_w2
            grad_topk_weights[begin_chunk_idx:end_chunk_idx] = curr_grad_topk_weights

        return grad_intermediate_cache2, grad_w2, grad_topk_weights, None, None, None


class MoeSumReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        intermediate_cache3: torch.Tensor,
        hidden_states_shape,
    ):
        out_hidden_states = torch.empty(
            hidden_states_shape, device=intermediate_cache3.device, dtype=intermediate_cache3.dtype
        )
        # SGLang's runner combines with this same kernel under deterministic inference (its
        # torch.compile path for small batches is disabled there).
        moe_sum_reduce(
            intermediate_cache3,
            out_hidden_states,
            1.0,
        )
        ctx.save_for_backward(intermediate_cache3)
        return out_hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        (intermediate_cache3,) = ctx.saved_tensors
        return grad_output.unsqueeze(1).expand_as(intermediate_cache3), None


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_fp8: FP8Weight | None = None,
    w2_fp8: FP8Weight | None = None,
    w1_mx: MXTensor | None = None,
    w2_mx: MXTensor | None = None,
):
    """SGLang's Triton fused MoE at its deterministic config. ``w*_fp8`` runs SGLang's block-FP8 W8A8
    path and ``w*_mx`` the microscaling kernels of ``mx_fused_moe``; the backward of both is a bf16
    GEMM on the dequantized operands, with gradients to the bf16 ``w1``/``w2``."""
    assert not ((w1_fp8 or w2_fp8) and (w1_mx or w2_mx)), "pick one quantization"
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.bfloat16]

    intermediate_cache1 = GateUpProjFunction.apply(
        hidden_states,
        w1,
        topk_weights,
        topk_ids,
        w1_fp8,
        w1_mx,
    )
    intermediate_cache2 = SiluAndMulFunction.apply(intermediate_cache1)
    intermediate_cache3 = DownProjFunction.apply(
        intermediate_cache2,
        w2,
        topk_weights,
        topk_ids,
        w2_fp8,
        w2_mx,
    )
    output_hidden_states = MoeSumReduceFunction.apply(
        intermediate_cache3,
        hidden_states.shape,
    )
    return output_hidden_states
