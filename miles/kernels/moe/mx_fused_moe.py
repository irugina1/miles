"""MXFP8, MXFP4 and NVFP4 fused-MoE kernels that both the trainer and SGLang run.

The GEMMs use Blackwell's block-scaled tensor-core MMA through ``tl.dot_scaled``, with weights and
activations quantized. Each quantizer works on one row at a time (NVFP4 uses a per-token activation
scale) and the GEMM uses one fixed tile config without split-K, so a token's result does not depend
on the rest of the batch. See examples/infra_features/true_on_policy/README.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

FORMATS = ("mxfp8", "mxfp4", "nvfp4")

# Read by the trainer and by the SGLang plugin; the launcher sets it for every process.
MOE_QUANT_ENV = "MILES_MOE_QUANT"


def moe_mx_format() -> str | None:
    fmt = os.environ.get(MOE_QUANT_ENV) or None
    if fmt is not None and fmt not in FORMATS:
        raise ValueError(f"{MOE_QUANT_ENV}={fmt!r}; expected one of {FORMATS}")
    return fmt


_GROUP = {"mxfp8": 32, "mxfp4": 32, "nvfp4": 16}
_PACK = {"mxfp8": 1, "mxfp4": 2, "nvfp4": 2}  # elements per stored byte
_DOT_FORMAT = {"mxfp8": "e4m3", "mxfp4": "e2m1", "nvfp4": "e2m1"}
_FP4_MAX, _FP8_MAX = 6.0, 448.0
NVFP4_GLOBAL_DIV = _FP8_MAX * _FP4_MAX  # second-level scale = amax / (448 * 6)

# One fixed GEMM configuration: changing it changes the numerics on both sides at once.
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
_NUM_WARPS, _NUM_STAGES = 8, 3


@dataclass(frozen=True)
class MXTensor:
    """Row-quantized 2-D/3-D tensor: ``data`` [..., K / pack], ``scale`` [..., K / group] and,
    for nvfp4, the fp32 second-level ``global_scale`` ([rows] for activations, [E] for experts)."""

    fmt: str
    data: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor | None = None


# --------------------------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------------------------


@triton.jit
def _e2m1_code(y):
    """|y| <= 6 -> 4-bit e2m1 code (ties round toward zero), sign in bit 3."""
    a = tl.abs(y)
    mag = (
        (a > 0.25).to(tl.uint8)
        + (a > 0.75).to(tl.uint8)
        + (a > 1.25).to(tl.uint8)
        + (a > 1.75).to(tl.uint8)
        + (a > 2.5).to(tl.uint8)
        + (a > 3.5).to(tl.uint8)
        + (a > 5.0).to(tl.uint8)
    )
    return mag | ((y < 0).to(tl.uint8) << 3)


@triton.jit
def _quantize_rows_kernel(
    X,
    Q,
    S,
    G,
    stride_x,
    stride_q,
    stride_s,
    NG,  # groups per row
    FMT: tl.constexpr,  # 0 mxfp8, 1 mxfp4, 2 nvfp4
    GROUP: tl.constexpr,
    GB: tl.constexpr,  # groups per program (power of two)
):
    row = tl.program_id(0).to(tl.int64)
    gid = tl.program_id(1) * GB + tl.arange(0, GB)
    gmask = gid < NG
    offs = gid[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    x = tl.load(X + row * stride_x + offs, mask=gmask[:, None], other=0.0).to(tl.float32)  # [GB, GROUP]
    amax = tl.max(tl.abs(x), axis=1)  # [GB]

    if FMT == 2:
        # nvfp4: e4m3 block scale relative to the row's fp32 second-level scale
        g = tl.load(G + row)
        bs = amax / 6.0 / tl.where(g > 0, g, 1.0)
        bs8 = tl.minimum(bs, 448.0).to(tl.float8e4nv)
        denom = bs8.to(tl.float32) * g
        y = x / tl.where(denom > 0, denom, 1.0)[:, None]
        y = tl.where(denom[:, None] > 0, y, 0.0)
        code = _e2m1_code(tl.minimum(tl.maximum(y, -6.0), 6.0))
        tl.store(S + row * stride_s + gid, bs8, mask=gmask)
    else:
        # e8m0 scale 2^e with e = ceil(log2(amax / elem_max)), computed exactly from the bits
        elem_max: tl.constexpr = 448.0 if FMT == 0 else 6.0
        r = (amax / elem_max).to(tl.int32, bitcast=True)
        e = ((r >> 23) & 0xFF) - 127 + ((r & 0x7FFFFF) != 0).to(tl.int32)
        e = tl.where(amax > 0, e, -127)
        e = tl.minimum(tl.maximum(e, -127), 127)
        inv = ((127 - e) << 23).to(tl.float32, bitcast=True)  # 2^-e, exact
        y = x * inv[:, None]
        tl.store(S + row * stride_s + gid, (e + 127).to(tl.uint8), mask=gmask)
        if FMT == 0:
            q = tl.minimum(tl.maximum(y, -448.0), 448.0).to(tl.float8e4nv)
            tl.store(Q + row * stride_q + offs, q, mask=gmask[:, None])
        else:
            code = _e2m1_code(y)

    if FMT != 0:
        lo, hi = tl.split(tl.reshape(code, (GB, GROUP // 2, 2)))
        packed = lo | (hi << 4)
        offs_p = gid[:, None] * (GROUP // 2) + tl.arange(0, GROUP // 2)[None, :]
        tl.store(Q + row * stride_q + offs_p, packed, mask=gmask[:, None])


def quantize_rows(x: torch.Tensor, fmt: str, global_scale: torch.Tensor | None = None) -> MXTensor:
    """Quantize each row of a [R, K] tensor. ``global_scale`` ([R] fp32) is nvfp4's second level;
    when omitted it is the row's own amax / (448 * 6)."""
    assert fmt in FORMATS, fmt
    assert x.dim() == 2 and x.is_cuda
    x = x.contiguous()
    rows, k = x.shape
    group, pack = _GROUP[fmt], _PACK[fmt]
    assert k % group == 0, (k, group)
    q = torch.empty(rows, k // pack, device=x.device, dtype=torch.float8_e4m3fn if fmt == "mxfp8" else torch.uint8)
    s = torch.empty(rows, k // group, device=x.device, dtype=torch.float8_e4m3fn if fmt == "nvfp4" else torch.uint8)
    if fmt == "nvfp4" and global_scale is None:
        global_scale = x.abs().amax(dim=1).float() / NVFP4_GLOBAL_DIV
    if rows:
        num_groups, groups_per_program = k // group, 8
        _quantize_rows_kernel[(rows, triton.cdiv(num_groups, groups_per_program))](
            x,
            q,
            s,
            global_scale if global_scale is not None else x,
            x.stride(0),
            q.stride(0),
            s.stride(0),
            num_groups,
            FMT=FORMATS.index(fmt),
            GROUP=group,
            GB=groups_per_program,
            num_warps=4,
        )
    return MXTensor(fmt, q, s, global_scale.contiguous() if global_scale is not None else None)


def quantize_experts(weight: torch.Tensor, fmt: str, out: MXTensor | None = None) -> MXTensor:
    """[E, N, K] bf16 expert weights -> rows quantized along K; nvfp4 uses one second-level scale
    per expert. With ``out`` the result is written into its buffers (CUDA-graph safe refresh)."""
    num_experts, n, k = weight.shape
    with torch.no_grad():
        flat = weight.detach().reshape(num_experts * n, k)
        expert_scale = None
        row_scale = None
        if fmt == "nvfp4":
            expert_scale = weight.detach().abs().amax(dim=(1, 2)).float() / NVFP4_GLOBAL_DIV
            row_scale = expert_scale.repeat_interleave(n)
        q = quantize_rows(flat, fmt, row_scale)
        result = MXTensor(
            fmt,
            q.data.view(num_experts, n, -1),
            q.scale.view(num_experts, n, -1),
            expert_scale,
        )
        if out is None:
            return result
        out.data.copy_(result.data)
        out.scale.copy_(result.scale)
        if out.global_scale is not None:
            out.global_scale.copy_(result.global_scale)
        return out


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequantize(t: MXTensor, dtype: torch.dtype = torch.bfloat16, chunk_rows: int = 1 << 16) -> torch.Tensor:
    """The values the GEMM multiplies (elements x block scale x nvfp4 second-level scale), computed
    in fp32 and rounded once to ``dtype``. Works in row chunks to bound the fp32 temporaries."""
    group, pack = _GROUP[t.fmt], _PACK[t.fmt]
    lead = t.data.shape[:-1]
    k = t.data.shape[-1] * pack
    data = t.data.reshape(-1, t.data.shape[-1])
    scale = t.scale.reshape(-1, t.scale.shape[-1])
    glob = None
    if t.fmt == "nvfp4":
        rows_per_scale = data.shape[0] // t.global_scale.numel()  # 1 for activations, N for experts
        glob = t.global_scale.float().repeat_interleave(rows_per_scale)
    lut = torch.tensor(_E2M1_VALUES, device=data.device) if pack == 2 else None
    out = torch.empty(data.shape[0], k, device=data.device, dtype=dtype)
    for r0 in range(0, data.shape[0], chunk_rows):
        d, s = data[r0 : r0 + chunk_rows], scale[r0 : r0 + chunk_rows]
        if pack == 1:
            values = d.float()
        else:
            d = d.view(torch.uint8)
            values = torch.stack([lut[(d & 0xF).long()], lut[(d >> 4).long()]], -1).flatten(-2)
        if t.fmt == "nvfp4":
            factor = s.view(torch.float8_e4m3fn).float()
        else:
            factor = torch.exp2(s.view(torch.uint8).float() - 127)
        scaled = (values.view(-1, k // group, group) * factor[..., None]).view(-1, k)
        if glob is not None:  # applied after the block scale, as the GEMM does
            scaled = scaled * glob[r0 : r0 + chunk_rows, None]
        out[r0 : r0 + chunk_rows] = scaled.to(dtype)
    return out.view(*lead, k)


# --------------------------------------------------------------------------------------------
# grouped GEMM
# --------------------------------------------------------------------------------------------


@triton.jit
def _mx_moe_gemm_kernel(
    A,
    AS,
    AG,
    B,
    BS,
    BG,
    C,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    num_valid_tokens,
    N,
    stride_am,
    stride_asm,
    stride_be,
    stride_bn,
    stride_bse,
    stride_bsn,
    stride_cm,
    K: tl.constexpr,
    A_TOPK: tl.constexpr,
    PACK: tl.constexpr,
    GROUP: tl.constexpr,
    DOT_FMT: tl.constexpr,
    HAS_GLOBAL: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_m * BLOCK_M >= tl.load(num_tokens_post_padded):
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids + offs_m).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    a_row = tl.where(token_mask, offs_token // A_TOPK, 0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    expert = tl.load(expert_ids + pid_m).to(tl.int64)

    c_ptrs = C + offs_token[:, None] * stride_cm + offs_n[None, :]
    c_mask = token_mask[:, None] & n_mask[None, :]
    if expert == -1:
        tl.store(c_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=C.dtype.element_ty), mask=c_mask)
        return

    KP: tl.constexpr = BLOCK_K // PACK
    KG: tl.constexpr = BLOCK_K // GROUP
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kp = k0 // PACK + tl.arange(0, KP)
        kg = k0 // GROUP + tl.arange(0, KG)
        a = tl.load(A + a_row[:, None] * stride_am + kp[None, :], mask=token_mask[:, None], other=0)
        a_s = tl.load(AS + a_row[:, None] * stride_asm + kg[None, :], mask=token_mask[:, None], other=0)
        b = tl.load(B + expert * stride_be + offs_n[:, None] * stride_bn + kp[None, :], mask=n_mask[:, None], other=0)
        b_s = tl.load(
            BS + expert * stride_bse + offs_n[:, None] * stride_bsn + kg[None, :], mask=n_mask[:, None], other=0
        )
        if HAS_GLOBAL:  # nvfp4 block scales are e4m3, passed as raw bytes
            a_s = a_s.to(tl.float8e4nv, bitcast=True)
            b_s = b_s.to(tl.float8e4nv, bitcast=True)
        acc = tl.dot_scaled(a, a_s, DOT_FMT, tl.trans(b), b_s, DOT_FMT, acc)

    if HAS_GLOBAL:
        acc = acc * tl.load(AG + a_row, mask=token_mask, other=0.0)[:, None] * tl.load(BG + expert)
    if MUL_ROUTED_WEIGHT:
        w = tl.load(topk_weights + offs_token, mask=token_mask, other=0.0).to(tl.float32)
        acc = acc * w[:, None]
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)


def mx_moe_gemm(
    a: MXTensor,
    w: MXTensor,
    out: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    a_topk: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    """out[t] = A[t // a_topk] @ W[expert(t)].T for every routed row t (fixed tiles, no split-K)."""
    fmt = w.fmt
    assert a.fmt == fmt
    num_experts, n, _ = w.data.shape
    k = a.data.shape[1] * _PACK[fmt]
    assert k % BLOCK_K == 0, k
    grid = (triton.cdiv(sorted_token_ids.shape[0], BLOCK_M) * triton.cdiv(n, BLOCK_N),)
    has_global = fmt == "nvfp4"
    raw = lambda t: t.view(torch.uint8)  # noqa: E731  quantized buffers go in as raw bytes
    _mx_moe_gemm_kernel[grid](
        raw(a.data),
        raw(a.scale),
        a.global_scale if has_global else raw(a.scale),
        raw(w.data),
        raw(w.scale),
        w.global_scale if has_global else raw(w.scale),
        out,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_valid_tokens,
        n,
        a.data.stride(0),
        a.scale.stride(0),
        w.data.stride(0),
        w.data.stride(1),
        w.scale.stride(0),
        w.scale.stride(1),
        out.stride(0),
        K=k,
        A_TOPK=a_topk,
        PACK=_PACK[fmt],
        GROUP=_GROUP[fmt],
        DOT_FMT=_DOT_FORMAT[fmt],
        HAS_GLOBAL=has_global,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out


# --------------------------------------------------------------------------------------------
# the MoE forward both sides run
# --------------------------------------------------------------------------------------------


def align(topk_ids: torch.Tensor, num_experts: int):
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size

    return moe_align_block_size(topk_ids, BLOCK_M, num_experts)


def gate_up(hidden_states, w13: MXTensor, topk_weights, topk_ids, routing) -> torch.Tensor:
    """[T, H] -> [T * topk, 2I] bf16 (pre-activation)."""
    num_tokens, topk = topk_ids.shape
    out = torch.empty(num_tokens * topk, w13.data.shape[1], device=hidden_states.device, dtype=torch.bfloat16)
    if num_tokens == 0:
        return out
    a = quantize_rows(hidden_states, w13.fmt)
    return mx_moe_gemm(a, w13, out, topk_weights, *routing, num_tokens * topk, a_topk=topk, mul_routed_weight=False)


def down(intermediate, w2: MXTensor, topk_weights, topk_ids, routing) -> torch.Tensor:
    """[T * topk, I] -> [T, topk, H] bf16, already multiplied by the routing weights."""
    num_tokens, topk = topk_ids.shape
    out = torch.empty(num_tokens, topk, w2.data.shape[1], device=intermediate.device, dtype=torch.bfloat16)
    if num_tokens == 0:
        return out
    a = quantize_rows(intermediate, w2.fmt)
    mx_moe_gemm(
        a,
        w2,
        out.view(num_tokens * topk, -1),
        topk_weights,
        *routing,
        num_tokens * topk,
        a_topk=1,
        mul_routed_weight=True,
    )
    return out


def mx_fused_experts(hidden_states, w13: MXTensor, w2: MXTensor, topk_weights, topk_ids) -> torch.Tensor:
    """Inference forward (the SGLang side); the trainer runs the same steps inside autograd."""
    from sgl_kernel import moe_sum_reduce
    from sglang.kernels.ops.activation.activation import silu_and_mul

    num_tokens = hidden_states.shape[0]
    routing = align(topk_ids, w13.data.shape[0])
    cache1 = gate_up(hidden_states, w13, topk_weights, topk_ids, routing)
    cache2 = torch.empty(cache1.shape[0], cache1.shape[1] // 2, device=cache1.device, dtype=cache1.dtype)
    if num_tokens:
        silu_and_mul(cache1, cache2)
    cache3 = down(cache2, w2, topk_weights, topk_ids, routing)
    out = torch.empty(num_tokens, cache3.shape[-1], device=cache3.device, dtype=cache3.dtype)
    if num_tokens:
        moe_sum_reduce(cache3, out, 1.0)
    return out
