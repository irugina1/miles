"""MXFP4, NVFP4 and MXFP8 linear layers that the trainer and SGLang both run (CUDA, SM100 only).

True on-policy with quantized linears: weights and activations are quantized, and the GEMM runs on
Blackwell's block-scaled tensor-core MMA through ``tl.dot_scaled``. The trainer and the sampler call
the same kernels on the same bf16 inputs, so their outputs are bitwise equal. Two properties make the
kernels batch-invariant, so a token's result does not depend on the rest of the batch:

* every quantizer works on one row at a time. NVFP4's second-level fp32 scale is per token for
  activations, and per weight tensor (per HF ``*_proj`` weight, also inside SGLang's merged
  ``qkv_proj`` / ``gate_up_proj``) for weights;
* the GEMM has one fixed tile config, no split-K, and is not specialized on the number of rows.

Element rounding is round-to-nearest-even. MX block scales are ``2^ceil(log2(amax / elem_max))``
(the block's largest element is never clipped); NVFP4 block scales are e4m3 ``amax / 6 / global``.

The backward is not quantized: bf16 GEMMs (fp32 accumulation) on dequant(Q(W)) and dequant(Q(X)),
straight through to the master weights.

Which format a run uses is part of its true-on-policy contract (miles/true_on_policy); this module only
holds the kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

FORMATS = ("mxfp4", "nvfp4", "mxfp8")


_GROUP = {"mxfp8": 32, "mxfp4": 32, "nvfp4": 16}
_PACK = {"mxfp8": 1, "mxfp4": 2, "nvfp4": 2}  # elements per stored byte
_DOT_FORMAT = {"mxfp8": "e4m3", "mxfp4": "e2m1", "nvfp4": "e2m1"}
NVFP4_GLOBAL_DIV = 448.0 * 6.0  # second-level scale = amax / (e4m3 max * e2m1 max)

# One fixed GEMM configuration. BLOCK_M (the MMA instruction's M) and BLOCK_K (the K chunk of each
# dot_scaled) fix every output's accumulation order: changing either changes the numerics on both sides at
# once. BLOCK_N, the warps and the stages do not; 64 x 4 warps fills more SMs than 128 x 8 at decode batch
# sizes, where the grid is a single row of programs.
BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 128
_NUM_WARPS, _NUM_STAGES = 4, 3
# Blocks per quantizer program; bitwise-neutral (each block is quantized on its own).
_QUANT_BLOCKS = 64


@dataclass(frozen=True)
class MXTensor:
    """Row-quantized [R, K] tensor: ``data`` [R, K / pack], ``scale`` [R, K / group] and, for nvfp4,
    the fp32 second-level ``global_scale`` [R]."""

    fmt: str
    data: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor | None = None


# --------------------------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------------------------


@triton.jit
def _e2m1_code(y):
    """|y| <= 6 -> 4-bit e2m1 code, round to nearest even (ties go to the even mantissa), sign in bit 3."""
    a = tl.abs(y)
    mag = (
        (a > 0.25).to(tl.uint8)
        + (a >= 0.75).to(tl.uint8)
        + (a > 1.25).to(tl.uint8)
        + (a >= 1.75).to(tl.uint8)
        + (a > 2.5).to(tl.uint8)
        + (a >= 3.5).to(tl.uint8)
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
    FMT: tl.constexpr,  # 0 mxfp4, 1 nvfp4, 2 mxfp8 (FORMATS order)
    GROUP: tl.constexpr,
    GB: tl.constexpr,  # groups per program (power of two)
):
    row = tl.program_id(0).to(tl.int64)
    gid = tl.program_id(1) * GB + tl.arange(0, GB)
    gmask = gid < NG
    offs = gid[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    x = tl.load(X + row * stride_x + offs, mask=gmask[:, None], other=0.0).to(tl.float32)  # [GB, GROUP]
    amax = tl.max(tl.abs(x), axis=1)  # [GB]

    if FMT == 1:
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
        elem_max: tl.constexpr = 448.0 if FMT == 2 else 6.0
        r = (amax / elem_max).to(tl.int32, bitcast=True)
        e = ((r >> 23) & 0xFF) - 127 + ((r & 0x7FFFFF) != 0).to(tl.int32)
        e = tl.where(amax > 0, e, -127)
        e = tl.minimum(tl.maximum(e, -127), 127)
        inv = ((127 - e) << 23).to(tl.float32, bitcast=True)  # 2^-e, exact
        y = x * inv[:, None]
        tl.store(S + row * stride_s + gid, (e + 127).to(tl.uint8), mask=gmask)
        if FMT == 2:
            q = tl.minimum(tl.maximum(y, -448.0), 448.0).to(tl.float8e4nv)
            tl.store(Q + row * stride_q + offs, q, mask=gmask[:, None])
        else:
            code = _e2m1_code(y)

    if FMT != 2:
        lo, hi = tl.split(tl.reshape(code, (GB, GROUP // 2, 2)))
        packed = lo | (hi << 4)
        offs_p = gid[:, None] * (GROUP // 2) + tl.arange(0, GROUP // 2)[None, :]
        tl.store(Q + row * stride_q + offs_p, packed, mask=gmask[:, None])


# PyTorch divides a float tensor by a scalar as a multiply by the fp32 reciprocal; the activation scale
# below reproduces ``amax.float() / NVFP4_GLOBAL_DIV`` exactly that way. The dtype is explicit: SGLang
# imports this module while loading a model, under a bf16 default dtype.
_NVFP4_GLOBAL_INV = float(torch.tensor(1.0, dtype=torch.float32) / NVFP4_GLOBAL_DIV)


@triton.jit
def _row_global_scale_kernel(X, G, stride_x, K, INV, BLOCK: tl.constexpr):
    """nvfp4's per-row second-level scale, max|x| / (448 * 6), in one launch: the max is exact and INV is
    the fp32 reciprocal, so it equals ``x.abs().amax(1).float() / NVFP4_GLOBAL_DIV`` bitwise."""
    row = tl.program_id(0).to(tl.int64)
    amax = tl.zeros((BLOCK,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        x = tl.load(X + row * stride_x + offs, mask=offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(x))
    tl.store(G + row, tl.max(amax, axis=0) * INV)


def quantize_rows(
    x: torch.Tensor,
    fmt: str,
    global_scale: torch.Tensor | None = None,
    out: MXTensor | None = None,
) -> MXTensor:
    """Quantize each row of a bf16 [R, K] tensor. ``global_scale`` ([R] fp32) is nvfp4's second level;
    when omitted it is each row's own amax / (448 * 6). With ``out`` the result is written into its
    buffers (an in-place refresh that keeps captured CUDA graphs valid)."""
    assert fmt in FORMATS, fmt
    assert x.dim() == 2 and x.is_cuda and x.dtype == torch.bfloat16, (x.shape, x.dtype)
    x = x.contiguous()
    rows, k = x.shape
    group, pack = _GROUP[fmt], _PACK[fmt]
    assert k % group == 0, (k, group)
    if out is None:
        data_dtype = torch.float8_e4m3fn if fmt == "mxfp8" else torch.uint8
        scale_dtype = torch.float8_e4m3fn if fmt == "nvfp4" else torch.uint8
        out = MXTensor(
            fmt,
            torch.empty(rows, k // pack, device=x.device, dtype=data_dtype),
            torch.empty(rows, k // group, device=x.device, dtype=scale_dtype),
            torch.empty(rows, device=x.device, dtype=torch.float32) if fmt == "nvfp4" else None,
        )
    assert out.fmt == fmt and out.data.shape == (rows, k // pack), (out.fmt, out.data.shape, x.shape)
    if global_scale is not None:
        out.global_scale.copy_(global_scale)
    elif fmt == "nvfp4" and rows:  # each row's own amax / (448 * 6), straight into the buffer
        _row_global_scale_kernel[(rows,)](x, out.global_scale, x.stride(0), k, _NVFP4_GLOBAL_INV, BLOCK=1024, num_warps=4)
    if rows:
        num_groups, groups_per_program = k // group, _QUANT_BLOCKS
        _quantize_rows_kernel[(rows, triton.cdiv(num_groups, groups_per_program))](
            x,
            out.data,
            out.scale,
            out.global_scale if out.global_scale is not None else x,
            x.stride(0),
            out.data.stride(0),
            out.scale.stride(0),
            num_groups,
            FMT=FORMATS.index(fmt),
            GROUP=group,
            GB=groups_per_program,
            num_warps=4,
        )
    return out


def quantize_weight(
    weight: torch.Tensor,
    fmt: str,
    shard_sizes: tuple[int, ...] | None = None,
    out: MXTensor | None = None,
) -> MXTensor:
    """Quantize a bf16 [N, K] weight along K. For nvfp4 each shard (rows of one HF weight inside a
    merged SGLang weight; the whole weight by default) gets its own second-level scale, so a merged
    weight quantizes exactly as its parts do."""
    with torch.no_grad():
        weight = weight.detach()
        global_scale = None
        if fmt == "nvfp4":
            shard_sizes = shard_sizes or (weight.shape[0],)
            assert sum(shard_sizes) == weight.shape[0], (shard_sizes, weight.shape)
            shard_amax = torch.stack([shard.abs().amax() for shard in weight.split(list(shard_sizes))])
            sizes = torch.tensor(shard_sizes, device=weight.device)
            global_scale = (shard_amax.float() / NVFP4_GLOBAL_DIV).repeat_interleave(sizes)
        return quantize_rows(weight, fmt, global_scale, out=out)


def _e2m1_decode(code: torch.Tensor) -> torch.Tensor:
    """4-bit e2m1 codes (uint8, 0..15) -> fp32 values (0, 0.5, 1, 1.5, 2, 3, 4, 6, sign in bit 3), computed
    arithmetically: no host tensor is created, so dequantize can be captured in a CUDA graph."""
    c = code.to(torch.int32)
    e, m = ((c >> 1) & 3).float(), (c & 1).float()
    mag = torch.where(e == 0, 0.5 * m, torch.exp2(e - 1) * (1 + 0.5 * m))
    return torch.where((c & 8) != 0, -mag, mag)


def dequantize(t: MXTensor, dtype: torch.dtype = torch.bfloat16, chunk_rows: int = 1 << 14) -> torch.Tensor:
    """The values the GEMM multiplies (elements x block scale x nvfp4 second-level scale), computed in
    fp32 and rounded once to ``dtype``. Works in row chunks to bound the fp32 temporaries."""
    group, pack = _GROUP[t.fmt], _PACK[t.fmt]
    rows = t.data.shape[0]
    k = t.data.shape[1] * pack
    out = torch.empty(rows, k, device=t.data.device, dtype=dtype)
    for r0 in range(0, rows, chunk_rows):
        d, s = t.data[r0 : r0 + chunk_rows], t.scale[r0 : r0 + chunk_rows]
        if pack == 1:
            values = d.float()
        else:
            d = d.view(torch.uint8)
            values = torch.stack([_e2m1_decode(d & 0xF), _e2m1_decode(d >> 4)], -1).flatten(-2)
        if t.fmt == "nvfp4":
            factor = s.view(torch.float8_e4m3fn).float()
        else:
            factor = torch.exp2(s.view(torch.uint8).float() - 127)
        scaled = (values.view(-1, k // group, group) * factor[..., None]).view(-1, k)
        if t.global_scale is not None:  # applied after the block scale, as the GEMM does
            scaled = scaled * t.global_scale[r0 : r0 + chunk_rows, None]
        out[r0 : r0 + chunk_rows] = scaled.to(dtype)
    return out


# --------------------------------------------------------------------------------------------
# GEMM
# --------------------------------------------------------------------------------------------


# M (the token count) is not specialized: one compiled kernel serves every batch size.
@triton.jit(do_not_specialize=["M"])
def _mx_gemm_kernel(
    A,
    AS,
    AG,
    B,
    BS,
    BG,
    C,
    M,
    N,
    stride_am,
    stride_asm,
    stride_bn,
    stride_bsn,
    stride_cm,
    K: tl.constexpr,
    PACK: tl.constexpr,
    GROUP: tl.constexpr,
    DOT_FMT: tl.constexpr,
    HAS_GLOBAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    KP: tl.constexpr = BLOCK_K // PACK
    KG: tl.constexpr = BLOCK_K // GROUP
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kp = k0 // PACK + tl.arange(0, KP)
        kg = k0 // GROUP + tl.arange(0, KG)
        a = tl.load(A + offs_m[:, None] * stride_am + kp[None, :], mask=m_mask[:, None], other=0)
        a_s = tl.load(AS + offs_m[:, None] * stride_asm + kg[None, :], mask=m_mask[:, None], other=0)
        b = tl.load(B + offs_n[:, None] * stride_bn + kp[None, :], mask=n_mask[:, None], other=0)
        b_s = tl.load(BS + offs_n[:, None] * stride_bsn + kg[None, :], mask=n_mask[:, None], other=0)
        if HAS_GLOBAL:  # nvfp4 block scales are e4m3, passed as raw bytes
            a_s = a_s.to(tl.float8e4nv, bitcast=True)
            b_s = b_s.to(tl.float8e4nv, bitcast=True)
        acc = tl.dot_scaled(a, a_s, DOT_FMT, tl.trans(b), b_s, DOT_FMT, acc)

    if HAS_GLOBAL:
        a_g = tl.load(AG + offs_m, mask=m_mask, other=0.0)
        b_g = tl.load(BG + offs_n, mask=n_mask, other=0.0)
        acc = acc * a_g[:, None] * b_g[None, :]
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def mx_gemm(a: MXTensor, w: MXTensor) -> torch.Tensor:
    """bf16 [M, N] = A [M, K] @ W [N, K].T on the quantized operands (fixed tiles, no split-K)."""
    fmt = w.fmt
    assert a.fmt == fmt, (a.fmt, fmt)
    m, n = a.data.shape[0], w.data.shape[0]
    k = a.data.shape[1] * _PACK[fmt]
    assert k == w.data.shape[1] * _PACK[fmt] and k % BLOCK_K == 0, (k, w.data.shape)
    out = torch.empty(m, n, device=a.data.device, dtype=torch.bfloat16)
    if m == 0:
        return out
    has_global = fmt == "nvfp4"
    raw = lambda t: t.view(torch.uint8)  # noqa: E731  quantized buffers go in as raw bytes
    _mx_gemm_kernel[(triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))](
        raw(a.data),
        raw(a.scale),
        a.global_scale if has_global else raw(a.scale),
        raw(w.data),
        raw(w.scale),
        w.global_scale if has_global else raw(w.scale),
        out,
        m,
        n,
        a.data.stride(0),
        a.scale.stride(0),
        w.data.stride(0),
        w.scale.stride(0),
        out.stride(0),
        K=k,
        PACK=_PACK[fmt],
        GROUP=_GROUP[fmt],
        DOT_FMT=_DOT_FORMAT[fmt],
        HAS_GLOBAL=has_global,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out


# --------------------------------------------------------------------------------------------
# the linear forward both sides run
# --------------------------------------------------------------------------------------------


def mx_linear_inference(x: torch.Tensor, w: MXTensor) -> torch.Tensor:
    """The SGLang side: bf16 [..., K] -> bf16 [..., N] with ``w`` already quantized."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
    return mx_gemm(quantize_rows(x2, w.fmt), w).view(*lead, -1)


class MXLinearFunction(torch.autograd.Function):
    """The trainer side: quantized forward (bitwise the SGLang side), dequantized bf16 backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, fmt: str) -> torch.Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
        qx = quantize_rows(x2, fmt)
        qw = quantize_weight(weight.to(torch.bfloat16), fmt)
        out = mx_gemm(qx, qw)
        # keep the 4/8-bit activation for the backward; the weight is re-quantized there (FSDP frees
        # the unsharded weight between forward and backward)
        ctx.save_for_backward(weight, qx.data, qx.scale, *(() if qx.global_scale is None else (qx.global_scale,)))
        ctx.fmt, ctx.x_shape, ctx.x_dtype = fmt, x.shape, x.dtype
        return out.view(*lead, -1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        weight, qx_data, qx_scale, *qx_global = ctx.saved_tensors
        fmt = ctx.fmt
        qx = MXTensor(fmt, qx_data, qx_scale, qx_global[0] if qx_global else None)
        qw = quantize_weight(weight.to(torch.bfloat16), fmt)
        g = grad_out.reshape(-1, grad_out.shape[-1]).to(torch.bfloat16)
        with torch.autocast(device_type="cuda", enabled=False):
            grad_x = g @ dequantize(qw)
            grad_w = g.t() @ dequantize(qx)
        return grad_x.view(ctx.x_shape).to(ctx.x_dtype), grad_w.to(weight.dtype), None


class MXLinear(torch.nn.Linear):
    """An ``nn.Linear`` (no bias) whose forward is ``MXLinearFunction``; set by class swap so the
    parameters, their names and FSDP sharding are untouched."""

    mx_fmt: str

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MXLinearFunction.apply(x, self.weight, self.mx_fmt)


def swap_to_mx_linear(linear: torch.nn.Linear, fmt: str) -> None:
    if linear.bias is not None:
        raise ValueError("MXLinear supports bias-free linears only")
    linear.__class__ = MXLinear
    linear.mx_fmt = fmt
