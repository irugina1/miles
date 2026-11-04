"""Block-FP8 forward of the true-on-policy fused experts.

The forward runs SGLang's FP8 W8A8 Triton path on weights quantized like the FSDP weight sync
quantizes them; the backward is dequantized (bf16 GEMMs on the dequantized FP8 operands) and
straight-through to the bf16 weights.
"""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
pytest.importorskip("sglang.kernels.ops.moe.fused_moe_triton_kernels")

from miles.kernels.moe.fused_experts import quantize_experts_fp8 as _quantize_experts  # noqa: E402
from miles.backends.fsdp_utils.quantization import fp8_block_quantize  # noqa: E402
from miles.kernels.moe.fused_experts import fused_experts  # noqa: E402

E, H, I, TOPK, T = 8, 256, 256, 2, 96
BLOCK = [128, 128]


def _inputs(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, generator=g)
    w13 = (torch.randn(E, 2 * I, H, device="cuda", generator=g) / H**0.5).bfloat16().requires_grad_()
    w2 = (torch.randn(E, H, I, device="cuda", generator=g) / I**0.5).bfloat16().requires_grad_()
    logits = torch.randn(T, E, device="cuda", generator=g)
    weights, ids = torch.topk(logits.softmax(-1), TOPK, dim=-1)
    return x, w13, w2, (weights / weights.sum(-1, keepdim=True)).bfloat16(), ids


def test_batched_quantization_equals_the_per_matrix_quantization_of_the_weight_sync():
    _, w13, _, _, _ = _inputs()
    batched = _quantize_experts(w13, BLOCK)
    for e in range(E):
        for half in (slice(0, I), slice(I, 2 * I)):  # gate_proj / up_proj as separate HF tensors
            qweight, scale = fp8_block_quantize(w13[e, half].detach(), BLOCK)
            assert torch.equal(batched.weight[e, half].view(torch.uint8), qweight.view(torch.uint8))
            rows = slice(half.start // BLOCK[0], half.stop // BLOCK[0])
            assert torch.equal(batched.scale[e, rows], scale)


def test_fp8_forward_is_close_to_bf16_and_backward_is_straight_through():
    x, w13, w2, weights, ids = _inputs()
    ref = fused_experts(x, w13, w2, weights, ids)
    out = fused_experts(
        x, w13, w2, weights, ids, w1_fp8=_quantize_experts(w13, BLOCK), w2_fp8=_quantize_experts(w2, BLOCK)
    )
    assert not torch.equal(out, ref)
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.1, rel

    out.float().square().sum().backward()
    assert w13.grad is not None and w2.grad is not None
    assert torch.isfinite(w13.grad).all() and torch.isfinite(w2.grad).all()
    assert w13.grad.abs().sum() > 0 and w2.grad.abs().sum() > 0


def test_fp8_forward_is_deterministic():
    x, w13, w2, weights, ids = _inputs(seed=3)
    fp8 = dict(w1_fp8=_quantize_experts(w13, BLOCK), w2_fp8=_quantize_experts(w2, BLOCK))
    with torch.no_grad():
        assert torch.equal(
            fused_experts(x, w13, w2, weights, ids, **fp8), fused_experts(x, w13, w2, weights, ids, **fp8)
        )


def _ste(t, fake_quant):
    """t in the forward replaced by its dequantized quantization; identity in the backward."""
    return t + (fake_quant(t.detach()).to(t.dtype) - t).detach()


def _dequant_weights(w):
    q = _quantize_experts(w, BLOCK)
    scale = q.scale.repeat_interleave(BLOCK[0], 1).repeat_interleave(BLOCK[1], 2)
    return (q.weight.float() * scale).bfloat16()


def _dequant_activations(x):
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

    q, s = sglang_per_token_group_quant_fp8(x.contiguous(), BLOCK[1])
    return (q.float() * s.repeat_interleave(BLOCK[1], 1)).bfloat16()


def _grouped_gemm_reference(a, w, ids, a_topk, quantized):
    """fp32 autograd: row r is A[r // a_topk] @ W[ids.flat[r]].T, with both operands replaced by
    their dequantized FP8 values through straight-through estimators when ``quantized``."""
    if quantized:
        a, w = _ste(a, _dequant_activations), _ste(w, _dequant_weights)
    rows = torch.arange(ids.numel(), device="cuda")
    return torch.einsum("rk,rnk->rn", a[rows // a_topk].float(), w[ids.reshape(-1)].float())


def _grads(fn, *tensors, probe):
    leaves = [t.detach().clone().requires_grad_() for t in tensors]
    (fn(*leaves).float() * probe).sum().backward()
    return [t.grad.float() for t in leaves]


def _rel_errors(got, want):
    return [((g - w).norm() / w.norm()).item() for g, w in zip(got, want, strict=True)]


def _assert_dequantized_backward(got, want, plain, floor):
    """``got`` (quantized kernels) matches the dequantized-operand reference ``want`` as closely as
    the same bf16 kernels without quantization match the unquantized reference (``floor``: bf16
    rounding only), and is clearly closer to it than straight-through on unquantized operands."""
    for i, (g, w, p, f) in enumerate(zip(got, want, plain, floor, strict=True)):
        err = ((g - w).norm() / w.norm()).item()
        plain_err = ((p - w).norm() / w.norm()).item()
        assert err < 1.25 * f + 1e-3, (i, err, f)
        assert err < plain_err / 2, (i, err, plain_err)


def test_fp8_gate_up_backward_is_the_dequantized_backward():
    from miles.kernels.moe.fused_experts import GateUpProjFunction

    x, w13, _, weights, ids = _inputs(seed=5)
    probe = torch.randn(T * TOPK, 2 * I, device="cuda")

    def kernel(x, w13):
        return GateUpProjFunction.apply(x, w13, weights, ids, _quantize_experts(w13, BLOCK), None)

    got = _grads(kernel, x, w13, probe=probe)
    want = _grads(lambda x, w: _grouped_gemm_reference(x, w, ids, TOPK, True), x, w13, probe=probe)
    plain = _grads(lambda x, w: _grouped_gemm_reference(x, w, ids, TOPK, False), x, w13, probe=probe)
    bf16 = _grads(lambda x, w: GateUpProjFunction.apply(x, w, weights, ids, None, None), x, w13, probe=probe)
    _assert_dequantized_backward(got, want, plain, _rel_errors(bf16, plain))


def test_fp8_down_backward_is_the_dequantized_backward():
    """Also covers the routing-weight gradient, which needs the quantized product itself."""
    from miles.kernels.moe.fused_experts import DownProjFunction

    _, _, w2, weights, ids = _inputs(seed=6)
    act = torch.randn(T * TOPK, I, device="cuda").bfloat16()
    probe = torch.randn(T, TOPK, H, device="cuda")

    def kernel(act, w2, weights):
        return DownProjFunction.apply(act, w2, weights, ids, _quantize_experts(w2, BLOCK), None)

    def reference(quantized):
        def fn(act, w2, weights):
            out = _grouped_gemm_reference(act, w2, ids, 1, quantized).view(T, TOPK, H)
            return out * weights.float()[..., None]

        return fn

    got = _grads(kernel, act, w2, weights, probe=probe)
    want = _grads(reference(True), act, w2, weights, probe=probe)
    plain = _grads(reference(False), act, w2, weights, probe=probe)
    bf16 = _grads(lambda a, w, wt: DownProjFunction.apply(a, w, wt, ids, None, None), act, w2, weights, probe=probe)
    _assert_dequantized_backward(got, want, plain, _rel_errors(bf16, plain))
