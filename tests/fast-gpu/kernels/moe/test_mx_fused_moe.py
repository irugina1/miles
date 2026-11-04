"""MXFP8 / MXFP4 / NVFP4 fused MoE: accuracy against a dequantized reference, and the batch
invariance that zero train-inference mismatch rests on (a token's output must not depend on which
other tokens share its batch)."""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="stage-c-4-gpu-b200", labels=["precision"], hardware=["blackwell"], num_gpus=1)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 10,
        reason="native block-scaled MMA needs Blackwell",
    ),
]
pytest.importorskip("sglang.kernels.ops.activation.activation")

from miles.kernels.moe import mx_fused_moe as mx  # noqa: E402

E, H, I, TOPK = 16, 256, 384, 4  # I / 32 = 12 groups: exercises non-power-of-two group counts
_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def dequantize(t: mx.MXTensor) -> torch.Tensor:
    data, scale = t.data, t.scale
    if t.fmt == "mxfp8":
        values = data.float()
    else:
        lut = _E2M1.to(data.device)
        values = torch.stack([lut[(data & 0xF).long()], lut[(data >> 4).long()]], -1).flatten(-2)
    if t.fmt == "nvfp4":
        factor = scale.float()
        glob = t.global_scale
        glob = glob[:, None, None] if values.dim() == 3 else glob[:, None]
    else:
        factor = torch.exp2(scale.float() - 127)
        glob = 1.0
    group = values.shape[-1] // factor.shape[-1]
    return values * factor.repeat_interleave(group, -1) * glob


def _routing(num_tokens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    probs = torch.randn(num_tokens, E, device="cuda", generator=g).softmax(-1)
    weights, ids = torch.topk(probs, TOPK, dim=-1)
    return (weights / weights.sum(-1, keepdim=True)).bfloat16(), ids


def _weights(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w13 = (torch.randn(E, 2 * I, H, device="cuda", generator=g) / H**0.5).bfloat16()
    w2 = (torch.randn(E, H, I, device="cuda", generator=g) / I**0.5).bfloat16()
    return w13, w2


@pytest.mark.parametrize("fmt", mx.FORMATS)
def test_quantize_roundtrip(fmt):
    x = torch.randn(64, I, device="cuda").bfloat16()
    q = mx.quantize_rows(x, fmt)
    rel = (dequantize(q) - x.float()).norm() / x.float().norm()
    assert rel < (0.05 if fmt == "mxfp8" else 0.15), rel
    # rows are quantized independently: a row alone gives the same bits as in the batch
    alone = mx.quantize_rows(x[5:6], fmt)
    assert torch.equal(alone.data.view(torch.uint8), q.data[5:6].view(torch.uint8))
    assert torch.equal(alone.scale.view(torch.uint8), q.scale[5:6].view(torch.uint8))


@pytest.mark.parametrize("fmt", mx.FORMATS)
def test_moe_matches_dequantized_reference(fmt):
    w13, w2 = _weights()
    q13, q2 = mx.quantize_experts(w13, fmt), mx.quantize_experts(w2, fmt)
    x = torch.randn(96, H, device="cuda").bfloat16()
    weights, ids = _routing(96, seed=1)
    out = mx.mx_fused_experts(x, q13, q2, weights, ids)

    # reference with the very same quantized operands, fp32 math
    d13, d2 = dequantize(q13), dequantize(q2)
    ref = torch.zeros(96, H, device="cuda")
    xq = dequantize(mx.quantize_rows(x, fmt))
    for t in range(96):
        for j in range(TOPK):
            e = ids[t, j]
            h = (xq[t] @ d13[e].T).bfloat16().float()  # the pipeline stores gate/up in bf16
            act = (torch.nn.functional.silu(h[:I]) * h[I:]).bfloat16()
            actq = dequantize(mx.quantize_rows(act[None], fmt))[0]
            ref[t] += (actq @ d2[e].T) * weights[t, j].float()
    rel = (out.float() - ref).norm() / ref.norm()
    assert rel < 2e-2, rel


@pytest.mark.parametrize("fmt", mx.FORMATS)
def test_moe_output_is_batch_invariant(fmt):
    w13, w2 = _weights(seed=2)
    q13, q2 = mx.quantize_experts(w13, fmt), mx.quantize_experts(w2, fmt)
    n = 700
    x = torch.randn(n, H, device="cuda").bfloat16()
    weights, ids = _routing(n, seed=3)
    full = mx.mx_fused_experts(x, q13, q2, weights, ids)
    perm = torch.randperm(n, device="cuda")
    shuffled = mx.mx_fused_experts(x[perm], q13, q2, weights[perm], ids[perm])
    assert torch.equal(shuffled, full[perm])
    for rows in (slice(0, 1), slice(3, 10), slice(100, 357)):
        part = mx.mx_fused_experts(x[rows], q13, q2, weights[rows], ids[rows])
        assert torch.equal(part, full[rows]), rows


@pytest.mark.parametrize("fmt", mx.FORMATS)
def test_in_place_refresh_matches_fresh_quantization(fmt):
    w13, _ = _weights(seed=4)
    buffers = mx.quantize_experts(torch.zeros_like(w13), fmt)
    refreshed = mx.quantize_experts(w13, fmt, out=buffers)
    fresh = mx.quantize_experts(w13, fmt)
    assert refreshed is buffers
    assert torch.equal(refreshed.data.view(torch.uint8), fresh.data.view(torch.uint8))
    assert torch.equal(refreshed.scale.view(torch.uint8), fresh.scale.view(torch.uint8))


@pytest.mark.parametrize("fmt", mx.FORMATS)
def test_training_forward_equals_the_rollout_forward_and_has_gradients(fmt):
    """The trainer's autograd path (fused_experts with w*_mx) must be bitwise the engine's path."""
    from miles.kernels.moe.fused_experts import fused_experts

    w13, w2 = _weights(seed=5)
    w13.requires_grad_()
    w2.requires_grad_()
    x = torch.randn(200, H, device="cuda").bfloat16()
    weights, ids = _routing(200, seed=6)
    q13, q2 = mx.quantize_experts(w13, fmt), mx.quantize_experts(w2, fmt)

    train = fused_experts(x, w13, w2, weights, ids, w1_mx=q13, w2_mx=q2)
    rollout = mx.mx_fused_experts(x, q13, q2, weights, ids)
    assert torch.equal(train, rollout)

    train.float().square().sum().backward()
    for grad in (w13.grad, w2.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize("fmt", mx.FORMATS)
def test_dequantize_matches_the_reference_dequantization(fmt):
    w13, _ = _weights(seed=8)
    q = mx.quantize_experts(w13, fmt)
    assert torch.equal(mx.dequantize(q, torch.float32, chunk_rows=100), dequantize(q).float())
    x = torch.randn(70, H, device="cuda").bfloat16()
    qx = mx.quantize_rows(x, fmt)
    assert torch.equal(mx.dequantize(qx, torch.float32), dequantize(qx).float())


def _ste(t, fake_quant):
    """t in the forward replaced by its dequantized quantization; identity in the backward."""
    return t + (fake_quant(t.detach()).to(t.dtype) - t).detach()


def _grouped_gemm_reference(a, w, ids, a_topk, quant):
    """fp32 autograd: row r is A[r // a_topk] @ W[ids.flat[r]].T, with both operands replaced by
    dequant(Q(.)) through straight-through estimators (``quant=None``: unquantized operands)."""
    if quant is not None:
        a = _ste(a, lambda v: dequantize(mx.quantize_rows(v, quant)))
        w = _ste(w, lambda v: dequantize(mx.quantize_experts(v, quant)))
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


@pytest.mark.parametrize("fmt", mx.FORMATS)
def test_gate_up_backward_is_the_dequantized_backward(fmt):
    from miles.kernels.moe.fused_experts import GateUpProjFunction

    w13, _ = _weights(seed=9)
    x = torch.randn(96, H, device="cuda").bfloat16()
    weights, ids = _routing(96, seed=10)
    probe = torch.randn(96 * TOPK, 2 * I, device="cuda")

    def kernel(x, w13):
        return GateUpProjFunction.apply(x, w13, weights, ids, None, mx.quantize_experts(w13, fmt))

    got = _grads(kernel, x, w13, probe=probe)
    want = _grads(lambda x, w: _grouped_gemm_reference(x, w, ids, TOPK, fmt), x, w13, probe=probe)
    plain = _grads(lambda x, w: _grouped_gemm_reference(x, w, ids, TOPK, None), x, w13, probe=probe)
    bf16 = _grads(lambda x, w: GateUpProjFunction.apply(x, w, weights, ids, None, None), x, w13, probe=probe)
    _assert_dequantized_backward(got, want, plain, _rel_errors(bf16, plain))


@pytest.mark.parametrize("fmt", mx.FORMATS)
def test_down_backward_is_the_dequantized_backward(fmt):
    """Also covers the routing-weight gradient, which needs the quantized product itself."""
    from miles.kernels.moe.fused_experts import DownProjFunction

    _, w2 = _weights(seed=11)
    act = torch.randn(96 * TOPK, I, device="cuda").bfloat16()
    weights, ids = _routing(96, seed=12)
    probe = torch.randn(96, TOPK, H, device="cuda")

    def kernel(act, w2, weights):
        return DownProjFunction.apply(act, w2, weights, ids, None, mx.quantize_experts(w2, fmt))

    def reference(quant):
        def fn(act, w2, weights):
            out = _grouped_gemm_reference(act, w2, ids, 1, quant).view(96, TOPK, H)
            return out * weights.float()[..., None]

        return fn

    got = _grads(kernel, act, w2, weights, probe=probe)
    want = _grads(reference(fmt), act, w2, weights, probe=probe)
    plain = _grads(reference(None), act, w2, weights, probe=probe)
    bf16 = _grads(lambda a, w, wt: DownProjFunction.apply(a, w, wt, ids, None, None), act, w2, weights, probe=probe)
    _assert_dequantized_backward(got, want, plain, _rel_errors(bf16, plain))
