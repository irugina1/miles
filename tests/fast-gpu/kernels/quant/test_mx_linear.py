"""Miles' quantized linear (miles/kernels/quant/mx_linear.py): the trainer's forward is bitwise the SGLang
side's, a token's result does not depend on its batch, and SGLang's merged projections quantize exactly
as the trainer's separate ones."""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

# est_time is an unmeasured upper bound
register_cuda_ci(est_time=300, suite="stage-c-4-gpu-b200", labels=["precision"], hardware=["blackwell"], num_gpus=1)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="needs SM100"
)

FORMATS = ("mxfp4", "nvfp4", "mxfp8")
HIDDEN, Q, KV, INTER = 2560, 4096, 1024, 9728  # Qwen3-4B


def _weights(sizes, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # different magnitudes per shard, so a shared nvfp4 second-level scale would change the result
    return [torch.randn(n, k, device="cuda", generator=g) * 0.02 * (i + 1) for i, n in enumerate(sizes)]


def _activations(m, k, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=g)
    x[:, :7] *= 30  # outlier channels
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("fmt", FORMATS)
def test_token_result_does_not_depend_on_batch(fmt):
    from miles.kernels.quant.mx_linear import mx_linear_inference, quantize_weight

    (w,) = _weights([KV], HIDDEN)
    qw = quantize_weight(w.to(torch.bfloat16), fmt)
    x = _activations(1000, HIDDEN)
    full = mx_linear_inference(x, qw)
    for rows in (slice(0, 1), slice(5, 6), slice(3, 130), slice(127, 129), slice(900, 1000)):
        assert torch.equal(mx_linear_inference(x[rows], qw), full[rows]), rows
    perm = torch.randperm(1000, device="cuda")
    assert torch.equal(mx_linear_inference(x[perm], qw), full[perm])


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("sizes,k", [((Q, KV, KV), HIDDEN), ((INTER, INTER), HIDDEN)])
def test_merged_projection_equals_separate_ones(fmt, sizes, k):
    from miles.kernels.quant.mx_linear import mx_linear_inference, quantize_weight

    parts = [w.to(torch.bfloat16) for w in _weights(sizes, k)]
    x = _activations(300, k)
    merged = mx_linear_inference(x, quantize_weight(torch.cat(parts), fmt, shard_sizes=sizes))
    separate = torch.cat([mx_linear_inference(x, quantize_weight(w, fmt)) for w in parts], dim=-1)
    assert torch.equal(merged, separate)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("k,n", [(HIDDEN, Q), (Q, HIDDEN), (INTER, HIDDEN)])
def test_trainer_forward_is_bitwise_sglang(fmt, k, n):
    """The trainer holds fp32 master weights and runs under bf16 autocast; SGLang holds the bf16 weights
    that weight sync sends and casts its fp32 activations to bf16."""
    from miles.kernels.quant.mx_linear import MXLinearFunction, mx_linear_inference, quantize_weight

    (w32,) = _weights([n], k)
    x32 = _activations(517, k).float() + 1e-3  # fp32 input, as the true-on-policy norms emit
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        trained = MXLinearFunction.apply(x32[None], w32.requires_grad_(), fmt)[0]
    sampled = mx_linear_inference(x32.to(torch.bfloat16), quantize_weight(w32.detach().to(torch.bfloat16), fmt))
    assert trained.dtype == sampled.dtype == torch.bfloat16
    assert torch.equal(trained, sampled)


@pytest.mark.parametrize("fmt,tol", [("mxfp4", 1e-2), ("nvfp4", 1e-2), ("mxfp8", 1e-2)])
def test_gemm_matches_dequantized_reference(fmt, tol):
    from miles.kernels.quant.mx_linear import dequantize, mx_gemm, quantize_rows, quantize_weight

    (w,) = _weights([Q], HIDDEN)
    x = _activations(256, HIDDEN)
    qx, qw = quantize_rows(x, fmt), quantize_weight(w.to(torch.bfloat16), fmt)
    ref = dequantize(qx, torch.float32) @ dequantize(qw, torch.float32).t()
    out = mx_gemm(qx, qw).float()
    assert ((out - ref).norm() / ref.norm()).item() < tol
    # and the quantization itself is roughly right
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()  # noqa: E731
    expected = {"mxfp4": 0.2, "nvfp4": 0.15, "mxfp8": 0.05}[fmt]
    assert rel(dequantize(qw, torch.float32), w.to(torch.bfloat16).float()) < expected


def test_nvfp4_activation_scale_is_the_row_amax():
    """The fused per-row second-level scale equals PyTorch's amax / (448 * 6), bitwise, zero rows included."""
    from miles.kernels.quant.mx_linear import NVFP4_GLOBAL_DIV, quantize_rows

    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(257, HIDDEN, device="cuda", generator=g)
    x = (x * torch.exp2(torch.randint(-20, 20, x.shape, device="cuda", generator=g))).to(torch.bfloat16)
    x[5] = 0
    expected = x.abs().amax(dim=1).float() / NVFP4_GLOBAL_DIV
    assert torch.equal(quantize_rows(x, "nvfp4").global_scale, expected)


def test_nvfp4_scale_constant_does_not_depend_on_the_default_dtype():
    """SGLang imports the kernels while loading a model, under a bf16 default dtype."""
    import importlib

    import miles.kernels.quant.mx_linear as mx

    expected = float(torch.tensor(1.0, dtype=torch.float32) / mx.NVFP4_GLOBAL_DIV)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        reloaded = importlib.reload(mx)
    finally:
        torch.set_default_dtype(previous)
    assert reloaded._NVFP4_GLOBAL_INV == expected
    importlib.reload(mx)


def test_e2m1_rounds_to_nearest_even():
    from miles.kernels.quant.mx_linear import dequantize, quantize_rows

    # with the block's amax 6 the scale is 1, so these are the e2m1 values and the midpoints between them
    vals = [6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, 0.3, 4.9]
    expected = [6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0, 0.5, 4.0]
    x = torch.zeros(1, 32, device="cuda", dtype=torch.bfloat16)
    x[0, : len(vals)] = torch.tensor(vals)
    got = dequantize(quantize_rows(x, "mxfp4"), torch.float32)[0, : len(vals)]
    assert got.tolist() == expected


@pytest.mark.parametrize("fmt", FORMATS)
def test_backward_is_bf16_gemm_on_dequantized_operands(fmt):
    from miles.kernels.quant.mx_linear import MXLinearFunction, dequantize, quantize_rows, quantize_weight

    (w32,) = _weights([KV], HIDDEN)
    w32.requires_grad_()
    x = _activations(200, HIDDEN).float().requires_grad_()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = MXLinearFunction.apply(x, w32, fmt)
    grad = torch.randn_like(out)
    out.backward(grad)
    g = grad.to(torch.bfloat16)
    qw = dequantize(quantize_weight(w32.detach().to(torch.bfloat16), fmt))
    qx = dequantize(quantize_rows(x.detach().to(torch.bfloat16), fmt))
    assert x.grad.dtype == torch.float32 and w32.grad.dtype == torch.float32
    assert torch.equal(x.grad, (g @ qw).float())
    assert torch.equal(w32.grad, (g.t() @ qx).float())
