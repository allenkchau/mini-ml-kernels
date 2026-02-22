import pytest
import torch
import numpy as np

from kernels.relu import relu_naive

def reference_relu(x: torch.Tensor) -> torch.Tensor:


def test_relu_correctness():


def reference_layernorm(x, gamma, beta, eps=1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    sigma = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(sigma + eps) * gamma + beta

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("batch, hidden", [(32, 512), (64, 1024), (16, 256)])
def test_layernorm_correctness(batch, hidden, dtype):
    torch.manual_seed(0)

    # create random inputs
    x = torch.randn(batch, hidden, dtype=dtype, device="cuda")
    gamma = torch.randn(hidden, dtype=dtype, device="cuda")
    beta = torch.randn(hidden, dtype=dtype, device="cuda")

    # PyTorch reference
    ref_out = reference_layernorm(x, gamma, beta)

    # your kernel
    out = layernorm_forward(x, gamma, beta)

    # compare
    assert torch.allclose(out, ref_out, rtol=1e-3, atol=1e-3), \
        "Mismatch with PyTorch reference"
