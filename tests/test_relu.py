import torch
from csrc.relu.relu_naive import relu_naive

x = torch.randn(1024, device="cuda")

torch_res = torch.relu(x)
custom_res = 

# compare PyTorch with my kernel for accuracy
torch.allclose()
