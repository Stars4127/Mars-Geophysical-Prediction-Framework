import torch

# Detect device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"PyTorch device: {device}")

# Create a small test tensor
x = torch.randn(1000, 1000)

if device != "cpu":
    x = x.to(device)
    print(f"Tensor moved to {device} successfully.")

# Simple operation to test GPU compute
y = x @ x.T  # matrix multiplication
print("Computation done. Tensor shape:", y.shape)
print("y device:", y.device)
