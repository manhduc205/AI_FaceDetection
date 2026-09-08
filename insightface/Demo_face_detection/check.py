import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Current CUDA device:", torch.cuda.current_device() if torch.cuda.is_available() else "No CUDA")
