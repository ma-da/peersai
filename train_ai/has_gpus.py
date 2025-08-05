import torch

if torch.cuda.is_available():
    print("CUDA-enabled GPU detected!")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("No CUDA-enabled GPU found.")