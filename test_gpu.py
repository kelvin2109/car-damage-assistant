import torch
import nn

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}") 
    