import torch

# 사용할 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# CPU 텐서 생성
cpu_tensor = torch.randn(3, 3)
print("CPU Tensor:", cpu_tensor)

# 텐서를 GPU로 이동
gpu_tensor = cpu_tensor.to(device)
print("GPU Tensor:", gpu_tensor)