import torch
import torch.nn as nn

input=torch.randn(2,3,3)
print("Input:",input)

elu_layer = nn.ELU(alpha=1.0)  # 可以自定义alpha参数，通常为1.0
output = elu_layer(input)
print("ELU Activation Output:")
print(output)

def custom_elu(input_tensor, alpha=1.0):
    return torch.where(input_tensor >= 0, input_tensor, alpha * (torch.exp(input_tensor) - 1))

# 测试自定义 ELU 函数
output_custom = custom_elu(input)
print("Custom ELU Output:")
print(output_custom)

print(input>=0)









