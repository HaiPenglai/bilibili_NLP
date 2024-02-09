import torch
import torch.nn as nn

input=torch.randn(2,3,3)
print("Input:",input)

tanh_layer = nn.Tanh()
output = tanh_layer(input)
print("Tanh Activation Output:")
print(output)

def custom_tanh(input_tensor):
    # 计算tanh函数
    tanh_output = (torch.exp(input_tensor) - torch.exp(-input_tensor)) / (torch.exp(input_tensor) + torch.exp(-input_tensor))
    return tanh_output

# 测试自定义tanh函数
output_custom = custom_tanh(input)
print("Custom Tanh Output:")
print(output_custom)









