import torch
import torch.nn as nn

input=torch.randn(2,3,3)
print("Input:",input)

relu_layer = nn.ReLU()
output = relu_layer(input)
print("ReLU Activation Output:")
print(output)

# 手动实现ReLU激活函数
def custom_relu(input_tensor):
    # 大于0的元素保持不变，小于0的元素置为0
    return input_tensor * (input_tensor > 0).float()

# 调用手动实现的ReLU激活函数
output_custom = custom_relu(input)
print("Custom ReLU Activation Output:")
print(output_custom)

input_tensor = torch.tensor([[1, -2, 3], [0, 4, -5]])
print(input_tensor>0)