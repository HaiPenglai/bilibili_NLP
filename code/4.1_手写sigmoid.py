import torch
import torch.nn as nn

input=torch.randn(2,2)
print("Input:",input)
print(input)

sigmoid_layer = nn.Sigmoid()
output = sigmoid_layer(input)
print("Sigmoid Activation Output:")
print(output)

# 手动实现Sigmoid激活函数
def custom_sigmoid(input_tensor):
    return 1 / (1 + torch.exp(-input_tensor))

# 调用手动实现的Sigmoid激活函数
output_custom = custom_sigmoid(input)
print("Custom Sigmoid Activation Output:")
print(output_custom)

input_tensor = torch.tensor([[-100,0], [1,2]])
print(torch.exp(input_tensor))
