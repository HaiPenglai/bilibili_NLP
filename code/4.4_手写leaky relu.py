import torch
import torch.nn as nn

input=torch.randn(2,3,3)
print("Input:",input)


leaky_relu_layer = nn.LeakyReLU(negative_slope=0.01)  # 可以自定义负斜率，通常为小于1的正数
output = leaky_relu_layer(input)
print("Leaky ReLU Activation Output:")
print(output)


def custom_leaky_relu(input_tensor, negative_slope=0.01):
    # 对于每个元素，如果大于等于0，保持不变；如果小于0，乘以负斜率
    leaky_relu_output = torch.where(input_tensor >= 0, input_tensor, input_tensor * negative_slope)
    return leaky_relu_output

# 测试自定义Leaky ReLU函数
output_custom = custom_leaky_relu(input)
print("Custom Leaky ReLU Output:")
print(output_custom)

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
condition = torch.tensor([True, False, True])

result = torch.where(condition, x, y)
print(result)  # 输出: tensor([1, 5, 3])








