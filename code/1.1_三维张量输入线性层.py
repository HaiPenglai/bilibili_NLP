import torch
import torch.nn as nn

# 创建输入数据矩阵 input_data
input_data = torch.tensor([[[1, 2, 3, 1],
                           [0, 1, 2, 0]],

                           [[1, 2, 3, 1],
                           [0, 1, 2, 0]]], dtype=torch.float)  # 注意：输入数据需要是float类型，因为要进行权重赋值

# 创建权重矩阵 weight
weight = torch.tensor([[1, 0, 1, 2],
                       [2, 1, 1, 0],
                       [0, 2, 2, 1]], dtype=torch.float)  # 注意：权重需要是float类型

# 创建偏置项向量 bias
bias = torch.tensor([1, 2, 3], dtype=torch.float)  # 注意：偏置需要是float类型

# 定义一个权重相同的全连接层
linear_layer = nn.Linear(4, 3, bias=True)

# 将自定义的权重和偏置赋值给线性层
with torch.no_grad():  # 不跟踪这些操作的梯度，否则报错
    linear_layer.weight.copy_(weight)  # 使用copy_方法
    linear_layer.bias.copy_(bias)

# 使用自定义权重和偏置的线性层进行前向传播计算
output = linear_layer(input_data)

print("Linear Layer Output:")
print(output)

# 验证自定义权重和偏置的正确性
expected_output = input_data @ weight.t() + bias
print("Expected Output (Manual Calculation):")
print(expected_output)
