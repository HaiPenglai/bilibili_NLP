import torch
import torch.nn as nn

# 定义输入数据
input_data = torch.randn(2, 4)  # 2个样本，4个特征
print('Input Data:', input_data)
'''tensor([[-0.0651,  0.1765, -0.5925, -0.6215],
        [-0.4160,  0.4052,  0.6214,  0.0385]])'''

# nn.Linear: 全连接层/线性层
linear_layer = nn.Linear(4, 3)  # 输入特征维度为4，输出特征维度为3
print('Linear Layer:',linear_layer.weight.data, linear_layer.bias.data)
'''tensor([[-0.3712,  0.1380,  0.1242, -0.1603],
        [-0.0283,  0.3212, -0.4816,  0.2166],
        [-0.1501, -0.2759,  0.1762, -0.0754]])'''
'''tensor([-0.2100, -0.4055, -0.2082])'''

output = linear_layer(input_data)
print("Linear Layer Output:")
print(output)
print(input_data@linear_layer.weight.t()+linear_layer.bias)
print()
'''tensor([[-0.1355, -0.1963, -0.3046],
        [ 0.0712, -0.5546, -0.1509]], grad_fn=<AddmmBackward0>)'''
'''tensor([[-0.1355, -0.1963, -0.3046],
        [ 0.0712, -0.5546, -0.1509]], grad_fn=<AddBackward0>)'''