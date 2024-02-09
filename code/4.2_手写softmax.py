import torch
import torch.nn as nn

input=torch.tensor([[-1,0,1],[0,1,2]]).float()#用整数可能报错，应该用浮点数

softmax_layer = nn.Softmax(dim=1)
# dim=1表示在第二个维度上进行Softmax计算，通常是在多分类问题的输出层使用
output = softmax_layer(input)
print("Softmax Activation Output:")
print(output)

print(nn.Softmax(dim=0)(input))

def custom_softmax(input_tensor, dim=1):
    # 计算指数
    exp_input = torch.exp(input_tensor)
    # 沿着指定维度求和
    sum_exp = torch.sum(exp_input, dim=dim, keepdim=True)
    # 进行softmax计算
    softmax_output = exp_input / sum_exp
    return softmax_output

# 测试自定义softmax函数
output_custom = custom_softmax(input, dim=1)
print("Custom Softmax Output:")
print(output_custom)

output_custom = custom_softmax(input, dim=0)
print("Custom Softmax Output:")
print(output_custom)











