import torch
import torch.nn.functional as F


# 创建一个Conv1d层并提取其权重和偏置
conv1d_layer = torch.nn.Conv1d(in_channels=300, out_channels=64, kernel_size=3, stride=1, padding=1)
conv1d_weight = conv1d_layer.weight.data
conv1d_bias = conv1d_layer.bias.data


# 创建一个输入张量
input_tensor = torch.randn(2, 300, 100)  # 假设有1个样本，每个样本有300个通道，宽度为100

# 使用PyTorch的Conv1d
output_conv1d = conv1d_layer(input_tensor)

#获取权重和偏置
weights = conv1d_layer.weight.data
bias = conv1d_layer.bias.data

# 手动实现卷积的函数
def manual_conv1d(input_tensor, weights, bias, stride=1, padding=1):
    batch_size, in_channels, width = input_tensor.shape
    out_channels, _, kernel_size = weights.shape

    # 计算输出宽度
    output_width = ((width + 2 * padding - kernel_size) // stride) + 1

    # 应用padding
    if padding > 0:
        input_padded = F.pad(input_tensor, (padding, padding), "constant", 0)
    else:
        input_padded = input_tensor

    # 初始化输出张量
    output = torch.zeros(batch_size, out_channels, output_width)

    # 执行卷积操作
    for i in range(out_channels):
        for j in range(output_width):
            start = j * stride
            end = start + kernel_size
            # 对所有输入通道执行卷积并求和
            output[:, i, j] = torch.sum(input_padded[:, :, start:end] * weights[i, :, :].unsqueeze(0), dim=(1, 2)) + \
                              bias[i]

    return output

print("output_conv1d:", output_conv1d)

# 应用手动卷积
manual_conv1d_output = manual_conv1d(input_tensor, weights, bias, stride=1, padding=1)
print("manual_conv1d_output:", manual_conv1d_output)

# 比较结果
print("Output close:", torch.allclose(output_conv1d, manual_conv1d_output, atol=1e-4))
