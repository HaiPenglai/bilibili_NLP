import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建输入图像
input_image = torch.randn(2, 3, 4, 4)  # 两张4*4的RGB图片
print("Input Image:", input_image)

# 定义卷积层
conv_layer = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
print("Conv Layer:", conv_layer.weight.data.size() )
print(conv_layer.weight.data)
print(conv_layer.bias.data)

conv_output = conv_layer(input_image)
print("nn.Conv2d Output:")
print(conv_output.size())
print(conv_output)


# 提取权重和偏差
weights = conv_layer.weight.data
bias = conv_layer.bias.data

# 手动实现卷积的函数
def manual_conv2d(input_img, weights, bias, stride=1, padding=1):
    # 添加填充
    input_padded = F.pad(input_img, (padding, padding, padding, padding), mode='constant', value=0)

    # 提取输入和权重的维度
    batch_size, in_channels, in_height, in_width = input_padded.shape
    out_channels, _, kernel_height, kernel_width = weights.shape

    # 计算输出维度
    out_height = (in_height - kernel_height) // stride + 1
    out_width = (in_width - kernel_width) // stride + 1

    # 初始化输出
    output = torch.zeros((batch_size, out_channels, out_height, out_width))

    # 执行卷积操作
    for i in range(batch_size):
        for j in range(out_channels):
            for k in range(out_height):
                for l in range(out_width):
                    h_start = k * stride
                    h_end = h_start + kernel_height
                    w_start = l * stride
                    w_end = w_start + kernel_width
                    output[i, j, k, l] = torch.sum(input_padded[i, :, h_start:h_end, w_start:w_end] * weights[j]) + bias[j]

    return output

# 应用手动卷积
manual_conv_output = manual_conv2d(input_image, weights, bias, stride=1, padding=1)

# 打印结果
print("Manual Conv2d Output:")
print(manual_conv_output.size())
print(manual_conv_output)