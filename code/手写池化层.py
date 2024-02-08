import torch
import torch.nn as nn
import torch.nn.functional as F

conv_output = torch.tensor([[[[ 0.5914, -0.8443,  0.3207,  0.3029],
          [ 0.6956, -0.2633, -0.2755,  0.0091],
          [ 1.0091,  0.0539, -0.4332,  0.3565],
          [-0.0718, -0.2377,  0.0800,  0.7624]],

         [[-0.2488, -0.2749, -1.1166, -0.2491],
          [ 0.5504,  0.3816,  0.2963,  0.2610],
          [-0.0412, -0.0039, -0.4768, -0.0611],
          [ 0.7517,  0.1665, -0.2231, -0.3370]]],


        [[[-0.2135,  0.4644, -0.2044,  0.5666],
          [-0.0925, -0.2376, -0.2448,  0.6950],
          [-0.0976,  0.7593, -1.6869,  1.1621],
          [ 0.2258,  0.2534, -0.2848, -0.0522]],

         [[-0.0054, -0.7709,  0.0086, -0.3171],
          [ 0.6791,  0.1246, -0.1360,  0.1951],
          [ 0.0818, -0.3583, -0.7911, -1.8213],
          [-0.1488,  0.4026, -0.3277,  0.3289]]]])
'''之前卷积层输出的结果，有两张图片，每张图片有两个通道，每个通道是4x4的矩阵'''

# nn.MaxPool2d: 2D 最大池化层
maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
'''MaxPool2d层是用来执行最大池化操作的，它只是对输入数据进行池化操作，而不涉及任何可学习的参数。因此，在使用MaxPool2d层时，不应该期望像卷积层或线性层那样可以访问权重和偏置。'''

# 池化核大小为2x2，步长为2
output = maxpool_layer(conv_output)
print("MaxPooling Layer Output:")
print(output.size())
print(output)


def manual_maxpool2d(input_tensor, kernel_size=2, stride=2):
    # 提取输入特征图的维度
    batch_size, channels, in_height, in_width = input_tensor.shape

    # 计算输出特征图的维度
    out_height = (in_height - kernel_size) // stride + 1
    out_width = (in_width - kernel_size) // stride + 1

    # 初始化输出特征图
    output = torch.zeros((batch_size, channels, out_height, out_width))

    # 执行最大池化操作
    for i in range(batch_size):
        for j in range(channels):
            for k in range(out_height):
                for l in range(out_width):
                    h_start = k * stride
                    h_end = h_start + kernel_size
                    w_start = l * stride
                    w_end = w_start + kernel_size

                    # 在当前池化窗口中提取最大值
                    window = input_tensor[i, j, h_start:h_end, w_start:w_end]
                    output[i, j, k, l] = torch.max(window)

    return output

# 测试手动实现的最大池化
manual_maxpool_output = manual_maxpool2d(conv_output, kernel_size=2, stride=2)

# 打印结果
print("Manual MaxPooling Output:")
print(manual_maxpool_output.size())
print(manual_maxpool_output)
