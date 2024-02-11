import torch
import torch.nn as nn
from torch.optim import SGD

# 假设的特征和权重
true_weights = torch.tensor([2.0, -3.5])
true_bias = torch.tensor([5.0])

# 创建一些合成数据
x_data = torch.randn(100, 2)  # 100个样本，每个样本2个特征
print('x_data',x_data)
y_data = x_data @ true_weights + true_bias  # @表示矩阵乘法
print(y_data)

# 在y_data中添加一些随机数
random_noise = torch.randn(y_data.shape)   # 添加正态分布的随机数
y_data += random_noise

# 初始化权重和偏置
weights = torch.randn(2, requires_grad=True)
bias = torch.randn(1, requires_grad=True)

# 设置学习率
learning_rate = 0.01

# 迭代次数
iterations = 1000

# 损失函数 - 均方误差
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

# 执行梯度下降
for _ in range(iterations):
    # 计算预测值
    y_pred = x_data @ weights + bias

    # 计算损失
    loss = mse_loss(y_pred, y_data)

    # 计算梯度
    loss.backward()

    # 更新权重和偏置，使用 torch.no_grad() 来暂停梯度追踪
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        bias -= learning_rate * bias.grad

        # 清零梯度
        weights.grad.zero_()
        bias.grad.zero_()

# 打印结果
print(f"Estimated weights: {weights}")
print(f"Estimated bias: {bias}")


