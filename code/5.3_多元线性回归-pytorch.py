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

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        # 定义模型的层
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # 前向传播函数-
        return self.linear(x)

# 实例化模型
model = LinearRegressionModel(input_size=2)

# 均方误差损失函数
loss_function = nn.MSELoss()

# 随机梯度下降优化器
optimizer = SGD(model.parameters(), lr=0.01)

# 训练模型
epochs = 1000  # 训练轮数
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清空过往梯度

    y_pred = model(x_data)  # 进行预测
    # print(y_pred == x_data @ model.linear.weight.t() +model.linear.bias )

    loss = loss_function(y_pred, y_data.unsqueeze(1))  # 计算损失

    loss.backward()  # 反向传播，计算当前梯度
    optimizer.step()  # 根据梯度更新网络参数

    # 每隔一段时间输出训练信息
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


print("模型参数:", model.linear.weight.data, model.linear.bias.data)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设模型已经训练完毕，并且我们有 model.linear.weight 和 model.linear.bias

# 创建一个新的图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据点
ax.scatter(x_data[:, 0].numpy(), x_data[:, 1].numpy(), y_data.numpy())

# 为了绘制平面，我们需要创建一个网格并计算相应的y值
x1_grid, x2_grid = torch.meshgrid(torch.linspace(-3, 3, 10), torch.linspace(-3, 3, 10))
y_grid = model.linear.weight[0, 0].item() * x1_grid + model.linear.weight[0, 1].item() * x2_grid + model.linear.bias.item()

# 绘制预测平面
ax.plot_surface(x1_grid.numpy(), x2_grid.numpy(), y_grid.numpy(), alpha=0.5)

# 设置坐标轴标签
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

# 显示图形
plt.show()











