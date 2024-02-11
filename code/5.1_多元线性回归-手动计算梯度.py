import torch

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

# 重新初始化权重和偏置
weights = torch.randn(2, requires_grad=False)
bias = torch.randn(1, requires_grad=False)

# 损失函数 - 均方误差
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

# 设置学习率
learning_rate = 0.01

# 迭代次数
iterations = 1000

# 执行手动梯度计算和更新
for _ in range(iterations):
    # 计算预测值
    y_pred = x_data @ weights + bias

    # 计算损失
    loss = mse_loss(y_pred, y_data)

    # 手动计算梯度
    grad_w = (2.0 / x_data.shape[0]) * (x_data.t() @ (y_pred - y_data))
    grad_b = (2.0 / x_data.shape[0]) * torch.sum(y_pred - y_data)

    # 更新权重和偏置
    weights -= learning_rate * grad_w
    bias -= learning_rate * grad_b

# 打印结果
print(f"Estimated weights: {weights}")
print(f"Estimated bias: {bias}")



