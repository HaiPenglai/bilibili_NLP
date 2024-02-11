import torch

# 假设的特征和权重
true_weights = torch.tensor([2.0, -3.5])
true_bias = torch.tensor([5.0])

# 创建一些合成数据
x_data = torch.randn(100, 2)  # 100个样本，每个样本2个特征
y_data = x_data @ true_weights + true_bias  # @表示矩阵乘法

# 在y_data中添加一些随机数
random_noise = torch.randn(y_data.shape)   # 添加正态分布的随机数
y_data += random_noise

# 向x_data添加一列1以包括偏置项
X_with_bias = torch.cat([x_data, torch.ones(x_data.shape[0], 1)], dim=1)

# 使用正规方程求解权重和偏置
w_with_bias = torch.inverse(X_with_bias.t() @ X_with_bias) @ X_with_bias.t() @ y_data

# 提取权重和偏置
estimated_weights = w_with_bias[:-1]
estimated_bias = w_with_bias[-1]

# 打印结果
print(f"Estimated weights: {estimated_weights}")
print(f"Estimated bias: {estimated_bias}")
