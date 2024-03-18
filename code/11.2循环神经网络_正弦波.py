import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 生成正弦波数据集
torch.manual_seed(44)  # 设置随机种子，保证每次运行结果一致
x = np.linspace(0, 100, 1000)
y = np.sin(x)
print("shape of x:",x.shape)
print("x",x[:20])
print("shape of y:",y.shape)
print("y",y[:20])

# 将数据转换为PyTorch张量
X = torch.tensor(y[:-1], dtype=torch.float32).view(-1, 1, 1)
Y = torch.tensor(y[1:], dtype=torch.float32).view(-1, 1, 1)
print("shape of X:",X.shape)
print("X",X[:20])
print("shape of Y:",Y.shape)
print("Y",Y[:20])

# 可视化部分数据集
plt.figure(figsize=(10,5))
plt.plot(x[:100], y[:100], label='Sin wave')
plt.title('Sin wave data')
plt.legend()
plt.show()

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x


# 模型参数
input_size = 1
hidden_size = 100
output_size = 1

# 实例化模型
model = SimpleRNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# 训练参数
learning_rate = 0.01 # 学习率
epochs = 150 # 训练轮数

# 损失函数和优化器
criterion = nn.MSELoss() # 均方误差，计算方法：(y_true - y_pred) ** 2 求和取平均
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam优化器

# 训练模型
losses = []  # 用于记录每个epoch的损失值
for epoch in range(epochs):
    model.train() # 确保模型处于训练模式，因为PyTorch中有一些层在训练和评估模式下行为不同
    optimizer.zero_grad()  # 清除之前的梯度，否则梯度将累加到一起
    output = model(X)  # 前向传播，得到的output的shape为(999, 1, 1)
    if epoch == 0:
        print("shape of output:",output.shape)
    loss = criterion(output, Y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数，主要工作之一确实是梯度下降

    losses.append(loss.item()) # 记录损失值
    if epoch % 10 == 0: # 每10个epoch打印一次损失值
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 绘制损失下降图
plt.figure(figsize=(10, 5)) # 设置画布大小
plt.plot(losses, label='Training Loss') # 绘制损失值曲线
plt.title('Training Loss') # 设置图标题
plt.xlabel('Epoch') # 设置x轴标签
plt.ylabel('Loss') # 设置y轴标签
plt.legend() # 显示图例
plt.show() # 显示图像

# 生成测试数据集
x_test = np.linspace(100, 110, 100)# 生成100个点,从100到110之间
y_test = np.sin(x_test)# 生成对应的sin值

# 将测试数据转换为PyTorch张量
X_test = torch.tensor(y_test[:-1], dtype=torch.float32).view(-1, 1, 1)
# 从测试数据中取出前999个点，转换为PyTorch张量，形状为(999, 1, 1)

# 使用模型进行预测
model.eval()  # 确保模型处于评估模式
with torch.no_grad():
    predictions_test = model(X_test).view(-1).numpy()# 使用模型进行预测，得到的预测值的shape为(999, 1, 1)，需要将其转换为一维数组

# 绘制实际值和预测值的对比图
plt.figure(figsize=(10,5))# 设置画布大小
plt.plot(x_test[:-1], y_test[1:], label='Actual Sin wave', color='blue')# 绘制实际值
plt.plot(x_test[:-1], predictions_test, label='Predicted Sin wave', color='red', linestyle='--')# 绘制预测值
plt.title('Sin wave prediction on test data (x in [100, 110])')# 设置图标题
plt.xlabel('x')# 设置x轴标签
plt.ylabel('sin(x)')# 设置y轴标签
plt.legend()# 显示图例
plt.show()# 显示图像

