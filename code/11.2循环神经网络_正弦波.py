import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

# 生成正弦波数据集
np.random.seed(42)
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
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
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
model

# 训练参数
learning_rate = 0.01
epochs = 150

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
losses = []  # 用于记录每个epoch的损失值
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # 清除之前的梯度
    output = model(X)  # 前向传播
    loss = criterion(output, Y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 绘制损失下降图
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 生成测试数据集
x_test = np.linspace(100, 110, 100)
y_test = np.sin(x_test)

# 将测试数据转换为PyTorch张量
X_test = torch.tensor(y_test[:-1], dtype=torch.float32).view(-1, 1, 1)

# 使用模型进行预测
model.eval()  # 确保模型处于评估模式
with torch.no_grad():
    predictions_test = model(X_test).view(-1).numpy()

# 绘制实际值和预测值的对比图
plt.figure(figsize=(10,5))
plt.plot(x_test[:-1], y_test[:-1], label='Actual Sin wave', color='blue')
plt.plot(x_test[:-1], predictions_test, label='Predicted Sin wave', color='red', linestyle='--')
plt.title('Sin wave prediction on test data (x in [100, 110])')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.show()

