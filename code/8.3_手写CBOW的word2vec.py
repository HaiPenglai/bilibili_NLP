import torch
import torch.nn as nn
import torch.optim as optim

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).mean(dim=0)
        out = self.linear(embeds)
        log_probs = torch.log_softmax(out, dim=0)
        return log_probs

word_to_ix = {"A": 0, "dog": 1, "barks": 2, "at": 3, "night": 4}
ix_to_word = {ix: word for word, ix in word_to_ix.items()}
vocab_size = len(word_to_ix)

print("word_to_ix",word_to_ix)
print("ix_to_word",ix_to_word)
print("vocab_size",vocab_size)

data = [
    (torch.tensor([word_to_ix["A"], word_to_ix["barks"]]), torch.tensor(word_to_ix["dog"])),
    (torch.tensor([word_to_ix["dog"], word_to_ix["at"]]), torch.tensor(word_to_ix["barks"])),
    (torch.tensor([word_to_ix["barks"], word_to_ix["night"]]), torch.tensor(word_to_ix["at"]))
]
print(data)

# 设置超参数
embedding_dim = 10

# 实例化模型
model = CBOW(vocab_size, embedding_dim)

# 定义损失函数和优化器
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(100):
    total_loss = 0
    for context, target in data:
        if epoch == 99:
            print(context, target)
        # 步骤 1. 准备数据
        context_idxs = context

        # 步骤 2. 运行模型的前向传递
        log_probs = model(context_idxs)
        if epoch == 99:
            print(log_probs)

        # 步骤 3. 计算损失
        loss = loss_function(log_probs.view(1, -1), target.view(1))
        '''view(1, -1) 将 log_probs 调整为一个形状为 (1, n) 的张量，其中 n 是 log_probs 中原始元素的数量，而 view(1) 将 target 调整为一个形状为 (1,) 的张量'''
        # 步骤 4. 反向传播并更新梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss}")

print(model(data[0][0]))

# 测试数据
test_context = torch.tensor([word_to_ix["A"], word_to_ix["barks"]])

# 使用模型进行预测
with torch.no_grad():
    log_probs = model(test_context)

# 获取概率最高的单词索引
predicted_word_idx = torch.argmax(log_probs).item()

# 将索引转换回单词
predicted_word = ix_to_word[predicted_word_idx]

print(f"Input context: ['A', 'barks']")
print(f"Predicted word: '{predicted_word}'")











