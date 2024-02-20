import torch
import torch.nn as nn
import torch.optim as optim

word_to_ix = {"A": 0, "dog": 1, "barks": 2, "at": 3, "night": 4}
ix_to_word = {ix: word for word, ix in word_to_ix.items()}
vocab_size = len(word_to_ix)

# 设置超参数
embedding_dim = 10

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, word):
        embed = self.embeddings(word)
        out = self.linear(embed)
        log_probs = torch.log_softmax(out, dim=0)
        return log_probs

skipgram_data = [
    (torch.tensor(word_to_ix["dog"]), torch.tensor([word_to_ix["A"], word_to_ix["barks"]])),
    (torch.tensor(word_to_ix["barks"]), torch.tensor([word_to_ix["dog"], word_to_ix["at"]])),
    (torch.tensor(word_to_ix["at"]), torch.tensor([word_to_ix["barks"], word_to_ix["night"]]))
]

# 实例化模型
skipgram_model = SkipGram(vocab_size, embedding_dim)

# 同样的优化器和损失函数
optimizer = optim.SGD(skipgram_model.parameters(), lr=0.001)
loss_function = nn.NLLLoss()

# 训练模型
for epoch in range(100):
    total_loss = 0
    for center_word, context_words in skipgram_data:
        for context_word in context_words:
            log_probs = skipgram_model(center_word)
            loss = loss_function(log_probs, context_word)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss}")

# 测试数据
test_word = torch.tensor([word_to_ix["dog"]])

# 使用模型进行预测
with torch.no_grad():
    log_probs = skipgram_model(test_word)

# 获取概率最高的单词索引
predicted_indices = torch.topk(log_probs, 2).indices

predicted_words = [ix_to_word[idx.item()] for idx in predicted_indices[0]]
print(f"Input word: 'dog'")
print(f"Predicted context words: {predicted_words}")


