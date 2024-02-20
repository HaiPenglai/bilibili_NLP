import torch
import torch.nn as nn

# 假设的词汇大小和嵌入维度
vocab_size = 10
embedding_dim = 5

# 随机初始化嵌入矩阵
embedding_matrix = torch.rand(vocab_size, embedding_dim)
print("embedding_matrix",embedding_matrix)
indexs=torch.tensor([[0,1],[2,3]])
print("用矩阵作为索引",embedding_matrix[indexs])
'''矩阵查询的索引可以是一个张量,此时会对这个张量中的每个元素进行查询,结果按照张良的形状拼接起来'''

# 手动实现嵌入查找
def manual_embedding_lookup(indices):
    return embedding_matrix[indices]

# 使用 nn.Embedding
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# 将 nn.Embedding 的权重设置为与手动嵌入相同的值
'''不需要计算梯度'''
with torch.no_grad():
    embedding_layer.weight = nn.Parameter(embedding_matrix)
'''张量赋值给weight就是需要nn.Parameter'''

print("embedding_layer",embedding_layer.weight)

# 生成随机索引
indices = torch.randint(0, vocab_size, (3,2))

print("indices",indices)

# 使用手动嵌入方法
manual_embeds = manual_embedding_lookup(indices)

# 使用 nn.Embedding
nn_embeds = embedding_layer(indices)

# 比较结果
print("Manual Embedding Result:\n", manual_embeds)
print("\nnn.Embedding Result:\n", nn_embeds)
print("\nAre the results equal? ", torch.all(manual_embeds == nn_embeds))
