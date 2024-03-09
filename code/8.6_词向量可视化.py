import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 生成随机词向量
num_words = 100  # 词汇量大小
embedding_dim = 50  # 词向量维度
word_embeddings = torch.randn(num_words, embedding_dim)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=0)
word_embeddings_2d = tsne.fit_transform(word_embeddings)

# 可视化
plt.figure(figsize=(10, 10))
for i in range(num_words):
    plt.scatter(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1])
    plt.annotate(f'word_{i}', xy=(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')
plt.show()