import fasttext

# 定义训练文件的路径
training_file = './data/wikitext.txt'  # 请将这里的路径替换成您文件的实际路径

# 训练模型
model = fasttext.train_unsupervised(training_file, model='skipgram')

# 保存模型
model.save_model('./data/word2vec_skipgram_model.bin')

# 加载模型
model = fasttext.load_model('./data/word2vec_skipgram_model.bin')

# 获取单词的词向量
word = "example"  # 替换为您感兴趣的单词
word_vector = model.get_word_vector(word)
print(f"词向量（{word}）: {word_vector}")

# 找出与特定单词最接近的单词
nearest_neighbors = model.get_nearest_neighbors(word,5) # k表示返回的最近邻单词的数量
print(f"与单词（{word}）最接近的单词及其相似度:")
for neighbor in nearest_neighbors:
    similar_word, similarity = neighbor
    print(f"{similar_word}, 相似度: {similarity}")