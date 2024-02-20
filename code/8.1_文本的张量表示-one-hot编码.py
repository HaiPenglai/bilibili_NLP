from sklearn.preprocessing import OneHotEncoder

# 创建词汇表
vocab = [["apple"], ["banana"], ["cherry"]]

# 初始化 OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# 对词汇表进行独热编码
one_hot_encoded = encoder.fit_transform(vocab)
# 打印结果
print("One-Hot Encoded Vocab:")
print(one_hot_encoded)

# 对新的词进行编码
new_word = [["banana"],['apple']]
new_word_encoded = encoder.transform(new_word)

print("\nEncoded New Word (banana):")
print(new_word_encoded)

