import pandas as pd
import jieba

# 加载数据
train_data_path = './data/wxTextClassification/train.news.csv'
test_data_path = './data/wxTextClassification/test.news.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# 选择需要分词的列，例如：'Report Content'
train_texts = train_data['Report Content'].tolist()
test_texts = test_data['Report Content'].tolist()

print(type(train_texts))
print(train_texts[:5])

a=[1,2,3,4,5]
print(type(a))

# 分词函数
def segment_words(texts):
    word_set = set()
    for text in texts:
        if isinstance(text, str):  # 确保文本是字符串
            words = jieba.lcut(text)
            word_set.update(words)
    return word_set

# 统计训练集和测试集中的不同词语
train_words = segment_words(train_texts)
test_words = segment_words(test_texts)

# 输出不同词语的数量
print("Number of unique words in training set:", len(train_words))
print("Number of unique words in testing set:", len(test_words))
