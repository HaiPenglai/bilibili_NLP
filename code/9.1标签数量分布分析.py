import pandas as pd

# Load the datasets
train_data_path = './data/wxTextClassification/train.news.csv'
test_data_path = './data/wxTextClassification/test.news.csv'

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

print(test_data)
print(test_data['label'])
print(test_data.columns)

print("the distribution of the labels")
print("训练集标签分布")
print(train_data['label'].value_counts())
print(len(train_data))
print("测试集标签分布")
print(test_data['label'].value_counts())
print(len(test_data))

