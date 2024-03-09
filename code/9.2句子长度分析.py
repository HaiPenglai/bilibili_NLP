import pandas as pd
import matplotlib.pyplot as plt

# 加载数据集
train_data_path = './data/wxTextClassification/train.news.csv'
train_data = pd.read_csv(train_data_path)

# 计算每个标题的长度
train_data['Title_Length'] = train_data['Title'].apply(len)
print(train_data['Title_Length'])

# 绘制标题长度的柱状图
plt.figure(figsize=(10, 6))
plt.hist(train_data['Title_Length'], bins=range(0, max(train_data['Title_Length']) + 10, 10), edgecolor='black')
plt.title('Distribution of Title Lengths in Training Set')
plt.xlabel('Title Length')
plt.ylabel('Number of Samples')
plt.xticks(range(0, max(train_data['Title_Length']) + 10, 10))
plt.show()










