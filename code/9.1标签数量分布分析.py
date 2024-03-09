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

import matplotlib.pyplot as plt
import numpy as np

# 绘制训练集和测试集的标签分布柱状图
train_label_counts = train_data['label'].value_counts()
test_label_counts = test_data['label'].value_counts()

# 创建一个图和子图
fig, ax = plt.subplots()

# 柱状图的数据
labels = ['Real News', 'Fake News']
train_counts = [train_label_counts[0], train_label_counts[1]]
test_counts = [test_label_counts[0], test_label_counts[1]]

# 设置柱状图的位置和宽度
x = np.arange(len(labels))  # 标签位置
width = 0.35  # 柱状图的宽度

# 绘制柱状图
rects1 = ax.bar(x - width/2, train_counts, width, label='Train')
rects2 = ax.bar(x + width/2, test_counts, width, label='Test')

# 添加一些文本用于标签、标题和自定义x轴刻度标签等
ax.set_ylabel('Counts')
ax.set_title('Label distribution in Training and Testing Sets')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 为每个条形图添加一个文本标签
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# 显示图形
plt.show()
