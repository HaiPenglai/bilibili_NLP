import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba

# 加载数据
train_data_path = './data/wxTextClassification/train.news.csv'
train_data = pd.read_csv(train_data_path)

# 定义生成词云的函数
def generate_wordcloud(text, font_path):
    word_list = jieba.lcut(text)
    clean_text = ' '.join(word_list)
    wordcloud = WordCloud(font_path=font_path, width=800, height=800, background_color='white').generate(clean_text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

print(train_data['label'] == 0)#对于每一行，如果label等于0，返回True，否则返回False
print(train_data[train_data['label'] == 0])#对于每一行，如果刚才的结果为True，保留，否则删除
print(train_data[train_data['label'] == 0]['Report Content'])#对上面的结果，只要Report Content这一列，也就是所有真新闻的内容

# 真新闻和假新闻的内容
real_news_content = ' '.join(train_data[train_data['label'] == 0]['Report Content'].dropna())
fake_news_content = ' '.join(train_data[train_data['label'] == 1]['Report Content'].dropna())

# 替换为您的中文字体路径
font_path = './data/msyh.ttc'

# 生成并显示真新闻的词云
print("真新闻词云：")
generate_wordcloud(real_news_content, font_path)

 
