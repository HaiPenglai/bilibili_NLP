import jieba.posseg as pseg

text = "词性标注是指为文本中的每个词分配一个词性"
words = pseg.cut(text)

for word, flag in words:
    print(f'{word}/{flag}', end=' ')