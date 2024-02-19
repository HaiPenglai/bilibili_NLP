import hanlp

# 初始化分词器
tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

# 中文分词
text_cn = "今天天气真好，我们一起去公园散步吧。"
tokens_cn = tokenizer(text_cn)
print("中文分词:", tokens_cn)

# 英文分词
text_en = "Today is a good day, let's go to the park for a walk."
tokens_en = tokenizer(text_en)
print("英文分词:", tokens_en)
