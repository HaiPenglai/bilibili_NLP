import hanlp
recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)

text = "汉克斯出生于加州的康科德市，他的父亲是厨师，母亲是医院工作者。"
entities = recognizer(text)
print(entities)