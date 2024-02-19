import hanlp
tagger = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ALBERT_BASE)
text = "汉克斯出生于加州的康科德市，他的父亲是厨师，母亲是医院工作者。"
pos_tags = tagger(text)
print(pos_tags)
# Output: [('汉克斯', 'NR'), ('出生', 'VV'), ('于', 'P'), ('加州', 'NR'), ('的', 'DEG'), ('康科德', 'NR'), ('市', 'NN'), ('，', 'PU'), ('他', 'PN'), ('的', 'DEG'), ('父亲', 'NN'), ('是', 'VC'), ('厨师', 'NN'), ('，', 'PU'), ('母亲', 'NN'), ('是', 'VC'), ('医院', 'NN'), ('工作者', 'NN'), ('。', 'PU')]