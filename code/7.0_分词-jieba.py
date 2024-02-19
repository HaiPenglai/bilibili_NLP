import jieba
jieba.load_userdict("./data/Tokenization/userdict.txt")

seg_list = jieba.cut("jieba是一个非常流行的中文分词库广泛用于中文自然语言处理", cut_all=False)
print("精确模式: " + "/ ".join(seg_list))

seg_list = jieba.cut("jieba是一个非常流行的中文分词库广泛用于中文自然语言处理", cut_all=True)
print("全模式: " + "/ ".join(seg_list))

seg_list = jieba.cut_for_search("jieba是一个非常流行的中文分词库广泛用于中文自然语言处理")
print("搜索引擎模式: " + "/ ".join(seg_list))

seg_list = jieba.cut("jieba是壹個非常流行的中文分詞庫廣泛用于中文自然語言處理", cut_all=False)
print("精确模式: " + "/ ".join(seg_list))
