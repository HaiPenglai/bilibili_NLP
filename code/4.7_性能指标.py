# 定义给定的值
TP = 8  # 预测标签为1且实际标签为1
FP = 2  # 预测标签为1且实际标签为0
FN = 3  # 预测标签为0且实际标签为1
TN = 7  # 预测标签为0且实际标签为0

# 计算标签1上的精确率(Precision)，召回率(Recall)和F1分数
precision_1 = TP / (TP + FP)
recall_1 = TP / (TP + FN)
F1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)

# 计算标签0上的精确率，召回率和F1分数
precision_0 = TN / (TN + FN)
recall_0 = TN / (TN + FP)
F1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)

# 计算准确率
accuracy = (TP + TN) / (TP + TN + FP + FN)

print(
{
    "precision_1": precision_1,
    "recall_1": recall_1,
    "F1_1": F1_1,
    "precision_0": precision_0,
    "recall_0": recall_0,
    "F1_0": F1_0,
    "accuracy": accuracy
}
)