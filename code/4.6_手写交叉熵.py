import torch

# 假设我们有一个三类分类问题的 logits
logits = torch.tensor([[2.0, 1.0]])#可能不止一个样本，所以是二维的

# 计算 Softmax
softmax_probs = torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True)
print("Softmax (manual, unstable):", softmax_probs)

# 计算 Log Softmax
log_softmax_manual = torch.log(softmax_probs)

print("Log Softmax (manual, unstable):", log_softmax_manual)
print("Log Softmax (PyTorch):", torch.log_softmax(logits, dim=1))
'''Log Softmax (PyTorch): tensor([[-0.4170, -1.4170, -2.3170]])'''

# 计算 Cross Entropy
# 假设目标标签是第二类
print(-log_softmax_manual[0, 1])
