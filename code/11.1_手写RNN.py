import torch
import torch.nn as nn

torch.manual_seed(42)
rnn = nn.RNN(input_size=3, hidden_size=10,num_layers=1, batch_first=True)
input_data= torch.randn(1,5,3)
output, hidden = rnn(input_data)
print(input_data)
print(input_data.shape,output.shape,hidden.shape)

W_xh = rnn.weight_ih_l0.data
W_hh = rnn.weight_hh_l0.data
b_h = rnn.bias_ih_l0.data + rnn.bias_hh_l0.data
print("shape of W_xh, W_hh, b_h",W_xh.shape, W_hh.shape, b_h.shape)

def manual_rnn_forward(X, W_xh, W_hh, b_h):
    # 由于输入是单个批次，我们直接取第一个批次的数据
    seq_len, input_size = X.shape
    hidden_size = W_hh.shape[0]
    h_prev = torch.zeros(hidden_size)
    outputs = []

    for t in range(seq_len):
        x_t = X[t, :]  # 获取当前时间步的输入
        # 按照原始公式顺序进行计算
        h_t = torch.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
        outputs.append(h_t.unsqueeze(0))
        h_prev = h_t

    print("return:",len(outputs),outputs[0].shape,torch.cat(outputs, dim=0).shape, h_t.unsqueeze(0).shape)
    return torch.cat(outputs, dim=0), h_t.unsqueeze(0)

# 获取单个批次的输入数据
input_data = input_data[0]  # 从批次中取出单个序列

# 使用修改后的函数进行前向传播计算
manual_output, manual_hidden = manual_rnn_forward(input_data, W_xh, W_hh, b_h)

# 比较结果
print("Modified manual implementation (single batch):")
print("Output shape:", manual_output.shape)
print("Hidden state shape:", manual_hidden.shape)

# 验证输出与隐藏状态是否接近PyTorch的实现
output = output[0]  # 从PyTorch的输出中取出对应单个批次的输出
hidden = hidden[0]  # 对于单层RNN，取出对应单个批次的最后一个隐藏状态

print("Output close:", torch.allclose(manual_output, output, atol=1e-4))
print("Hidden state close:", torch.allclose(manual_hidden, hidden, atol=1e-4))













