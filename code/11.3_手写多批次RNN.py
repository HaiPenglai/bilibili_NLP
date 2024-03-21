import torch
import torch.nn as nn

torch.manual_seed(42)
rnn = nn.RNN(input_size=3, hidden_size=10,num_layers=1, batch_first=True)
input_data= torch.randn(2,5,3)
output, hidden = rnn(input_data)
print(input_data)
print(input_data.shape,output.shape,hidden.shape)

W_xh = rnn.weight_ih_l0.data
W_hh = rnn.weight_hh_l0.data
b_h = rnn.bias_ih_l0.data + rnn.bias_hh_l0.data
print("shape of W_xh, W_hh, b_h",W_xh.shape, W_hh.shape, b_h.shape)

def manual_rnn_forward(X, W_xh, W_hh, b_h):
    batch_size, seq_len, input_size = X.shape
    hidden_size = W_hh.shape[0]
    h_prev = torch.zeros(batch_size, hidden_size)
    outputs = []

    for t in range(seq_len):
        x_t = X[:, t, :]
        h_t = torch.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b_h)
        outputs.append(h_t.unsqueeze(1))
        h_prev = h_t

    return torch.cat(outputs, dim=1), h_t.unsqueeze(0)

# 手动实现的RNN前向传播
manual_output, manual_hidden = manual_rnn_forward(input_data, W_xh, W_hh, b_h)

# 比较手动实现和PyTorch实现的结果
print("Output close:", torch.allclose(manual_output, output, atol=1e-4))
print("Hidden state close:", torch.allclose(manual_hidden, hidden, atol=1e-4))
