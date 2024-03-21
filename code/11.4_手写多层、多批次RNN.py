import torch
import torch.nn as nn

torch.manual_seed(42)

num_layers = 3
rnn = nn.RNN(input_size=3, hidden_size=10, num_layers=num_layers, batch_first=True)

input_data = torch.randn(2, 5, 3)  # Batch size = 2, Sequence length = 5, Input size = 3

def manual_rnn_forward(X, rnn):
    batch_size, seq_len, _ = X.shape
    hidden_size = rnn.hidden_size
    num_layers = rnn.num_layers

    # 初始化隐藏状态
    h_prev = [torch.zeros(batch_size, hidden_size) for _ in range(num_layers)]

    # 存储每一层的最终输出
    layer_outputs = []

    # 对于每一层
    for layer in range(num_layers):
        layer_input = X if layer == 0 else layer_outputs[-1]
        W_xh = getattr(rnn, f'weight_ih_l{layer}').data
        W_hh = getattr(rnn, f'weight_hh_l{layer}').data
        b_h = getattr(rnn, f'bias_ih_l{layer}').data + getattr(rnn, f'bias_hh_l{layer}').data

        outputs = []
        for t in range(seq_len):
            x_t = layer_input[:, t, :]
            h_t = torch.tanh(x_t @ W_xh.T + h_prev[layer] @ W_hh.T + b_h)
            outputs.append(h_t.unsqueeze(1))
            h_prev[layer] = h_t
        layer_outputs.append(torch.cat(outputs, dim=1))

    return layer_outputs[-1], torch.stack(h_prev, dim=0)


# 使用修改后的手动RNN前向传播
manual_output, manual_hidden = manual_rnn_forward(input_data, rnn)

# 使用PyTorch的RNN前向传播
output, hidden = rnn(input_data)

# 比较手动实现和PyTorch实现的结果
print("Output close:", torch.allclose(manual_output, output, atol=1e-4))
print("Hidden state close:", torch.allclose(manual_hidden, hidden, atol=1e-4))
