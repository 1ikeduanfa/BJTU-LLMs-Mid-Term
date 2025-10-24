import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AddNorm(nn.Module):
    """
    残差连接 + 层归一化
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        """
        Args:
            x: 子层的输入 (residual)
            sublayer_output: 子层的输出
        """
        return self.layer_norm(x + self.dropout(sublayer_output))

class PositionwiseFeedForward(nn.Module):
    """
    逐位置前馈网络
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        x = torch.relu(self.linear_1(x))
        x = self.dropout(x)
        # (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.linear_2(x)
        return x

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须被 n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads # d_k = d_v

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        """
        # Q, K, V shape: (batch_size, n_heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask shape: (batch_size, 1, seq_len, seq_len) or (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        # output shape: (batch_size, n_heads, seq_len, d_k)
        return output, attn_weights

    def forward(self, query, key, value, mask=None):
        # query, key, value shape: (batch_size, seq_len, d_model)
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # x shape: (batch_size, n_heads, seq_len, d_k)
        x, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # x shape: (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.W_o(x)
        
        return x

class PositionalEncoding(nn.Module):
    # 正弦位置编码
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # [cite: 55]
        pe[:, 1::2] = torch.cos(position * div_term) # [cite: 55]
        
        pe = pe.unsqueeze(0) # shape (1, max_len, d_model)
        
        # register_buffer 使得pe成为模型的一部分，但不是可训练参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # self.pe shape: (1, max_len, d_model)
        # 取x序列长度的pe
        x = x + self.pe[:, :x.size(1), :]
        return x