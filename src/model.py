import torch
import torch.nn as nn
from src.modules import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding, AddNorm
import math

class EncoderLayer(nn.Module):
    """
    单个编码器层
    包含：1. 多头自注意力 2. 逐位置前馈网络
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm_1 = AddNorm(d_model, dropout)
        self.add_norm_2 = AddNorm(d_model, dropout)

    def forward(self, x, mask):
        # 1. Multi-Head Self-Attention
        attn_output = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = self.add_norm_1(x, attn_output) # Add & Norm
        
        # 2. Position-wise Feed-Forward
        ffn_output = self.ffn(x)
        x = self.add_norm_2(x, ffn_output) # Add & Norm
        
        return x

class DecoderLayer(nn.Module):
    """
    单个解码器层
    包含：1. 带掩码的多头自注意力 2. 编码器-解码器注意力 3. 逐位置前馈网络
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm_1 = AddNorm(d_model, dropout)
        self.add_norm_2 = AddNorm(d_model, dropout)
        self.add_norm_3 = AddNorm(d_model, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Masked Multi-Head Self-Attention
        attn_output = self.masked_self_attn(query=x, key=x, value=x, mask=tgt_mask)
        x = self.add_norm_1(x, attn_output) # Add & Norm
        
        # Encoder-Decoder Attention
        # Query: from decoder (x), Key/Value: from encoder_output
        enc_dec_attn_output = self.enc_dec_attn(query=x, key=encoder_output, value=encoder_output, mask=src_mask)
        x = self.add_norm_2(x, enc_dec_attn_output) # Add & Norm
        
        # Position-wise Feed-Forward
        ffn_output = self.ffn(x)
        x = self.add_norm_3(x, ffn_output) # Add & Norm
        
        return x

class Encoder(nn.Module):
    """
    N个EncoderLayer堆叠
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return x # shape: (batch_size, seq_len, d_model)

class Decoder(nn.Module):
    """
    N个DecoderLayer堆叠
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
            
        return x # shape: (batch_size, seq_len, d_model)

class Transformer(nn.Module):
    """
    完整的Transformer(Encoder-Decoder)模型
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_params()

    def _init_params(self):
        # 初始化参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_src_mask(self, src, pad_token):
        # (batch_size, 1, 1, src_len)
        src_mask = (src != pad_token).unsqueeze(1).unsqueeze(2)
        return src_mask

    def create_tgt_mask(self, tgt, pad_token):
        # (batch_size, 1, tgt_len, 1)
        tgt_pad_mask = (tgt != pad_token).unsqueeze(1).unsqueeze(3)
        
        # (tgt_len, tgt_len)
        seq_len = tgt.size(1)
        tgt_future_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool() # [cite: 60]
        
        # (batch_size, 1, tgt_len, tgt_len)
        tgt_mask = tgt_pad_mask & tgt_future_mask
        return tgt_mask

    def forward(self, src, tgt, src_pad_token, tgt_pad_token):
        # src: (batch_size, src_len)
        # tgt: (batch_size, tgt_len)
        src_mask = self.create_src_mask(src, src_pad_token)
        tgt_mask = self.create_tgt_mask(tgt, tgt_pad_token)
        
        # (batch_size, src_len, d_model)
        encoder_output = self.encoder(src, src_mask)
        
        # (batch_size, tgt_len, d_model)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # (batch_size, tgt_len, tgt_vocab_size)
        output = self.output_linear(decoder_output)
        
        return output