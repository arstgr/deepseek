import torch
import torch.nn as nn

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads ==0, \
            "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.d_head)
        self.W_v = nn.Linear(d_model, self.d_head)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(1,1,1024), diagonal=1))

        def forward(self, x):
            batch_size, seq_len, _ = x.shape

            q = self.W_q(x).view(batch_size, seq_len, self.num_heads,
                self.d_head).transpose(1,2)

            k = self.W_k(x).view(batch_size, seq_len, 1,
                self.d_head).transpose(1,2)

            v = self.W_v(x).view(batch_size, seq_len, 1,
                self.d_head).transpose(1,2)

            k = k.repeat(1, self.num_heads, 1, 1)
            v = v.repeat(1, self.num_heads, 1, 1)

            attn_scores = (q@k.transpose(-2,-1)) / (self.d_head ** 0.5)

            attn_scores = attn_scores.masked_fill(
                self.mask[:,:,:seq_len,:seq_len] == 1, float('-inf')
            )

            attn_weights = torch.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context_vector = (attn_weights @ v).transpose(1,2) \
                .contigous().view(batch_size, seq_len, self.d_model)

            output = self.W_o(context_vector)
            return output
