import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model = 16, d_k = 8):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_k, bias = False)
        self.k_proj = nn.Linear(d_model, d_k, bias = False)
        self.v_proj = nn.Linear(d_model, d_k, bias = False)
        self.w_0 = nn.Linear(d_k, d_model, bias = False)
        self.d_model = d_model
        self.d_k = d_k

    def forward(self, x):
        '''
        x: (seq_length x d_model)
        output : (seq_length x d_model)
        '''
        q = self.q_proj(x) # (seq_length x d_k)
        k = self.k_proj(x) # (seq_length x d_k)
        v = self.v_proj(x) # (seq_length x d_k)

        # compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k) # (seq_length x seq_length)

        # compute attention weights
        weights = torch.softmax(scores, dim=-1) # (seq_length x seq_length)

        # compute output
        output = torch.matmul(weights, v)  # (seq_length x d_k)

        # project back to d_model
        return self.w_0(output) # (seq_length x d_model)

        