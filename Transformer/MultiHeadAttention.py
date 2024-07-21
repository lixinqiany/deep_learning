import torch
from torch import nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask):
        """
        :param Q: (batch, heads, seq_length, d_Q)
        :param V:
        :param mask: (batch, heads, seq_len, seq_len)
        :return:
        """
        # (batch, heads, seq_length, d_Q) * (batch, heads, d_Q, seq_length) =>
        # (batch, heads, seq_len, seq_len)
        d_Q = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_Q)
        # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(mask, -1e9)
        # the sum of each row is 1
        attention_layer = nn.Softmax(dim=-1)
        attetion = attention_layer(scores)
        # (batch_size, n_heads, seq_len, d_v)
        context = torch.matmul(attetion, V)

        return context, attetion



class MultiHeadAttention(nn.Module):
    """
    @:param input_size the dimension of each input sequence
    @:param heads the number of heads you wang to define
    """
    def __init__(self, input_size, heads, d_Q, d_K, d_V):
        """
        Dimensions of three matrix =>
        Assume the input: (num_embeddings, seq_len)
        W_Q * A = Q => (d_Q, num_embeddings) * (num_embeddings, seq_len) = (d_Q, seq_len) <= query matrix
        K^T * Q = Att => (seq_len, d_W) * (d_Q, seq_len) = (seq_len, seq_len) <= attention matrix
        So, d_W = d_Q

        W_K * A = K => (d_W, num_embeddings) * (num_embeddings, seq_len) = (d_W, seq_len)
        W_V * A = V => (d_V, num_embeddings) * (num_embeddings, seq_len) = (d_V, seq_len)
        context = V * Attn => (d_V, seq_len) * (seq_len, seq_len) = (d_V, seq_len)
        """
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.d_Q = d_Q
        self.d_V = d_V
        self.W_Q = nn.Linear(input_size, d_Q*heads, bias=False)
        self.W_K = nn.Linear(input_size, d_K*heads, bias=False)
        self.W_V = nn.Linear(input_size, d_V*heads, bias=False)
        self.fc = nn.Linear(heads * d_V, input_size, bias=False)
        self.Norm = nn.LayerNorm(input_size)

    def forward(self, input, mask):
        """
        :param input: the output of the embedding layer that will be multiplied by W_Q, W_K & W_V
        :return:

        Dimensions of the input =>
        (batch, seq_len, num_embeddings)
        Dimensions of the param Matrix =>
        (num_embeddings, d_Q)
        Dimensions of the mask for decoder =>
        (batch, seq_len, seq_len) => (batch, heads, seq_len, seq_len)
        """
        batch = input.size(0)
        mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)
        residual = input

        # (batch, seq_len, num_embeddings) => (batch, seq_length, d_Q*heads) => (batch, seq_length, heads, d_Q)
        # (batch, heads, seq_length, d_Q)
        Q = self.W_Q(input)
        Q = Q.view(batch, -1, self.heads, self.d_Q).transpose(1,2)
        K = self.W_K(input)
        K = K.view(batch, -1, self.heads, self.d_Q).transpose(1, 2)
        V = self.W_V(input)
        V = V.view(batch, -1, self.heads, self.d_V).transpose(1, 2)

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context, attention = ScaledDotProductAttention()(Q, K, V, mask)
        context = context.transpose(1, 2).reshape(batch, -1, self.heads * self.d_V)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        Add = output + residual

        return self.Norm(Add), attention






