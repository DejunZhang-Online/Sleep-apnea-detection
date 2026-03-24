import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple, Optional
from functools import partial
from collections import OrderedDict
from einops import rearrange


class TopKSparseAttention(nn.Module):
    """
    Top-K sparse self-attention.

    For each query row, only the top-k attention scores are kept and the remaining
    entries are masked to -inf before softmax. The retention ratio kr controls k:

        k = max(1, ceil(kr * seq_len))

    Input shape:
        (batch_size, channels, seq_len)

    Output shape:
        (batch_size, channels, seq_len)
    """

    def __init__(self, dim: int, num_heads: int, k_ratio: float = 0.25, bias: bool = False, attn_dropout: float = 0.0):
        super().__init__()
        if not (0.0 < k_ratio <= 1.0):
            raise ValueError(f"k_ratio must be in (0, 1], got {k_ratio}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.k_ratio = k_ratio
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv1d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv1d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, _, seq_len = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) s -> b head s c", head=self.num_heads)
        k = rearrange(k, "b (head c) s -> b head s c", head=self.num_heads)
        v = rearrange(v, "b (head c) s -> b head s c", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn_scores = (q @ k.transpose(-2, -1)) * self.temperature
        topk = max(1, min(seq_len, int(torch.ceil(torch.tensor(self.k_ratio * seq_len)).item())))
        indices = torch.topk(attn_scores, k=topk, dim=-1, largest=True).indices

        sparse_scores = torch.full_like(attn_scores, float("-inf"))
        sparse_scores.scatter_(-1, indices, torch.gather(attn_scores, -1, indices))
        sparse_scores = torch.clamp(sparse_scores, min=-1e4, max=1e4)

        attn = torch.softmax(sparse_scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = rearrange(out, "b head s c -> b (head c) s", head=self.num_heads, s=seq_len)
        out = self.project_out(out)

        if return_attention:
            return out, attn
        return out

class Seq_Encoder(nn.Module):
    """Sparse Attention Model for many to many translation."""

    def __init__(
            self,
            seq_length: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderTSABlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        output = self.layers(self.dropout(input))
        return self.ln(output)


class EncoderTSABlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.RMSNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = TopKSparseAttention(hidden_dim, num_heads, bias=False)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (seq_length, batch_size, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = x.permute(0, 2, 1)
        x = self.self_attention(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)


if __name__ == '__main__':
    model = Seq_Encoder(
        seq_length=60,
        num_layers=2,
        num_heads=2,
        hidden_dim=64,
        mlp_dim=64,
        dropout=0.1,
        attention_dropout=0.1)

    input = torch.rand(3, 60, 64)
    output = model(input)
    print(output.shape)
