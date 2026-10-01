# Derived from ELF-pytorch (https://github.com/Ugness/ELF-pytorch).
# Copyright (c) 2026 ELF authors. MIT License (see LICENSE).
"""Building blocks of the ELF transformer."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_(t, std=0.02):
    return nn.init.normal_(t, mean=0.0, std=std)


def linear(in_features, out_features, bias=True, init=nn.init.xavier_uniform_):
    """nn.Linear with the given weight init and a zero bias."""
    lin = nn.Linear(in_features, out_features, bias=bias)
    init(lin.weight)
    if bias:
        nn.init.zeros_(lin.bias)
    return lin


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight.to(torch.float32) * x).to(dtype)


def rotate_half(x):
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x[..., 0], x[..., 1]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    """1D rotary position embedding. The first `num_prefix` positions (the
    conditioning tokens) are left unrotated."""

    def __init__(self, dim, seq_len, num_prefix=0, theta=10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
        pos = torch.arange(seq_len).float()
        freqs = torch.einsum("..., f -> ... f", pos, freqs)
        freqs = freqs.unsqueeze(-1).expand(*freqs.shape, 2).flatten(-2)
        cos = torch.cat([torch.ones(num_prefix, freqs.shape[-1]), torch.cos(freqs)], dim=0)
        sin = torch.cat([torch.zeros(num_prefix, freqs.shape[-1]), torch.sin(freqs)], dim=0)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, t):
        cos = self.freqs_cos.to(dtype=t.dtype, device=t.device).view(1, 1, *self.freqs_cos.shape)
        sin = self.freqs_sin.to(dtype=t.dtype, device=t.device).view(1, 1, *self.freqs_sin.shape)
        return t * cos + rotate_half(t) * sin


class BottleneckProj(nn.Module):
    """Input projection through a low-dimensional bottleneck."""

    def __init__(self, in_dim, hidden_size, bottleneck_dim):
        super().__init__()
        self.proj1 = linear(in_dim, bottleneck_dim, bias=False)
        self.proj2 = linear(bottleneck_dim, hidden_size)

    def forward(self, x):
        return self.proj2(self.proj1(x))


class Attention(nn.Module):
    """Self-attention with RMSNorm on queries and keys and rotary positions."""

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = linear(dim, dim * 3)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = linear(dim, dim)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = rope(self.q_norm(q)), rope(self.k_norm(k))
        # Attention weights are computed in float32.
        w = torch.einsum("bhld,bhsd->bhls", q.float(), k.float()) / math.sqrt(self.head_dim)
        w = torch.softmax(w, dim=-1).to(v.dtype)
        out = torch.einsum("bhls,bhsd->bhld", w, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden = int(hidden_dim * 2 / 3)
        self.w12 = linear(dim, 2 * hidden)
        self.w3 = linear(hidden, dim)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class ScalarEmbedder(nn.Module):
    """Sinusoidal features of a scalar (time or guidance scale) followed by an MLP."""

    def __init__(self, hidden_size, num_freqs=256):
        super().__init__()
        self.num_freqs = num_freqs
        self.mlp_0 = linear(num_freqs, hidden_size, init=normal_)
        self.mlp_2 = linear(hidden_size, hidden_size, init=normal_)

    def forward(self, t):
        half = self.num_freqs // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32,
                                                          device=t.device) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp_2(F.silu(self.mlp_0(emb)))


class FinalLayer(nn.Module):
    """Zero-initialized output projection."""

    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = linear(hidden_size, out_dim, init=nn.init.zeros_)

    def forward(self, x):
        return self.linear(self.norm(x))
