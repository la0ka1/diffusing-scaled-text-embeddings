# Derived from ELF-pytorch (https://github.com/Ugness/ELF-pytorch).
# Copyright (c) 2026 ELF authors. MIT License (see LICENSE).
"""The ELF transformer: a denoiser and a decoding head on one shared trunk."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import (Attention, BottleneckProj, FinalLayer, RMSNorm, RotaryEmbedding,
                     ScalarEmbedder, SwiGLU, normal_)

NUM_TIME_TOKENS = 4    # prefix tokens carrying the time t
NUM_SC_TOKENS = 4      # prefix tokens carrying the self-conditioning guidance scale
NUM_MODE_TOKENS = 4    # prefix tokens that switch the trunk to decoding mode
BOTTLENECK_DIM = 128
CE_CHUNK = 16          # positions per chunk of the decoding loss


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class ELF(nn.Module):
    """Input: noisy embeddings concatenated with the self-conditioning estimate,
    shape (B, L, 2 * dim). Two modes share the trunk:

    - denoising: returns the prediction of the clean embeddings, (B, L, dim);
    - decoding (`decode=True`): returns token logits, (B, L, vocab_size).
    """

    def __init__(self, dim, seq_len, vocab_size, hidden_size, depth, num_heads):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.self_cond_proj = nn.Linear(2 * dim, dim)
        nn.init.xavier_uniform_(self.self_cond_proj.weight)
        nn.init.zeros_(self.self_cond_proj.bias)
        self.text_proj = BottleneckProj(dim, hidden_size, BOTTLENECK_DIM)

        self.t_embedder = ScalarEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(normal_(torch.empty(1, NUM_TIME_TOKENS, hidden_size)))
        self.self_cond_cfg_embedder = ScalarEmbedder(hidden_size)
        self.self_cond_cfg_tokens = nn.Parameter(normal_(torch.empty(1, NUM_SC_TOKENS, hidden_size)))
        self.mode_tokens = nn.Parameter(normal_(torch.empty(1, NUM_MODE_TOKENS, hidden_size)))

        self.num_prefix = NUM_TIME_TOKENS + NUM_SC_TOKENS + NUM_MODE_TOKENS
        self.feat_rope = RotaryEmbedding(hidden_size // num_heads, seq_len, self.num_prefix)
        self.blocks = nn.ModuleList([Block(hidden_size, num_heads) for _ in range(depth)])

        # Decoding head: hidden -> dim -> vocabulary.
        self.proj_kernel = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_size, dim)))
        self.proj_bias = nn.Parameter(torch.zeros(dim))
        self.unembed_kernel = nn.Parameter(nn.init.xavier_uniform_(torch.empty(dim, vocab_size)))
        self.unembed_bias = nn.Parameter(torch.zeros(vocab_size))
        # Denoising head.
        self.final_layer = FinalLayer(hidden_size, dim)

    def logits(self, h):
        h = F.gelu(h @ self.proj_kernel + self.proj_bias)
        return h @ self.unembed_kernel + self.unembed_bias

    def forward(self, x, t, sc_scale, decode=False, targets=None):
        """x: (B, L, 2*dim), t: (B,), sc_scale: (B,).

        With `decode=True` and `targets=(ids, pad_id)` the cross-entropy is
        computed here in chunks, so that the full (B, L, vocab) logits are never
        held in memory; the return value is then (loss_sum, num_tokens).
        """
        B = x.shape[0]
        x = self.text_proj(self.self_cond_proj(x))

        mode = self.mode_tokens.expand(B, -1, -1)
        if not decode:
            mode = torch.zeros_like(mode)
        prefix = torch.cat([
            self.t_emb_tokens.expand(B, -1, -1) + self.t_embedder(t).unsqueeze(1),
            self.self_cond_cfg_tokens.expand(B, -1, -1)
            + self.self_cond_cfg_embedder(sc_scale).unsqueeze(1),
            mode], dim=1)
        x = torch.cat([prefix, x], dim=1)
        for block in self.blocks:
            x = block(x, self.feat_rope)
        x = x[:, self.num_prefix:]

        if not decode:
            return self.final_layer(x)
        if targets is None:
            return self.logits(x)

        ids, pad_id = targets

        def chunk_ce(h, y):
            logp = F.log_softmax(self.logits(h).float(), dim=-1)
            ce = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            mask = (y != pad_id).float()
            return (ce * mask).sum(), mask.sum()

        ce_sum = torch.zeros((), device=x.device, dtype=torch.float32)
        n_tok = torch.zeros((), device=x.device, dtype=torch.float32)
        for i in range(0, x.shape[1], CE_CHUNK):
            c, n = checkpoint(chunk_ce, x[:, i:i + CE_CHUNK], ids[:, i:i + CE_CHUNK],
                              use_reentrant=False)
            ce_sum, n_tok = ce_sum + c, n_tok + n
        return ce_sum, n_tok


SIZES = {
    "ELF-B": dict(depth=12, hidden_size=768, num_heads=12),
    "ELF-M": dict(depth=24, hidden_size=1056, num_heads=16),
}


def build_elf(size, dim, seq_len, vocab_size):
    return ELF(dim=dim, seq_len=seq_len, vocab_size=vocab_size, **SIZES[size])
