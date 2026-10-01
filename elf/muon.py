# Derived from ELF-pytorch (https://github.com/Ugness/ELF-pytorch).
# Copyright (c) 2026 ELF authors. MIT License (see LICENSE).
"""Muon optimizer (after Keller Jordan's reference implementation): momentum
followed by Newton-Schulz orthogonalization of the update, for 2D weights only."""

import torch


def orthogonalize(G, steps=5):
    """Newton-Schulz iteration; returns a matrix with singular values close to 1."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(dtype=torch.bfloat16)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.transpose(-1, -2)
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-1, -2)
    return X.to(dtype=G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.95, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                update = orthogonalize(g.add(buf, alpha=group["momentum"]), group["ns_steps"])
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(update, alpha=-group["lr"] * scale)


def build_optimizers(model, lr):
    """Muon for the 2D weights, AdamW for everything else (biases, norm gains, tokens)."""
    params = [p for p in model.parameters() if p.requires_grad]
    muon = Muon([p for p in params if p.ndim == 2], lr=lr)
    adamw = torch.optim.AdamW([p for p in params if p.ndim != 2], lr=lr, betas=(0.9, 0.95),
                              weight_decay=0.0)
    return [muon, adamw]
