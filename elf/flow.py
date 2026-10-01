# Derived from ELF-pytorch (https://github.com/Ugness/ELF-pytorch).
# Copyright (c) 2026 ELF authors. MIT License (see LICENSE).
"""Flow matching with x-prediction: training losses, sampler and decoding.

Time runs from t=0 (noise) to t=1 (data): z_t = t * x + (1 - t) * noise_scale * eps.
The network predicts x; the velocity is v = (x - z_t) / max(1 - t, T_EPS).
"""

import math

import torch

T_EPS = 0.05
NOISE_SCALE = 2.0                   # noise scale of the denoising task
P_MEAN, P_STD = -1.5, 0.8           # logit-normal time distribution
SELF_COND_PROB = 0.5
SC_SCALE_MIN, SC_SCALE_MAX = 0.5, 5.0
DECODER_PROB = 0.2                  # fraction of training steps spent on the decoding task
DECODER_NOISE_SCALE = 5.0
DECODER_P_MEAN, DECODER_P_STD = 0.8, 0.8
DECODE_CHUNK = 4                    # sequences decoded at once; their (chunk, L, vocab) logits take ~1 GB each


def sample_timesteps(gen, n, device):
    return torch.sigmoid(torch.randn(n, generator=gen, device=device) * P_STD + P_MEAN)


def sample_sc_scale(gen, n, device):
    """Guidance scale for self-conditioning, log-uniform in [SC_SCALE_MIN, SC_SCALE_MAX]."""
    u = torch.rand(n, generator=gen, device=device)
    a, b = 1.0 + SC_SCALE_MIN, 1.0 + SC_SCALE_MAX
    return a * torch.exp(u * math.log(b / a)) - 1.0


def velocity(x_pred, z, t):
    return (x_pred - z) / torch.clamp(1.0 - t.view(-1, 1, 1), min=T_EPS)


def decoder_loss(model, x0, ids, pad_id, gen):
    """Cross-entropy of the decoding head on corrupted embeddings. The corruption
    level is drawn per token; the trunk runs at t=1."""
    B, L, _ = x0.shape
    device, dtype = x0.device, x0.dtype
    lam = torch.randn(B * L, generator=gen, device=device) * DECODER_P_STD + DECODER_P_MEAN
    lam = torch.sigmoid(lam).view(B, L, 1).to(dtype)
    noise = torch.randn(x0.shape, generator=gen, device=device, dtype=dtype) * DECODER_NOISE_SCALE
    z = lam * x0 + (1.0 - lam) * noise
    t = torch.ones(B, device=device, dtype=dtype)
    sc_scale = sample_sc_scale(gen, B, device).to(dtype)
    ce_sum, n_tok = model(torch.cat([z, torch.zeros_like(z)], dim=-1), t, sc_scale,
                          decode=True, targets=(ids, pad_id))
    return ce_sum / n_tok.clamp(min=1.0)


def denoiser_loss(model, x0, ids, pad_id, gen, nograd_model=None):
    """Velocity loss on non-padding positions, with self-conditioning and the
    self-conditioning guidance target. `nograd_model` runs the passes that need
    no gradient (the bare module when `model` is wrapped for distributed training)."""
    nograd_model = nograd_model or model
    B = x0.shape[0]
    device, dtype = x0.device, x0.dtype
    t = sample_timesteps(gen, B, device)
    noise = torch.randn(x0.shape, generator=gen, device=device, dtype=dtype)
    tt = t.view(-1, 1, 1)
    z = tt * x0 + (1 - tt) * noise * NOISE_SCALE
    v_target = velocity(x0, z, t)

    use_sc = (torch.rand(B, generator=gen, device=device) < SELF_COND_PROB).view(-1, 1, 1).to(dtype)
    sc_scale = sample_sc_scale(gen, B, device).to(dtype)
    zeros = torch.zeros_like(z)

    # Self-conditioning input: a first estimate of x, used with probability SELF_COND_PROB.
    with torch.no_grad():
        x_first = nograd_model(torch.cat([z, zeros], dim=-1), t, sc_scale)
    x_pred = model(torch.cat([z, x_first * use_sc], dim=-1), t, sc_scale)
    v_pred = velocity(x_pred, z, t)

    # Guidance target: move the target velocity along (conditioned - unconditioned).
    with torch.no_grad():
        x_uncond = nograd_model(torch.cat([z, zeros], dim=-1), t, sc_scale)
        v_uncond = velocity(x_uncond, z, t)
        v_cond = velocity(nograd_model(torch.cat([z, x_uncond], dim=-1), t, sc_scale), z, t)
        guidance = (1 - 1 / sc_scale.view(B, 1, 1)) * (v_cond - v_uncond)
        guidance = torch.where(use_sc > 0, guidance, torch.zeros_like(guidance))
        v_target = (v_target + guidance).detach()

    mask = (ids != pad_id).to(dtype)
    per_token = ((v_pred - v_target) ** 2).mean(dim=-1)
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def training_loss(model, x0, ids, pad_id, gen, nograd_model=None):
    """One training step: the decoding task with probability DECODER_PROB,
    otherwise the denoising task."""
    if bool((torch.rand((), generator=gen, device=x0.device) < DECODER_PROB).item()):
        return decoder_loss(model, x0, ids, pad_id, gen)
    return denoiser_loss(model, x0, ids, pad_id, gen, nograd_model)


def sampling_schedule(gen, nfe, device):
    """Time grid 0 = t_0 < ... < t_nfe = 1; inner points are sorted logit-normal draws."""
    if nfe < 2:
        return torch.tensor([0.0, 1.0], device=device)
    steps, _ = torch.sort(sample_timesteps(gen, nfe - 1, device))
    return torch.cat([torch.zeros(1, device=device), steps, torch.ones(1, device=device)])


@torch.no_grad()
def sample(model, n, nfe, sc_scale, gen, gamma=1.5, noise_scale=NOISE_SCALE):
    """SDE sampler. Before each Euler step the state is moved back in time by
    re-noising (churn `gamma`); the last step is a plain Euler step.
    Returns embeddings in the normalized space, shape (n, seq_len, dim)."""
    device = next(model.parameters()).device
    t_steps = sampling_schedule(gen, nfe, device)
    z = torch.randn((n, model.seq_len, model.dim), generator=gen, device=device) * noise_scale
    x_pred = torch.zeros_like(z)
    sc = torch.full((n,), float(sc_scale), device=device, dtype=z.dtype)
    last = len(t_steps) - 2
    for k in range(last + 1):
        t, t_next = float(t_steps[k].item()), float(t_steps[k + 1].item())
        if k < last:
            alpha = max(0.0, min(1.0, 1.0 - gamma * (t_next - t)))
            eps = torch.randn(z.shape, generator=gen, device=device, dtype=z.dtype) * noise_scale
            z = alpha * z + (1.0 - alpha) * eps
            t = alpha * t
        tb = torch.full((n,), t, device=device, dtype=z.dtype)
        x_pred = model(torch.cat([z, x_pred], dim=-1), tb, sc)
        z = z + (t_next - t) * velocity(x_pred, z, tb)
    return z


@torch.no_grad()
def decode(model, z, sc_scale, eos_id, pad_id):
    """Embeddings -> token ids with the decoding head (trunk at t=1, argmax).
    Tokens after the first end-of-sequence token are replaced by padding."""
    out = []
    for i in range(0, z.size(0), DECODE_CHUNK):
        zc = z[i:i + DECODE_CHUNK]
        t = torch.full((zc.size(0),), 1.0, device=zc.device, dtype=zc.dtype)
        sc = torch.full((zc.size(0),), float(sc_scale), device=zc.device, dtype=zc.dtype)
        logits = model(torch.cat([zc, torch.zeros_like(zc)], dim=-1), t, sc, decode=True)
        out.append(torch.argmax(logits, dim=-1))
    ids = torch.cat(out, 0)
    is_eos = (ids == eos_id).long()
    after_eos = (is_eos.cumsum(1) - is_eos) > 0
    return ids.masked_fill(after_eos, pad_id)
