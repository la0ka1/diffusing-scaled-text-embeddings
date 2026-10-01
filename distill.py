"""Distill the T5Gemma-2 encoder into a smaller student encoder.

The text is encoded twice, by the frozen teacher encoder (T5Gemma-2) and by the
student, and both embeddings are read by the frozen teacher decoder with teacher
forcing. This gives two next-token distributions for the same text, and the
student is trained to minimize the KL divergence between them.

Distillation objectives (`--objective`):
  kl   match the teacher's top-k renormalized probabilities (ours)
  ce   predict the ground-truth token through the T5Gemma-2 decoder (hard labels)
  mse  regress the teacher's embeddings directly

Single GPU:  python distill.py --data data/owt --out runs/student
Multi GPU:   torchrun --standalone --nproc_per_node=8 distill.py --data data/owt --out runs/student
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from encoders import TEACHER, collate, init_student, text_encoder

BF16 = dict(device_type="cuda", dtype=torch.bfloat16)
TOPK = 256            # teacher's soft labels are Top-K renormalized probabilities
LOGIT_CHUNK = 32      # positions per chunk when computing vocabulary logits
WARMUP = 1000
EVAL_EVERY = 1000
HELDOUT_ROWS = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="token cache built by data.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--objective", default="kl", choices=["kl", "ce", "mse"])
    ap.add_argument("--layers", type=int, default=9, help="student depth (the teacher has 18)")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="per GPU; the paper uses 8 GPUs x 32 = 256")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank, world = 0, 1
    dev = "cuda"
    torch.manual_seed(args.seed + rank)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(TEACHER)
    pad_id = tok.pad_token_id
    model = AutoModelForSeq2SeqLM.from_pretrained(TEACHER)
    teacher = text_encoder(model)
    if args.objective != "mse":
        seq2seq = model.model.float().to(dev).eval().requires_grad_(False)   # encoder-decoder without the LM head
        W_out = model.get_output_embeddings().weight.float().to(dev)         # output embedding (vocab, dim)
        dec_start = getattr(model.config, "decoder_start_token_id", None) or \
            getattr(model.config.get_text_config(decoder=True), "bos_token_id", 2)
    teacher = teacher.float().to(dev).eval().requires_grad_(False)

    student, keep = init_student(teacher, args.layers)
    student = student.float().to(dev).train()
    net = torch.nn.parallel.DistributedDataParallel(student, device_ids=[rank]) if ddp else student
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(     # linear warmup, then cosine decay
        opt, lambda s: min(s / WARMUP, 0.5 * (1 + math.cos(math.pi * s / args.steps))))

    coll = collate(args.seq_len, pad_id)
    train = load_from_disk(args.data)
    heldout = load_from_disk(args.data.rstrip("/") + "_heldout")
    sampler = DistributedSampler(train, shuffle=True, seed=args.seed) if ddp else None
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=4, collate_fn=coll, drop_last=True)
    heldout_ids = next(iter(DataLoader(heldout.select(range(HELDOUT_ROWS)),
                                       batch_size=HELDOUT_ROWS, collate_fn=coll)))

    def embed(enc, ids, am):
        return enc(input_ids=ids, attention_mask=am).last_hidden_state.float()

    def decoder_states(z, ids, am):
        """Teacher-forced pass of the frozen decoder reading the embeddings z."""
        dec_in = torch.cat([torch.full_like(ids[:, :1], dec_start), ids[:, :-1]], 1)
        with torch.autocast(**BF16):
            return seq2seq(encoder_outputs=BaseModelOutput(last_hidden_state=z),
                           attention_mask=am, decoder_input_ids=dec_in).last_hidden_state

    def decoder_loss(ids, am, zs, zt):
        """Cross-entropy between the target distribution and the student-side
        prediction, averaged over non-padding tokens. For `kl` this equals the KL
        divergence up to the entropy of the target, which does not depend on the student."""
        hs = decoder_states(zs, ids, am)
        if args.objective == "kl":
            with torch.no_grad():
                ht = decoder_states(zt.to(zs.dtype), ids, am)
        mask = (ids != pad_id).float()
        loss = zs.new_zeros(())
        for i in range(0, hs.size(1), LOGIT_CHUNK):
            sl = slice(i, i + LOGIT_CHUNK)
            if args.objective == "kl":
                with torch.no_grad():
                    top_v, idx = F.linear(ht[:, sl].float(), W_out).topk(TOPK, dim=-1)
                    target = F.softmax(top_v, dim=-1)       # renormalized on the top-k support
            else:
                idx = ids[:, sl].unsqueeze(-1)
                target = torch.ones_like(idx, dtype=torch.float32)

            def chunk_loss(h, idx, target, m):
                logp = F.log_softmax(F.linear(h.float(), W_out), dim=-1)
                return (-(target * logp.gather(-1, idx)).sum(-1) * m).sum()

            loss = loss + checkpoint(chunk_loss, hs[:, sl], idx, target, mask[:, sl],
                                     use_reentrant=False)
        return loss / mask.sum().clamp_min(1)

    def losses(ids, train=True):
        """Returns (objective, cosine similarity between student and teacher embeddings)."""
        am = (ids != pad_id).long()
        valid = am.bool()
        with torch.autocast(**BF16):
            with torch.no_grad():
                zt = embed(teacher, ids, am)
            with torch.set_grad_enabled(train):
                zs = embed(net if train else student, ids, am)
        with torch.set_grad_enabled(train):
            if args.objective == "mse":
                loss = ((zs - zt) ** 2)[valid].mean()
            else:
                loss = decoder_loss(ids, am, zs, zt)
        with torch.no_grad():
            cos = F.cosine_similarity(zs, zt, -1)[valid].mean().item()
        return loss, cos

    @torch.no_grad()
    def evaluate():
        res = [losses(heldout_ids[i:i + 64].to(dev), train=False)
               for i in range(0, len(heldout_ids), 64)]
        return sum(float(l) for l, _ in res) / len(res), sum(c for _, c in res) / len(res)

    if rank == 0:
        n_train = sum(p.numel() for p in params) / 1e6
        print(f"[distill] objective={args.objective} layers={keep} trainable={n_train:.1f}M "
              f"steps={args.steps} batch={args.batch_size}x{world}", flush=True)
    it, epoch = iter(loader), 0
    best, t0 = float("-inf"), time.time()
    for step in range(1, args.steps + 1):
        try:
            ids = next(it).to(dev)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            it = iter(loader)
            ids = next(it).to(dev)
        loss, cos = losses(ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (step == 1 or step % args.log_every == 0) and rank == 0:
            print(f"  step {step:6d}/{args.steps} {args.objective} {loss.item():.4f} cos {cos:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if step % EVAL_EVERY == 0 and rank == 0:
            val, val_cos = evaluate()
            print(f"  [eval {step}] heldout {args.objective} {val:.4f} cos {val_cos:.4f}", flush=True)
            ckpt = {"teacher": TEACHER, "keep": keep, "step": step, "objective": args.objective,
                    "student_sd": student.state_dict()}
            torch.save(ckpt, out / "last.pt")
            # Model selection on held-out data: cosine similarity for mse, else the loss itself.
            score = val_cos if args.objective == "mse" else -val
            if score > best:
                best = score
                torch.save(ckpt, out / "best.pt")
        if ddp and step % EVAL_EVERY == 0:
            dist.barrier()
    if rank == 0:
        print(f"[distill] done -> {out / 'best.pt'}", flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
