"""Train ELF on the embeddings of a frozen text encoder.

The encoder is our released student by default (T5Gemma-2-270M-OWTdistilled, downloaded
from Hugging Face), a best.pt written by distill.py, or a Hugging Face model id: T5Gemma-2
(google/t5gemma-2-270m-270m) or another T5-family model such as google-t5/t5-small,
on a cache built with its tokenizer. Embeddings are computed on the fly.

The paper trains for 5 epochs at a global batch of 512 sequences:
  ELF-B: torchrun --standalone --nproc_per_node=8 train.py --model ELF-B --batch-size 16 --accum 4 ...
  ELF-M: torchrun --standalone --nproc_per_node=8 train.py --model ELF-M --batch-size 8 --accum 8 ...
"""

import argparse
import contextlib
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from elf import SIZES, build_elf
from elf.flow import training_loss
from elf.muon import build_optimizers
from encoders import STUDENT, TEACHER, Encoder, collate

BASE_LR = 1e-3        # learning rate = BASE_LR * global batch / 256
WARMUP = 2000
EMA_DECAY = 0.9999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="token cache built by data.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default=STUDENT,
                    help=f"{STUDENT} (released), a student best.pt, or a Hugging Face id such as "
                         f"{TEACHER} or google-t5/t5-small")
    ap.add_argument("--model", default="ELF-B", choices=list(SIZES))
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=0, help="if > 0, overrides --epochs")
    ap.add_argument("--batch-size", type=int, default=16, help="per GPU")
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--save-every", type=int, default=7812, help="steps between checkpoints")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--resume", default="", help="checkpoint to resume from")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp = world > 1
    if ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    device = f"cuda:{local_rank}" if ddp else "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if ddp and rank > 0:    # rank 0 downloads the encoder first; the others then read the cache
        dist.barrier()
    encoder = Encoder(args.encoder, device)
    if ddp and rank == 0:
        dist.barrier()
    pad_id = encoder.pad_id
    data = load_from_disk(args.data)
    global_batch = args.batch_size * world * args.accum
    steps = args.steps or math.ceil(args.epochs * len(data) / global_batch)
    lr = BASE_LR * global_batch / 256

    # Each process shuffles the full dataset with its own seed.
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        collate_fn=collate(args.seq_len, pad_id), drop_last=True,
                        persistent_workers=True,
                        generator=torch.Generator().manual_seed(args.seed * 1000 + rank))

    def batches():
        while True:
            yield from loader

    mean, std = encoder.mean_std(data, args.seq_len)
    config = dict(model=args.model, encoder=args.encoder, tokenizer=encoder.tokenizer_name,
                  dim=encoder.dim, seq_len=args.seq_len, vocab_size=len(encoder.tokenizer),
                  mean=mean, std=std)

    model = build_elf(args.model, encoder.dim, args.seq_len, config["vocab_size"]).to(device)
    opts = build_optimizers(model, lr)
    # A step trains either the denoising head or the decoding head, so some
    # parameters receive no gradient.
    net = DistributedDataParallel(model, device_ids=[local_rank],
                                  find_unused_parameters=True) if ddp else model
    ema = {n: p.detach().clone() for n, p in model.named_parameters()}

    def save(path, step):
        torch.save({"config": config, "step": step,
                    "ema": {n: p.cpu() for n, p in ema.items()},
                    "model": {n: p.cpu() for n, p in model.state_dict().items()},
                    "optimizers": [o.state_dict() for o in opts]}, path)

    start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        ema = {n: p.clone() for n, p in ckpt["ema"].items()}
        for o, s in zip(opts, ckpt["optimizers"]):
            o.load_state_dict(s)
        start = ckpt["step"] + 1

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[train] {args.model} ({n_params:.1f}M) on {args.encoder}: {len(data):,} sequences, "
              f"{steps} steps, global batch {global_batch}, lr {lr:.1e}, "
              f"embedding mean {mean:.4f} std {std:.4f}", flush=True)

    gen = torch.Generator(device=device).manual_seed(args.seed + rank)
    data_iter = batches()
    t0 = time.time()
    model.train()
    for step in range(start, steps):
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = lr * (step + 1) / WARMUP if step < WARMUP else lr
            opt.zero_grad(set_to_none=True)
        total = 0.0
        for micro in range(args.accum):
            ids = next(data_iter).to(device)
            x0 = (encoder(ids) - mean) / std
            # Gradients are synchronized across processes on the last micro-batch only.
            sync = micro == args.accum - 1 or not ddp
            with contextlib.nullcontext() if sync else net.no_sync():
                loss = training_loss(net, x0, ids, pad_id, gen, nograd_model=model) / args.accum
                loss.backward()
            total += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if torch.isfinite(grad_norm):
            for opt in opts:
                opt.step()
            with torch.no_grad():
                for n, p in model.named_parameters():
                    ema[n].mul_(EMA_DECAY).add_(p.detach(), alpha=1 - EMA_DECAY)
        if step % args.log_every == 0 and rank == 0:
            print(f"  step {step:6d}/{steps} loss {total:.4f} {time.time() - t0:.0f}s", flush=True)
        if step > 0 and step % args.save_every == 0:
            if rank == 0:
                save(out / "checkpoint.pt", step)
            if ddp:
                dist.barrier()

    if rank == 0:
        save(out / "model.pt", steps - 1)
        print(f"[train] done -> {out / 'model.pt'}", flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
