"""Sample from a trained ELF and decode the embeddings to text.

  python sample.py --ckpt runs/elf_b/model.pt --out samples.json --n 1024 --nfe 256 --sc 1
  python sample.py --ckpt ELF-M-T5Gemma2distilled --out samples.json --n 1024 --nfe 512 --sc 3   # released checkpoint

Sampler: SDE with churn gamma 1.5 and noise scale 2.0. `--sc` is the
self-conditioning guidance scale and `--nfe` the number of function evaluations.
"""

import argparse
import json

import torch
from transformers import AutoTokenizer

from elf import build_elf
from elf.flow import decode, sample
from encoders import checkpoint_path


def load_model(path, device="cuda"):
    """Returns the model with the averaged (EMA) weights and its config. `path` is
    a model.pt written by train.py, or the name of a released checkpoint."""
    ckpt = torch.load(checkpoint_path(path), map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = build_elf(cfg["model"], cfg["dim"], cfg["seq_len"], cfg["vocab_size"])
    model.load_state_dict(ckpt["ema"])
    return model.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="model.pt of a run, or a released checkpoint: ELF-B-T5Gemma2, "
                         "ELF-M-T5Gemma2, ELF-B-T5Gemma2distilled, ELF-M-T5Gemma2distilled")
    ap.add_argument("--out", required=True, help="output JSON file")
    ap.add_argument("--n", type=int, default=1024, help="number of sequences")
    ap.add_argument("--nfe", type=int, default=256)
    ap.add_argument("--sc", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt)
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    ids = []
    while len(ids) < args.n:
        z = sample(model, min(args.batch_size, args.n - len(ids)), args.nfe, args.sc, gen)
        ids += decode(model, z, args.sc, tok.eos_token_id, tok.pad_token_id).tolist()
        print(f"[sample] {len(ids)}/{args.n}", flush=True)
    texts = [tok.decode(row, skip_special_tokens=True) for row in ids]
    with open(args.out, "w") as f:
        json.dump({"nfe": args.nfe, "sc": args.sc, "seed": args.seed,
                   "texts": texts, "token_ids": ids}, f)
    print(f"[sample] wrote {args.out}")


if __name__ == "__main__":
    main()
