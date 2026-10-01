# The metrics are derived from ELF-pytorch (https://github.com/Ugness/ELF-pytorch).
# Copyright (c) 2026 ELF authors. MIT License (see LICENSE).
"""Evaluate generated text, or real text as a reference.

Texts are re-tokenized with GPT-2. Reported:
  Gen. PPL   perplexity under GPT-2-Large
  entropy    unigram entropy of each sequence (nats), averaged
  MAUVE      optional, against held-out real text

  python evaluate.py --samples samples.json
  python evaluate.py --real data/owt_heldout --n 1024
  python evaluate.py --samples samples.json --mauve data/owt_heldout
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from encoders import TOKENIZER

JUDGE = "gpt2-large"
MAX_LENGTH = 1024
MAUVE_LENGTH = 256   # MAUVE is computed on the first 256 tokens, as in the paper
BATCH_SIZE = 16


@torch.no_grad()
def perplexity_and_entropy(texts, device="cuda"):
    tok = AutoTokenizer.from_pretrained(JUDGE)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.float32).to(device).eval()
    enc = tok(texts, return_tensors="np", return_attention_mask=True, truncation=True,
              padding=True, max_length=MAX_LENGTH)
    ids_all, mask_all = enc["input_ids"], enc["attention_mask"]

    nll, count = 0.0, 0.0
    for i in range(0, len(texts), BATCH_SIZE):
        ids = torch.from_numpy(ids_all[i:i + BATCH_SIZE]).long().to(device)
        mask = torch.from_numpy(mask_all[i:i + BATCH_SIZE]).long().to(device)
        logits = model(ids, attention_mask=mask).logits[:, :-1]
        targets = ids[:, 1:]
        nlls = (torch.logsumexp(logits, dim=-1)
                - logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)).float()
        # Padding is the EOS token: count every token up to and including the first EOS.
        is_eos = ids == tok.eos_token_id
        first_eos = torch.cumsum(is_eos.long(), dim=-1) == 1
        valid = (first_eos[:, 1:] | ~is_eos[:, 1:]).float()
        nll += float((nlls * valid).double().sum())
        count += float(valid.sum())

    entropies = []
    for ids, mask in zip(ids_all, mask_all):
        _, counts = np.unique(ids[:int(mask.sum())], return_counts=True)
        p = counts.astype(np.float64) / counts.sum()
        entropies.append(float(-np.sum(p * np.log(p + 1e-10))))
    return math.exp(nll / max(count, 1e-8)), float(np.mean(entropies))


def real_texts(path, n, tokenizer=TOKENIZER):
    """Decode the first n rows of a token cache built by data.py."""
    if not os.path.isdir(path):
        raise SystemExit(f"{path} not found: build the held-out split first with "
                         f"`python data.py --out {path.removesuffix('_heldout')} --heldout-only`")
    tok = AutoTokenizer.from_pretrained(tokenizer)
    rows = load_from_disk(path).select(range(n))["input_ids"]
    return [tok.decode(row[:MAX_LENGTH], skip_special_tokens=True) for row in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", help="JSON file written by sample.py")
    ap.add_argument("--real", help="token cache of real text to evaluate instead of samples")
    ap.add_argument("--n", type=int, default=1024, help="number of real sequences")
    ap.add_argument("--mauve", help="token cache of real text used as the MAUVE reference")
    ap.add_argument("--tokenizer", default=TOKENIZER,
                    help="tokenizer the real-text caches were built with (data.py --tokenizer)")
    args = ap.parse_args()
    assert bool(args.samples) != bool(args.real), "give either --samples or --real"

    texts = (real_texts(args.real, args.n, args.tokenizer) if args.real
             else json.load(open(args.samples))["texts"])
    texts = [t for t in texts if t.strip()]
    ppl, entropy = perplexity_and_entropy(texts)
    result = {"n": len(texts), "gen_ppl": round(ppl, 2), "entropy": round(entropy, 3)}
    if args.mauve:
        import mauve
        refs = real_texts(args.mauve, len(texts), args.tokenizer)
        result["mauve"] = round(float(mauve.compute_mauve(
            p_text=texts, q_text=refs, device_id=0, verbose=False,
            max_text_length=MAUVE_LENGTH).mauve), 4)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
