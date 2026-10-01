"""Build the OpenWebText token cache used for training, and a held-out split.

Documents are tokenized with the T5Gemma-2 tokenizer (or the one given by
`--tokenizer`, e.g. google-t5/t5-small; a cache is tied to its tokenizer) and
packed into rows of at most `seq_len` tokens, in the format of ELF's
openwebtext-t5 dataset:
  - a row is filled paragraph by paragraph, running across document boundaries
    without any separator, and is closed with a single end-of-sequence token as
    soon as the next paragraph does not fit. Rows therefore end a little short
    of `seq_len` (typically 850-1020 tokens for 1024) and each ends in one EOS;
  - a paragraph longer than a whole row is cut into full rows without EOS, and
    its remainder continues the packing.
Rows are stored at their own length, like ELF's dataset. At training time
`encoders.collate` right-pads each batch to `seq_len`, and the padding positions
are masked out of the attention and of every loss.

The last `--heldout-docs` documents of OpenWebText never enter the training
cache. The first `--heldout-rows` rows packed from them form the held-out split,
the real-text reference of evaluate.py.

Writes <out> and <out>_heldout (Hugging Face datasets with one column, input_ids).

  python data.py --out data/owt
  python data.py --out data/owt --heldout-only     # only the held-out split, for evaluation
"""

import argparse
import re

from datasets import Dataset, Features, List, Value, load_dataset
from transformers import AutoTokenizer

from encoders import TOKENIZER

SLICE = 20_000      # documents tokenized at a time


class Packer:
    """Greedy packer. feed() takes the token ids of one paragraph and yields finished rows."""

    def __init__(self, seq_len, eos_id):
        self.seq_len, self.eos_id, self.buf = seq_len, eos_id, []

    def feed(self, ids):
        ids = list(ids)
        while ids:
            space = self.seq_len - 1 - len(self.buf)      # one slot is kept for the final EOS
            if len(ids) <= space:
                self.buf.extend(ids)
                ids = []
            elif not self.buf and len(ids) >= self.seq_len:
                yield ids[:self.seq_len]
                ids = ids[self.seq_len:]
            else:
                yield from self.close()

    def close(self):
        if self.buf:
            yield self.buf + [self.eos_id]
            self.buf = []


def pack(docs, tokenizer, seq_len, start, end, max_rows=None):
    """Yield packed rows from documents docs[start:end]."""
    packer = Packer(seq_len, tokenizer.eos_token_id)
    n = 0
    for s in range(start, end, SLICE):
        texts = docs[s:min(s + SLICE, end)]["text"]
        # Packing unit: a paragraph together with its trailing newlines.
        units = [u for t in texts for u in re.findall(r"[^\n]*\n+|[^\n]+$", t)]
        for ids in tokenizer(units, add_special_tokens=False)["input_ids"]:
            for row in packer.feed(ids):
                yield {"input_ids": row}
                n += 1
                if n == max_rows:
                    return
        print(f"[data] {min(s + SLICE, end) - start:,} documents, {n:,} rows", flush=True)
    for row in packer.close():
        yield {"input_ids": row}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output path of the training cache")
    ap.add_argument("--tokenizer", default=TOKENIZER, help="Hugging Face id of the tokenizer")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--heldout-docs", type=int, default=193769)
    ap.add_argument("--heldout-rows", type=int, default=1024)
    ap.add_argument("--max-docs", type=int, default=0,
                    help="use only the first documents for training (0 = all; for quick tests)")
    ap.add_argument("--heldout-only", action="store_true",
                    help="write only the held-out split (enough to evaluate released checkpoints)")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    docs = load_dataset("Skylion007/openwebtext", split="train")
    n_train = len(docs) - args.heldout_docs
    features = Features({"input_ids": List(Value("int32"))})
    out = args.out.rstrip("/")

    kwargs = dict(docs=docs, tokenizer=tokenizer, seq_len=args.seq_len)
    heldout = Dataset.from_generator(
        pack, features=features,
        gen_kwargs=dict(kwargs, start=n_train, end=len(docs), max_rows=args.heldout_rows))
    heldout.save_to_disk(out + "_heldout")
    print(f"[data] wrote {out}_heldout ({len(heldout):,} rows)", flush=True)
    if args.heldout_only:
        return

    train = Dataset.from_generator(
        pack, features=features,
        gen_kwargs=dict(kwargs, start=0, end=min(args.max_docs, n_train) if args.max_docs else n_train))
    train.save_to_disk(out)
    print(f"[data] wrote {out} ({len(train):,} rows)", flush=True)


if __name__ == "__main__":
    main()
