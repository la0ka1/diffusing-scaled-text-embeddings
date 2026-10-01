"""Frozen text encoders: T5Gemma-2, the distilled student, or another T5-family
encoder-decoder from Hugging Face (e.g. google-t5/t5-small, the encoder of ELF).

An encoder maps token ids (B, L) to per-token embeddings (B, L, dim):

    enc = Encoder(STUDENT)            # released student; or a model id, or a best.pt
    ids = enc.tokenizer(texts, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    z = enc(ids)                      # (B, L, dim)

ELF is trained on (z - mean) / std with the two scalars of `Encoder.mean_std`.
"""

import copy
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

TEACHER = "google/t5gemma-2-270m-270m"
ENCODE_CHUNK = 16
NORM_ROWS = 512      # training sequences used to estimate the embedding mean and std

HF_USER = "la0ka1"   # each released checkpoint lives in its own repository, HF_USER/<name>
STUDENT = "T5Gemma-2-270M-OWTdistilled"
RELEASED = (STUDENT, "ELF-B-T5Gemma2", "ELF-M-T5Gemma2", "ELF-B-T5Gemma2distilled", "ELF-M-T5Gemma2distilled")
TOKENIZER = f"{HF_USER}/{STUDENT}"  # the T5Gemma-2 tokenizer, redistributed with the checkpoints (no gated access needed)


def checkpoint_path(name):
    """Local path of a checkpoint. `name` is a file, or the name of a released
    checkpoint, which is downloaded from its Hugging Face repository (and cached) on first use."""
    stem = name.removesuffix(".pt")
    if os.path.exists(name) or stem not in RELEASED:
        return name
    from huggingface_hub import hf_hub_download
    return hf_hub_download(f"{HF_USER}/{stem}", stem + ".pt")


def collate(seq_len, pad_id):
    """Truncate to seq_len and right-pad with pad_id."""
    def fn(batch):
        out = torch.full((len(batch), seq_len), pad_id, dtype=torch.long)
        for i, row in enumerate(batch):
            ids = row["input_ids"][:seq_len]
            out[i, :len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        return out
    return fn


def text_encoder(seq2seq):
    """The text encoder stack of a T5Gemma-2 sequence-to-sequence model."""
    enc = seq2seq.get_encoder()
    return getattr(enc, "text_model", enc)


def truncate_layers(encoder, keep):
    """Keep only the layers with indices `keep`."""
    if hasattr(encoder.config, "layer_types"):
        encoder.config.layer_types = [encoder.config.layer_types[i] for i in keep]
    encoder.layers = torch.nn.ModuleList([encoder.layers[i] for i in keep])
    encoder.config.num_hidden_layers = len(keep)
    return encoder


def init_student(teacher, num_layers):
    """Student = copy of the teacher encoder reduced to `num_layers` evenly spaced
    layers (9 of 18 keeps layers 0, 2, 4, 6, 8, 11, 13, 15, 17). The embedding
    table stays frozen."""
    student = copy.deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(True)
    keep = torch.linspace(0, len(teacher.layers) - 1, num_layers).round().long().tolist()
    truncate_layers(student, keep)
    student.get_input_embeddings().weight.requires_grad_(False)
    return student, keep


class Encoder:
    """Frozen encoder. `name` is a Hugging Face model id (T5Gemma-2, T5-small, ...),
    the path of a student checkpoint written by distill.py, or STUDENT for the
    released one. The token cache must have been built with the same tokenizer
    (data.py --tokenizer)."""

    def __init__(self, name, device="cuda"):
        assert name == STUDENT or name not in RELEASED, f"{name} is an ELF model, not an encoder"
        if name.endswith(".pt") or name == STUDENT:
            ckpt = torch.load(checkpoint_path(name), map_location="cpu", weights_only=True)
            teacher = ckpt.get("teacher", TEACHER)
            self.tokenizer_name = TOKENIZER if teacher == TEACHER else teacher
            model = text_encoder(AutoModelForSeq2SeqLM.from_pretrained(teacher)).float()
            truncate_layers(model, ckpt["keep"])
            model.load_state_dict(ckpt["student_sd"])
        else:
            self.tokenizer_name = TOKENIZER if name == TEACHER else name
            model = text_encoder(AutoModel.from_pretrained(name))   # for T5Gemma-2: without the vision tower
        self.model = model.to(device).eval().requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.pad_id = self.tokenizer.pad_token_id
        self.dim = self.model.get_input_embeddings().weight.shape[1]
        self.device = device

    @torch.no_grad()
    def __call__(self, ids):
        out = []
        for i in range(0, ids.size(0), ENCODE_CHUNK):
            sub = ids[i:i + ENCODE_CHUNK]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = self.model(input_ids=sub, attention_mask=(sub != self.pad_id).long())
            out.append(h.last_hidden_state.float())
        return torch.cat(out, 0)

    @torch.no_grad()
    def mean_std(self, dataset, seq_len):
        """Global scalar mean and std of the embeddings (padding included) over the
        first NORM_ROWS training sequences. ELF is trained on (embedding - mean) / std."""
        rows = dataset.select(range(min(NORM_ROWS, len(dataset))))
        loader = DataLoader(rows, batch_size=128, collate_fn=collate(seq_len, self.pad_id))
        z = torch.cat([self(b.to(self.device)) for b in loader], 0).reshape(-1, self.dim)
        return float(z.mean()), float(z.std().clamp_min(1e-6))
