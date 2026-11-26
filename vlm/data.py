"""
Datasets and tokenisers.

Data format (made by prepare_data.py): one JSON object per line, one line per image
    {"image": "train2017/000000391895.jpg", "captions": ["A man riding ...", "..."]}
Paths are relative to --image_root.

Training samples one random caption of the image each time it is seen;
evaluation uses every caption (for retrieval, an image has ~5 correct captions).
"""

from __future__ import annotations

import json
import os
import random
import re

import torch
from torch.utils.data import Dataset

SIGLIP_MEAN, SIGLIP_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def image_transforms(image_size: int, arch: str, train: bool):
    from torchvision import transforms as T
    mean, std = (CLIP_MEAN, CLIP_STD) if arch == "clip" else (SIGLIP_MEAN, SIGLIP_STD)
    if train:
        resize = [T.RandomResizedCrop(image_size, scale=(0.6, 1.0),
                                      interpolation=T.InterpolationMode.BICUBIC)]
    else:
        resize = [T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)]
    return T.Compose(resize + [T.ToTensor(), T.Normalize(mean, std)])


# ----------------------------------------------------------------------------
# Tokenisers
# ----------------------------------------------------------------------------

class HFTokenizer:
    """Wraps a Hugging Face tokenizer. SigLIP must be padded to exactly context_len."""

    def __init__(self, name: str, context_len: int):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.context_len = context_len
        self.vocab_size = len(self.tok)
        self.eos_token_id = self.tok.eos_token_id

    def __call__(self, texts: list[str]) -> torch.Tensor:
        return self.tok(texts, padding="max_length", truncation=True, max_length=self.context_len,
                        return_tensors="pt")["input_ids"]


class WordTokenizer:
    """Tiny word-level tokeniser built from the training captions.

    Only for the toy dataset and offline debugging: no download needed.
    ids: 0 = pad, 1 = eos, 2 = unknown word.
    """

    PAD, EOS, UNK = 0, 1, 2

    def __init__(self, captions: list[str], context_len: int, min_count: int = 1):
        counts: dict[str, int] = {}
        for c in captions:
            for w in self.split(c):
                counts[w] = counts.get(w, 0) + 1
        words = sorted(w for w, n in counts.items() if n >= min_count)
        self.vocab = {w: i + 3 for i, w in enumerate(words)}
        self.vocab_size = len(self.vocab) + 3
        self.context_len = context_len
        self.eos_token_id = self.EOS

    @staticmethod
    def split(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def __call__(self, texts: list[str]) -> torch.Tensor:
        out = torch.full((len(texts), self.context_len), self.PAD, dtype=torch.long)
        for i, t in enumerate(texts):
            ids = [self.vocab.get(w, self.UNK) for w in self.split(t)][: self.context_len - 1] + [self.EOS]
            out[i, : len(ids)] = torch.tensor(ids)
        return out


def build_tokenizer(name: str, context_len: int, train_records=None):
    if name == "word":
        caps = [c for r in train_records for c in r["captions"]]
        return WordTokenizer(caps, context_len)
    return HFTokenizer(name, context_len)


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------

class CaptionTrainDataset(Dataset):
    """Returns (image tensor, one random caption string)."""

    def __init__(self, records, image_root, transform):
        self.records, self.root, self.tf = records, image_root, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        from PIL import Image
        r = self.records[i]
        img = Image.open(os.path.join(self.root, r["image"])).convert("RGB")
        return self.tf(img), random.choice(r["captions"])


class ImageOnlyDataset(Dataset):
    def __init__(self, records, image_root, transform):
        self.records, self.root, self.tf = records, image_root, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.open(os.path.join(self.root, self.records[i]["image"])).convert("RGB"))


class SyntheticPairs(Dataset):
    """Random-noise images with random token captions, pair i always the same.

    Nothing to learn except memorisation -- used for the "can it overfit one batch"
    sanity check and for timing, with no files needed.
    """

    def __init__(self, n, image_size, context_len, vocab_size):
        self.n, self.s, self.L, self.V = n, image_size, context_len, vocab_size

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return (torch.randn(3, self.s, self.s, generator=g),
                torch.randint(3, self.V, (self.L,), generator=g))


class TrainCollate:
    """Stack images and tokenise the captions in the DataLoader workers."""

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, batch):
        images = torch.stack([b[0] for b in batch])
        if isinstance(batch[0][1], torch.Tensor):           # synthetic: already ids
            return images, torch.stack([b[1] for b in batch])
        return images, self.tok([b[1] for b in batch])
