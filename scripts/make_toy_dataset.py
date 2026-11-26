"""
Generate a tiny image-caption dataset of coloured shapes, e.g. "a small red circle at the top left".

It downloads nothing and trains on a laptop CPU in a few minutes, so you can test the
whole pipeline (jsonl -> images -> tokenizer -> training -> retrieval eval) before
spending GPU time on COCO. Retrieval should climb far above chance; if it doesn't,
something is broken.

    python scripts/make_toy_dataset.py --out data/toy
    python train.py --preset tiny-test --image_size 64 --tokenizer word --train_jsonl data/toy/train.jsonl \
        --val_jsonl data/toy/val.jsonl --image_root data/toy --batch_size 64 --epochs 15 \
        --lr 2e-3 --eval_every 200 --workers 2 --name toy_sigmoid
"""

import argparse
import json
import os
import random

from PIL import Image, ImageDraw

COLORS = {"red": (220, 40, 40), "green": (40, 180, 60), "blue": (50, 80, 220),
          "yellow": (230, 210, 40), "purple": (150, 60, 190), "orange": (240, 140, 30)}
SHAPES = ["circle", "square", "triangle", "cross"]
PLACES = {"top left": (0, 0), "top right": (1, 0), "bottom left": (0, 1), "bottom right": (1, 1)}
SIZES = {"small": 0.22, "large": 0.42}
TEMPLATES = ["a {size} {color} {shape} at the {place}",
             "{size} {shape} , {color} , {place}",
             "there is a {color} {shape} in the {place} and it is {size}"]


def draw(shape, color, place, size, S=64, rng=random):
    img = Image.new("RGB", (S, S), (20, 20, 20))
    d = ImageDraw.Draw(img)
    r = SIZES[size] * S / 2
    qx, qy = PLACES[place]
    cx = (qx + 0.5) * S / 2 + rng.uniform(-3, 3)
    cy = (qy + 0.5) * S / 2 + rng.uniform(-3, 3)
    col = COLORS[color]
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=col)
    else:
        w = r / 3
        d.rectangle([cx - r, cy - w, cx + r, cy + w], fill=col)
        d.rectangle([cx - w, cy - r, cx + w, cy + r], fill=col)
    return img


def make(out, n, split, rng):
    os.makedirs(os.path.join(out, split), exist_ok=True)
    recs = []
    for i in range(n):
        attrs = dict(shape=rng.choice(SHAPES), color=rng.choice(list(COLORS)),
                     place=rng.choice(list(PLACES)), size=rng.choice(list(SIZES)))
        path = f"{split}/{i:05d}.png"
        draw(**attrs, rng=rng).save(os.path.join(out, path))
        recs.append({"image": path, "captions": [t.format(**attrs) for t in TEMPLATES]})
    with open(os.path.join(out, f"{split}.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"{split}: {n} images -> {out}/{split}.jsonl")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/toy")
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_val", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = random.Random(a.seed)
    make(a.out, a.n_train, "train", rng)
    make(a.out, a.n_val, "val", rng)
