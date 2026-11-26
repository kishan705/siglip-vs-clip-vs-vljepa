"""
Convert standard caption datasets into the jsonl format used by train.py.

COCO 2017 (download train2017.zip, val2017.zip, annotations_trainval2017.zip from cocodataset.org)
    python prepare_data.py coco --root /data/coco --out data/coco
    -> train.jsonl (118k images), val_1k.jsonl (first 1000 val images, for quick evals during
       training), val_5k.jsonl (all 5000 val2017 images, for the final numbers)
    Note: papers usually report the "Karpathy" 5k test split; val2017 is a different 5k
    images, so compare your own runs with each other, not directly with paper tables.

Flickr30k with the Karpathy split (images + dataset_flickr30k.json)
    python prepare_data.py flickr30k --root /data/flickr30k --karpathy_json dataset_flickr30k.json \
        --out data/flickr30k
    -> train.jsonl (29k), val.jsonl (1k), test.jsonl (1k) -- the standard Flickr30k test split.
"""

import argparse
import json
import os
from collections import defaultdict


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records):>7} images, {sum(len(r['captions']) for r in records):>7} captions -> {path}")


def coco_split(root, split):
    ann = json.load(open(os.path.join(root, "annotations", f"captions_{split}.json")))
    files = {im["id"]: im["file_name"] for im in ann["images"]}
    caps = defaultdict(list)
    for a in ann["annotations"]:
        caps[a["image_id"]].append(a["caption"].strip())
    return [{"image": f"{split}/{files[i]}", "captions": caps[i]} for i in sorted(files) if caps[i]]


def prepare_coco(a):
    train = coco_split(a.root, "train2017")
    val = coco_split(a.root, "val2017")
    write_jsonl(train, os.path.join(a.out, "train.jsonl"))
    write_jsonl(val[:1000], os.path.join(a.out, "val_1k.jsonl"))
    write_jsonl(val, os.path.join(a.out, "val_5k.jsonl"))


def prepare_flickr(a):
    data = json.load(open(a.karpathy_json))
    by_split = defaultdict(list)
    for im in data["images"]:
        split = "train" if im["split"] in ("train", "restval") else im["split"]
        by_split[split].append({"image": os.path.join(a.images_dir, im["filename"]),
                                "captions": [s["raw"].strip() for s in im["sentences"]]})
    for split in ("train", "val", "test"):
        write_jsonl(by_split[split], os.path.join(a.out, f"{split}.jsonl"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="dataset", required=True)
    c = sub.add_parser("coco")
    c.add_argument("--root", required=True)
    c.add_argument("--out", default="data/coco")
    f = sub.add_parser("flickr30k")
    f.add_argument("--root", required=True, help="(only used to remind you to pass it as --image_root)")
    f.add_argument("--karpathy_json", required=True)
    f.add_argument("--images_dir", default="flickr30k-images", help="image folder, relative to --root")
    f.add_argument("--out", default="data/flickr30k")
    a = p.parse_args()
    prepare_coco(a) if a.dataset == "coco" else prepare_flickr(a)
