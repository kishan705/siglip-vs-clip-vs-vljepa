"""
Evaluate a model without training it.

Official pretrained weights, loaded into OUR from-scratch model (needs internet once):
    python eval.py --init_from google/siglip-base-patch16-224 --val_jsonl data/coco/val_5k.jsonl \
        --image_root /data/coco --zeroshot_cifar10 data/cifar
    python eval.py --init_from openai/clip-vit-base-patch32 --val_jsonl data/coco/val_5k.jsonl \
        --image_root /data/coco --zeroshot_cifar10 data/cifar

Your own trained run:
    python eval.py --checkpoint runs/siglip-ti16_sigmoid_bs256_s0/last.pt --val_jsonl data/coco/val_5k.jsonl \
        --image_root /data/coco
"""

import argparse
import json
import os

import torch

from vlm.data import build_tokenizer, read_jsonl
from vlm.evaluate import cifar10_zeroshot, retrieval_eval
from vlm.model import DualEncoder, ModelConfig


def load(args, device):
    if args.init_from:
        from vlm.convert import from_pretrained
        model = from_pretrained(args.init_from)
        tokenizer = build_tokenizer(args.tokenizer or args.init_from, model.c.context_len)
        return model.to(device), tokenizer

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_cfg = json.load(open(os.path.join(os.path.dirname(args.checkpoint), "config.json")))
    c = ModelConfig(**ck["config"])
    model = DualEncoder(c)
    loss = run_cfg["args"]["loss"]
    model.set_logit_params(loss)          # so the state dict has the right logit params
    model.load_state_dict(ck["model"])
    tok_name = args.tokenizer or run_cfg["args"]["tokenizer"]
    train_records = read_jsonl(run_cfg["args"]["train_jsonl"]) if tok_name == "word" else None
    return model.to(device), build_tokenizer(tok_name, c.context_len, train_records)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--init_from", help="HF model id, e.g. google/siglip-base-patch16-224")
    src.add_argument("--checkpoint", help="runs/<name>/last.pt from train.py")
    p.add_argument("--tokenizer", default=None, help="defaults to the model's own tokenizer")
    p.add_argument("--val_jsonl")
    p.add_argument("--image_root", default=".")
    p.add_argument("--zeroshot_cifar10", default=None, help="folder to download CIFAR-10 into")
    p.add_argument("--out", default=None, help="write results to this json file")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model, tokenizer = load(args, device)
    model.eval()
    amp = torch.float16 if device.type == "cuda" else None
    c = model.c
    print(f"{args.init_from or args.checkpoint}: {c.arch} arch, {model.num_params()/1e6:.1f}M params")

    res = {}
    if args.val_jsonl:
        res.update(retrieval_eval(model, tokenizer, read_jsonl(args.val_jsonl), args.image_root, device,
                                  c.image_size, c.arch, workers=args.workers, amp_dtype=amp))
    if args.zeroshot_cifar10:
        res["cifar10_zeroshot_top1"] = cifar10_zeroshot(model, tokenizer, args.zeroshot_cifar10, device,
                                                        c.image_size, c.arch, amp, args.workers)
    for k, v in res.items():
        print(f"  {k:>24}: {v:.2f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"source": args.init_from or args.checkpoint, **res}, f, indent=2)


if __name__ == "__main__":
    main()
