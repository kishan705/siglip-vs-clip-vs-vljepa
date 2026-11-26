"""
Train a SigLIP / CLIP dual encoder.

Examples
--------
# 0. Sanity check on CPU in ~1 min: can the model memorise 64 random pairs?
python train.py --data synthetic --preset tiny-test --synthetic_size 64 --batch_size 64 \
    --max_steps 300 --lr 3e-3 --eval_every 100 --name overfit

# 1. Train from scratch on COCO (after prepare_data.py), SigLIP loss
python train.py --preset siglip-s16 --loss sigmoid --train_jsonl data/coco/train.jsonl \
    --val_jsonl data/coco/val_1k.jsonl --image_root /path/to/coco --batch_size 256 \
    --total_samples 3000000 --seed 0 --name s16_sigmoid_bs256_s0

# 2. Same run with CLIP's softmax loss -- the only thing that changes is --loss
python train.py ... --loss softmax --name s16_softmax_bs256_s0

# 3. SigLiT (paper Sec. 4): frozen pretrained SigLIP image tower, text tower from scratch
python train.py --init_from google/siglip-base-patch16-224 --lock_vision --reinit_text ...

Everything needed to reproduce a run is written to runs/<name>/config.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader

from vlm.data import (CaptionTrainDataset, SyntheticPairs, TrainCollate, build_tokenizer,
                      image_transforms, read_jsonl)
from vlm.evaluate import cifar10_zeroshot, recall_at_k, retrieval_eval
from vlm.losses import chunked_sigmoid_loss, sigmoid_loss, softmax_loss
from vlm.model import PRESETS, DualEncoder, get_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("model")
    g.add_argument("--preset", default="siglip-s16", choices=list(PRESETS))
    g.add_argument("--arch", default=None, choices=["siglip", "clip"], help="override the preset's family")
    g.add_argument("--loss", default="sigmoid", choices=["sigmoid", "softmax"])
    g.add_argument("--image_size", type=int, default=None)
    g.add_argument("--init_from", default=None, help="HF id to load weights from (needs internet)")
    g.add_argument("--lock_vision", action="store_true", help="freeze the image tower (LiT / SigLiT)")
    g.add_argument("--reinit_text", action="store_true", help="with --init_from: text tower from scratch")
    g.add_argument("--chunk_size", type=int, default=0, help=">0: chunked sigmoid loss (same value)")
    g.add_argument("--compile", action="store_true")

    g = p.add_argument_group("data")
    g.add_argument("--data", default="jsonl", choices=["jsonl", "synthetic"])
    g.add_argument("--train_jsonl")
    g.add_argument("--val_jsonl")
    g.add_argument("--image_root", default=".")
    g.add_argument("--tokenizer", default="google/siglip-base-patch16-224",
                   help="HF tokenizer id, or 'word' for the offline word-level tokenizer")
    g.add_argument("--synthetic_size", type=int, default=1024)
    g.add_argument("--zeroshot_cifar10", default=None, help="folder to download CIFAR-10 into for eval")

    g = p.add_argument_group("optimisation")
    g.add_argument("--batch_size", type=int, default=256)
    g.add_argument("--total_samples", type=int, default=None,
                   help="train for this many image-text pairs (keeps compute equal across batch sizes)")
    g.add_argument("--epochs", type=int, default=10, help="used only if --total_samples is not set")
    g.add_argument("--max_steps", type=int, default=None, help="train exactly this many steps (overrides the above)")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--wd", type=float, default=0.1)
    g.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    g.add_argument("--grad_clip", type=float, default=1.0)

    g = p.add_argument_group("run")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--deterministic", action="store_true", help="slower, bit-reproducible on one GPU")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--out", default="runs")
    g.add_argument("--name", default=None)
    g.add_argument("--log_every", type=int, default=20)
    g.add_argument("--eval_every", type=int, default=1000)
    g.add_argument("--save_every", type=int, default=1000)
    g.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pt")
    g.add_argument("--wandb", action="store_true")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------
# Reproducibility helpers
# ----------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    # each DataLoader worker gets a distinct but reproducible seed (random caption choice, augmentation)
    s = torch.initial_seed() % 2**32
    random.seed(s)
    try:
        import numpy as np
        np.random.seed(s)
    except ImportError:
        pass


def git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, stderr=subprocess.DEVNULL)
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=here, stderr=subprocess.DEVNULL) != 0
        return sha.decode().strip() + ("-dirty" if dirty else "")
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Model / optimiser construction
# ----------------------------------------------------------------------------

def build_model(args, vocab_size, eos_token_id):
    if args.init_from:
        from vlm.convert import from_pretrained
        model = from_pretrained(args.init_from)
        if args.reinit_text:
            fresh = DualEncoder(model.c)
            model.text.load_state_dict(fresh.text.state_dict())
            model.text_proj.load_state_dict(fresh.text_proj.state_dict())
        family = "sigmoid" if model.c.arch == "siglip" else "softmax"
        if args.reinit_text or family != args.loss:
            model.set_logit_params(args.loss)
        return model

    overrides = dict(vocab_size=vocab_size, eos_token_id=eos_token_id)
    if args.arch:
        overrides["arch"] = args.arch
    if args.image_size:
        overrides["image_size"] = args.image_size
    model = DualEncoder(get_config(args.preset, **overrides))
    model.set_logit_params(args.loss)
    return model


def param_groups(model, wd):
    """No weight decay on biases, LayerNorms, embeddings-as-parameters and the logit params."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "pos_embed" in n or "probe" in n or "cls_token" in n or "logit_" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def cosine_with_warmup(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    args.name = args.name or f"{args.preset}_{args.loss}_bs{args.batch_size}_s{args.seed}"
    run_dir = os.path.join(args.out, args.name)
    os.makedirs(run_dir, exist_ok=True)
    set_seed(args.seed, args.deterministic)
    device = torch.device(args.device)

    # ---- tokenizer -> model -> data (the model's shapes decide the image size) ----
    preset = PRESETS[args.preset]
    ctx = preset.context_len
    val_records = None
    if args.data == "synthetic":
        tokenizer = None
        vocab_size, eos = preset.vocab_size, preset.eos_token_id
    else:
        train_records = read_jsonl(args.train_jsonl)
        tokenizer = build_tokenizer(args.tokenizer, ctx, train_records)
        vocab_size, eos = tokenizer.vocab_size, tokenizer.eos_token_id
        if args.val_jsonl:
            val_records = read_jsonl(args.val_jsonl)

    model = build_model(args, vocab_size, eos).to(device)
    c = model.c
    if tokenizer is not None and tokenizer.context_len != c.context_len:
        tokenizer.context_len = c.context_len     # e.g. --init_from a model with another text length

    if args.data == "synthetic":
        train_ds = SyntheticPairs(args.synthetic_size, c.image_size, c.context_len, c.vocab_size)
    else:
        train_ds = CaptionTrainDataset(train_records, args.image_root,
                                       image_transforms(c.image_size, c.arch, train=True))

    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        collate_fn=TrainCollate(tokenizer), worker_init_fn=seed_worker, generator=gen,
                        persistent_workers=args.workers > 0)
    if len(loader) == 0:
        sys.exit(f"Dataset has {len(train_ds)} pairs: smaller than one batch of {args.batch_size}.")

    # ---- model setup -----------------------------------------------------
    if args.lock_vision:
        for mod in (model.vision, model.vision_proj):
            if mod is not None:
                mod.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {c.arch} arch, {args.loss} loss | {model.num_params()/1e6:.1f}M params "
          f"({n_train/1e6:.1f}M trainable) | device {device}")

    steps_per_epoch = len(loader)
    total_steps = (args.total_samples // args.batch_size) if args.total_samples else args.epochs * steps_per_epoch
    if args.max_steps:
        total_steps = args.max_steps
    warmup = max(1, int(args.warmup * total_steps))

    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, total_steps, warmup))

    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16   # T4 -> fp16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    fwd_model = torch.compile(model) if args.compile else model

    # ---- resume / record config -----------------------------------------------
    step = 0
    ckpt_path = os.path.join(run_dir, "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        step = ck["step"]
        torch.set_rng_state(ck["rng_cpu"]); random.setstate(ck["rng_py"])
        print(f"resumed from step {step}")

    record = {
        "args": vars(args), "model_config": c.to_dict(), "total_steps": total_steps,
        "params_M": model.num_params() / 1e6, "git_commit": git_commit(),
        "torch": torch.__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(record, f, indent=2)

    wandb = None
    if args.wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project="siglip-vs-clip-vs-vljepa", name=args.name, config=record, resume="allow")
    log_f = open(os.path.join(run_dir, "metrics.jsonl"), "a")

    def log(d):
        d = {"step": step, **d}
        log_f.write(json.dumps(d) + "\n"); log_f.flush()
        if wandb:
            wandb.log(d, step=step)

    def evaluate():
        if args.data == "synthetic":
            # retrieval over the (memorised) synthetic pairs
            model.eval()
            ev = DataLoader(train_ds, batch_size=256, collate_fn=TrainCollate(None))
            img, txt = [], []
            with torch.no_grad():
                for x, ids in ev:
                    img.append(model.encode_image(x.to(device)).float())
                    txt.append(model.encode_text(ids.to(device)).float())
            res = recall_at_k(torch.cat(img), torch.cat(txt), list(range(len(train_ds))))
        elif val_records:
            res = retrieval_eval(model, tokenizer, val_records, args.image_root, device, c.image_size,
                                 c.arch, workers=args.workers, amp_dtype=amp_dtype)
        else:
            return {}
        model.train()
        return res

    # ---- train loop -------------------------------------------------------
    model.train()
    if args.lock_vision:
        model.vision.eval()
    t_last, seen_last = time.time(), 0
    data_iter = iter(loader)
    while step < total_steps:
        try:
            images, ids = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, ids = next(data_iter)
        images, ids = images.to(device, non_blocking=True), ids.to(device, non_blocking=True)

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            if args.lock_vision:
                with torch.no_grad():
                    img = fwd_model.encode_image(images)
            else:
                img = fwd_model.encode_image(images)
            txt = fwd_model.encode_text(ids)
        img, txt = img.float(), txt.float()          # compute the loss in fp32

        if args.loss == "sigmoid":
            if args.chunk_size:
                loss = chunked_sigmoid_loss(img, txt, model.logit_scale, model.logit_bias, args.chunk_size)
            else:
                loss = sigmoid_loss(img, txt, model.logit_scale, model.logit_bias)
        else:
            loss = softmax_loss(img, txt, model.logit_scale)

        if not torch.isfinite(loss):
            sys.exit(f"loss became {loss.item()} at step {step}: lower --lr or check the data")

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if args.loss == "softmax":   # CLIP paper: cap the logit scale at 100
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100))
        step += 1
        seen_last += images.shape[0]

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t_last
            d = {"loss": loss.item(), "lr": sched.get_last_lr()[0], "grad_norm": grad_norm.item(),
                 "t": model.logit_scale.exp().item(),
                 "b": model.logit_bias.item() if model.logit_bias is not None else None,
                 "samples_per_s": seen_last / dt}
            log(d)
            print(f"step {step:>6}/{total_steps}  loss {d['loss']:.4f}  t {d['t']:.1f}  "
                  f"b {d['b'] if d['b'] is None else round(d['b'], 2)}  lr {d['lr']:.2e}  "
                  f"{d['samples_per_s']:.0f} samples/s")
            t_last, seen_last = time.time(), 0

        if step % args.eval_every == 0 and step < total_steps:   # the final eval runs after the loop
            res = evaluate()
            if res:
                log({f"val/{k}": v for k, v in res.items()})
                print("  eval:", {k: round(v, 2) for k, v in res.items()})

        if step % args.save_every == 0 or step == total_steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "step": step, "config": c.to_dict(),
                        "rng_cpu": torch.get_rng_state(), "rng_py": random.getstate()}, ckpt_path)

    # ---- final results ----------------------------------------------------
    final = evaluate()
    if args.zeroshot_cifar10 and tokenizer is not None:
        final["cifar10_zeroshot_top1"] = cifar10_zeroshot(model, tokenizer, args.zeroshot_cifar10, device,
                                                          c.image_size, c.arch, amp_dtype, args.workers)
    final.update(final_loss=loss.item(), steps=step, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(final, f, indent=2)
    print("final:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in final.items()})
    log_f.close()
    if wandb:
        wandb.finish()
    return final


if __name__ == "__main__":
    main()
