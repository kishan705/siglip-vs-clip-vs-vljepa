"""
Evaluation used for the thesis comparison:

1. Zero-shot image-text retrieval (COCO / Flickr30k): Recall@1/5/10 both ways.
2. Zero-shot classification (e.g. CIFAR-10): turn class names into captions
   ("a photo of a dog."), embed them, pick the closest one for each image.

Both only use the embeddings, so they work the same for SigLIP, CLIP and
(next semester) VL-JEPA.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .data import ImageOnlyDataset, image_transforms


@torch.no_grad()
def embed_images(model, loader, device, amp_dtype=None):
    out = []
    for images in loader:
        images = images[0] if isinstance(images, (list, tuple)) else images
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out.append(model.encode_image(images.to(device, non_blocking=True)).float())
    return torch.cat(out)


@torch.no_grad()
def embed_texts(model, tokenizer, texts, device, batch_size=512, amp_dtype=None):
    out = []
    for i in range(0, len(texts), batch_size):
        ids = tokenizer(texts[i:i + batch_size]).to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out.append(model.encode_text(ids).float())
    return torch.cat(out)


def recall_at_k(img_emb, txt_emb, txt_to_img, ks=(1, 5, 10)):
    """img_emb [N_img, D], txt_emb [N_txt, D], txt_to_img[j] = index of caption j's image."""
    txt_to_img = torch.as_tensor(txt_to_img, device=img_emb.device)
    sim = txt_emb @ img_emb.t()                                   # [N_txt, N_img]
    res = {}
    # text -> image: is the caption's own image among the top-k images?
    topk_img = sim.topk(max(ks), dim=1).indices                   # [N_txt, K]
    hit = topk_img == txt_to_img[:, None]
    for k in ks:
        res[f"t2i_R@{k}"] = 100 * hit[:, :k].any(1).float().mean().item()
    # image -> text: is ANY of the image's captions among the top-k captions?
    topk_txt = sim.t().topk(max(ks), dim=1).indices               # [N_img, K]
    owner = txt_to_img[topk_txt]                                  # image index of each retrieved caption
    hit = owner == torch.arange(img_emb.shape[0], device=img_emb.device)[:, None]
    for k in ks:
        res[f"i2t_R@{k}"] = 100 * hit[:, :k].any(1).float().mean().item()
    res["mean_R"] = sum(v for k, v in res.items()) / len(res)
    return res


def retrieval_eval(model, tokenizer, records, image_root, device, image_size, arch,
                   batch_size=256, workers=4, amp_dtype=None):
    model.eval()
    ds = ImageOnlyDataset(records, image_root, image_transforms(image_size, arch, train=False))
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda")
    img_emb = embed_images(model, loader, device, amp_dtype)
    texts, owner = [], []
    for i, r in enumerate(records):
        texts += r["captions"]
        owner += [i] * len(r["captions"])
    txt_emb = embed_texts(model, tokenizer, texts, device, amp_dtype=amp_dtype)
    return recall_at_k(img_emb, txt_emb, owner)


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
TEMPLATES = ["a photo of a {}.", "a blurry photo of a {}.", "a photo of the small {}.",
             "a low resolution photo of a {}.", "a photo of the large {}."]


@torch.no_grad()
def zeroshot_classify(model, tokenizer, image_loader, labels, classnames, device,
                      templates=TEMPLATES, amp_dtype=None):
    model.eval()
    class_emb = []
    for name in classnames:
        e = embed_texts(model, tokenizer, [t.format(name) for t in templates], device, amp_dtype=amp_dtype)
        class_emb.append(torch.nn.functional.normalize(e.mean(0), dim=-1))   # average over prompts
    class_emb = torch.stack(class_emb)                                      # [C, D]
    img_emb = embed_images(model, image_loader, device, amp_dtype)
    pred = (img_emb @ class_emb.t()).argmax(1).cpu()
    return 100 * (pred == torch.as_tensor(labels)).float().mean().item()


def cifar10_zeroshot(model, tokenizer, root, device, image_size, arch, amp_dtype=None, workers=4):
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root, train=False, download=True, transform=image_transforms(image_size, arch, train=False))
    loader = DataLoader(ds, batch_size=256, num_workers=workers)
    return zeroshot_classify(model, tokenizer, loader, ds.targets, CIFAR10_CLASSES, device, amp_dtype=amp_dtype)
