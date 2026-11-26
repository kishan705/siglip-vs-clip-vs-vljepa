"""
The two losses being compared. Inputs are L2-normalised embeddings:
img, txt: [B, D], where row i of img and row i of txt are a matching pair.

CLIP (softmax / InfoNCE)
    For each image, a B-way classification "which of the B captions is mine?"
    (and the same for each caption). The softmax normalises over the whole
    batch, so every logit depends on every other one: you need the full B x B
    matrix at once, and quality depends strongly on batch size.

SigLIP (pairwise sigmoid)
    Every one of the B*B (image, caption) pairs is an independent binary
    classification: label +1 on the diagonal, -1 everywhere else.
        loss = -1/B * sum_ij log sigmoid( z_ij * (t * <x_i, y_j> + b) ),  z_ij = +-1
    No normalisation across the batch -> pairs can be processed in chunks
    (see chunked_sigmoid_loss) and it works well at small batch sizes.
"""

import torch
import torch.nn.functional as F


def sigmoid_loss(img, txt, logit_scale, logit_bias):
    n = img.shape[0]
    logits = img @ txt.t() * logit_scale.exp() + logit_bias        # [B, B]
    labels = 2 * torch.eye(n, device=img.device, dtype=logits.dtype) - 1   # +1 diag, -1 off-diag
    return -F.logsigmoid(labels * logits).sum() / n


def softmax_loss(img, txt, logit_scale, logit_bias=None):
    logits = img @ txt.t() * logit_scale.exp()                     # [B, B]
    if logit_bias is not None:   # a constant shift; it cancels inside a softmax
        logits = logits + logit_bias
    targets = torch.arange(img.shape[0], device=img.device)
    # image->text (rows) and text->image (columns), averaged: symmetric InfoNCE
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) / 2


def chunked_sigmoid_loss(img, txt, logit_scale, logit_bias, chunk_size: int):
    """Exactly the same value as sigmoid_loss, computed one block of captions at a time.

    This is the single-GPU version of the SigLIP paper's multi-device trick: each
    device keeps its own images and the text embeddings are passed round the devices
    in a ring, so no device ever builds the full B x B matrix. It is possible only
    because the sigmoid loss has no batch-wide normalisation; with softmax you would
    need every logit in a row before you could compute any of them.
    """
    n = img.shape[0]
    total = img.new_zeros(())
    for start in range(0, n, chunk_size):
        t = txt[start:start + chunk_size]
        logits = img @ t.t() * logit_scale.exp() + logit_bias      # [B, chunk]
        labels = -torch.ones_like(logits)
        idx = torch.arange(t.shape[0], device=img.device)
        labels[start + idx, idx] = 1.0                             # the positives in this block
        total = total + (-F.logsigmoid(labels * logits).sum())
    return total / n


LOSSES = {"sigmoid": sigmoid_loss, "softmax": softmax_loss}
