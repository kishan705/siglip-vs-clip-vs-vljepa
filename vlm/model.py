"""
SigLIP and CLIP dual encoders, written from scratch in plain PyTorch.

Both models are the same idea: an image tower and a text tower map their inputs
into one shared embedding space, and a loss pulls matching (image, caption)
pairs together. They differ in a handful of architectural details, and those
details are switches on ModelConfig so you can see (and ablate) each one:

                      SigLIP (Zhai et al. 2023)        CLIP (Radford et al. 2021)
  image pooling       MAP head (attention pooling)     [CLS] token
  pre-LayerNorm       no                               yes (after embeddings)
  patch conv bias     yes                              no
  text attention      bidirectional (no mask)          causal mask
  text pooling        last position (fixed length 64)  position of the EOS token
  projections         text head only (with bias)       image + text, no bias
  activation          GELU (tanh approx.)              QuickGELU
  LayerNorm eps       1e-6                             1e-5
  loss                pairwise sigmoid + bias b        softmax InfoNCE (no bias)

The losses live in losses.py so you can also train "SigLIP architecture with
the CLIP loss" and vice versa -- the cleanest way to isolate the effect of the
loss, which is what the SigLIP paper itself does.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    arch: str = "siglip"            # "siglip" | "clip": which family's switches to use

    # image tower (a ViT)
    image_size: int = 224
    patch_size: int = 16
    v_width: int = 768
    v_depth: int = 12
    v_heads: int = 12
    v_mlp: int = 3072

    # text tower (a transformer encoder)
    vocab_size: int = 32000
    context_len: int = 64
    t_width: int = 768
    t_depth: int = 12
    t_heads: int = 12
    t_mlp: int = 3072

    embed_dim: int = 768            # size of the shared image-text space

    # architectural switches (filled in from `arch` by __post_init__ unless set)
    act: str | None = None          # "gelu_tanh" | "quick_gelu" | "gelu"
    ln_eps: float | None = None
    vision_pool: str | None = None  # "map" | "cls"
    vision_pre_norm: bool | None = None
    patch_bias: bool | None = None
    vision_proj: bool | None = None  # extra linear projection after the image tower
    text_causal: bool | None = None
    text_pool: str | None = None    # "last" | "eos"
    proj_bias: bool | None = None
    eos_token_id: int = 1           # only used when text_pool == "eos"

    fused_attn: bool = True         # True: F.scaled_dot_product_attention, False: explicit math

    def __post_init__(self):
        defaults = {
            "siglip": dict(act="gelu_tanh", ln_eps=1e-6, vision_pool="map", vision_pre_norm=False,
                           patch_bias=True, vision_proj=False, text_causal=False, text_pool="last",
                           proj_bias=True),
            "clip": dict(act="quick_gelu", ln_eps=1e-5, vision_pool="cls", vision_pre_norm=True,
                         patch_bias=False, vision_proj=True, text_causal=True, text_pool="eos",
                         proj_bias=False),
        }[self.arch]
        for k, v in defaults.items():
            if getattr(self, k) is None:
                setattr(self, k, v)
        if not self.vision_proj and self.embed_dim != self.v_width:
            raise ValueError("Without a vision projection the shared space is the ViT width: "
                             f"set embed_dim={self.v_width} (got {self.embed_dim}).")

    def to_dict(self):
        return asdict(self)


PRESETS: dict[str, ModelConfig] = {
    # Same shapes as google/siglip-base-patch16-224 -> can load the official weights.
    "siglip-b16": ModelConfig(arch="siglip"),
    # Same shapes as openai/clip-vit-base-patch32 -> can load the official weights.
    "clip-b32": ModelConfig(arch="clip", patch_size=32, vocab_size=49408, context_len=77,
                            t_width=512, t_heads=8, t_mlp=2048, embed_dim=512, eos_token_id=49407),
    # Small models you can actually train from scratch on one T4 / free Colab GPU.
    "siglip-s16": ModelConfig(arch="siglip", v_width=384, v_heads=6, v_mlp=1536,
                              t_width=384, t_heads=6, t_mlp=1536, embed_dim=384),
    "siglip-ti16": ModelConfig(arch="siglip", v_width=192, v_heads=3, v_mlp=768,
                               t_width=192, t_heads=3, t_mlp=768, embed_dim=192),
    # Toy size for unit tests and CPU smoke runs.
    "tiny-test": ModelConfig(arch="siglip", image_size=32, patch_size=8, v_width=64, v_depth=2,
                             v_heads=4, v_mlp=128, vocab_size=100, context_len=16, t_width=64,
                             t_depth=2, t_heads=4, t_mlp=128, embed_dim=64),
}


def get_config(name: str, **overrides) -> ModelConfig:
    base = PRESETS[name]
    if overrides:
        # re-run __post_init__ so arch-dependent switches follow an `arch` override
        fields = {k: v for k, v in asdict(base).items()}
        if "arch" in overrides:
            for k in ("act", "ln_eps", "vision_pool", "vision_pre_norm", "patch_bias",
                      "vision_proj", "text_causal", "text_pool", "proj_bias"):
                fields[k] = None
        fields.update(overrides)
        return ModelConfig(**fields)
    return replace(base)


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------

def activation(name: str):
    if name == "gelu_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "quick_gelu":
        return lambda x: x * torch.sigmoid(1.702 * x)
    if name == "gelu":
        return F.gelu
    raise ValueError(name)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, act: str):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = activation(act)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """Multi-head attention. Self-attention by default; pass `kv` for cross-attention."""

    def __init__(self, dim: int, heads: int, fused: bool = True):
        super().__init__()
        assert dim % heads == 0, "width must be divisible by heads"
        self.heads, self.head_dim, self.fused = heads, dim // heads, fused
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x, kv=None, causal: bool = False):
        kv = x if kv is None else kv
        B, Lq, D = x.shape
        Lk = kv.shape[1]
        # [B, L, D] -> [B, heads, L, head_dim]
        q = self.q(x).view(B, Lq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(kv).view(B, Lk, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(kv).view(B, Lk, self.heads, self.head_dim).transpose(1, 2)

        if self.fused:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        else:
            # the same thing, written out: softmax(Q K^T / sqrt(d)) V
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # [B, h, Lq, Lk]
            if causal:
                future = torch.ones(Lq, Lk, dtype=torch.bool, device=x.device).triu(1)
                scores = scores.masked_fill(future, float("-inf"))
            out = scores.softmax(dim=-1) @ v                                   # [B, h, Lq, hd]

        out = out.transpose(1, 2).reshape(B, Lq, D)
        return self.o(out)


class Block(nn.Module):
    """Pre-LayerNorm transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, dim, heads, mlp_dim, act, eps, fused):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, heads, fused)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = MLP(dim, mlp_dim, act)

    def forward(self, x, causal: bool = False):
        x = x + self.attn(self.norm1(x), causal=causal)
        x = x + self.mlp(self.norm2(x))
        return x


class MAPHead(nn.Module):
    """Multihead Attention Pooling (SigLIP's replacement for the [CLS] token).

    A single learned query (the "probe") cross-attends over all patch tokens and
    the result goes through one residual MLP. Output: one vector per image.
    """

    def __init__(self, dim, heads, mlp_dim, act, eps, fused):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(1, 1, dim))
        self.attn = Attention(dim, heads, fused)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.mlp = MLP(dim, mlp_dim, act)

    def forward(self, tokens):                                  # tokens: [B, N, D]
        h = self.attn(self.probe.expand(tokens.shape[0], -1, -1), kv=tokens)   # [B, 1, D]
        h = h + self.mlp(self.norm(h))
        return h[:, 0]


# ----------------------------------------------------------------------------
# Towers
# ----------------------------------------------------------------------------

class VisionTower(nn.Module):
    """ViT: split the image into patches, embed them, run transformer blocks, pool."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        n_patches = (c.image_size // c.patch_size) ** 2
        # A conv with kernel = stride = patch size is exactly "cut into patches + linear layer".
        self.patch_embed = nn.Conv2d(3, c.v_width, c.patch_size, c.patch_size, bias=c.patch_bias)
        self.cls_token = nn.Parameter(torch.zeros(c.v_width)) if c.vision_pool == "cls" else None
        n_pos = n_patches + (1 if self.cls_token is not None else 0)
        self.pos_embed = nn.Parameter(torch.zeros(n_pos, c.v_width))
        self.pre_norm = nn.LayerNorm(c.v_width, eps=c.ln_eps) if c.vision_pre_norm else None
        self.blocks = nn.ModuleList(
            Block(c.v_width, c.v_heads, c.v_mlp, c.act, c.ln_eps, c.fused_attn) for _ in range(c.v_depth))
        self.post_norm = nn.LayerNorm(c.v_width, eps=c.ln_eps)
        self.pool_head = (MAPHead(c.v_width, c.v_heads, c.v_mlp, c.act, c.ln_eps, c.fused_attn)
                          if c.vision_pool == "map" else None)

    def forward(self, images, return_tokens: bool = False):
        x = self.patch_embed(images)                  # [B, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)              # [B, N, D]  (row-major patch order)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.shape[0], 1, -1), x], dim=1)
        x = x + self.pos_embed
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.post_norm(x)
        pooled = self.pool_head(x) if self.pool_head is not None else x[:, 0]
        # return_tokens=True gives the patch tokens too (needed later for VL-JEPA-style predictors)
        return (pooled, x) if return_tokens else pooled


class TextTower(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.tok_embed = nn.Embedding(c.vocab_size, c.t_width)
        self.pos_embed = nn.Parameter(torch.zeros(c.context_len, c.t_width))
        self.blocks = nn.ModuleList(
            Block(c.t_width, c.t_heads, c.t_mlp, c.act, c.ln_eps, c.fused_attn) for _ in range(c.t_depth))
        self.final_norm = nn.LayerNorm(c.t_width, eps=c.ln_eps)

    def forward(self, ids):                           # ids: [B, L] token ids
        L = ids.shape[1]
        x = self.tok_embed(ids) + self.pos_embed[:L]
        for blk in self.blocks:
            x = blk(x, causal=self.c.text_causal)
        x = self.final_norm(x)
        if self.c.text_pool == "last":
            # SigLIP: captions are always padded to a fixed length, and with bidirectional
            # attention the last position sees every token, so it works as a summary token.
            return x[:, -1]
        # CLIP: with a causal mask only the EOS position has seen the whole caption.
        eos_pos = (ids == self.c.eos_token_id).int().argmax(dim=-1)
        return x[torch.arange(x.shape[0], device=x.device), eos_pos]


# ----------------------------------------------------------------------------
# The dual encoder
# ----------------------------------------------------------------------------

class DualEncoder(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.vision = VisionTower(c)
        self.text = TextTower(c)
        self.vision_proj = nn.Linear(c.v_width, c.embed_dim, bias=c.proj_bias) if c.vision_proj else None
        self.text_proj = nn.Linear(c.t_width, c.embed_dim, bias=c.proj_bias)

        self.set_logit_params("sigmoid" if c.arch == "siglip" else "softmax")
        self.reset_parameters()

    def set_logit_params(self, loss: str):
        """Learnable temperature (stored as a log) and, for the sigmoid loss, a learnable bias.

        Follows the *loss*, not the architecture, so "SigLIP architecture + softmax loss"
        gets CLIP's initialisation and vice versa.
        sigmoid (SigLIP paper): t' = log 10, b = -10. The very negative bias matters: at the
            start almost every pair in the batch is a negative, so the model should start out
            predicting "not a match" instead of paying a huge loss on all B^2 - B negatives.
        softmax (CLIP paper): temperature 0.07, i.e. scale = 1 / 0.07, no bias.
        """
        if loss == "sigmoid":
            self.logit_scale = nn.Parameter(torch.tensor([math.log(10.0)]))
            self.logit_bias = nn.Parameter(torch.tensor([-10.0]))
        elif loss == "softmax":
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
            self.logit_bias = None
        else:
            raise ValueError(loss)

    @torch.no_grad()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.vision.pos_embed, std=0.02)
        nn.init.normal_(self.text.pos_embed, std=0.01)
        if self.vision.cls_token is not None:
            nn.init.normal_(self.vision.cls_token, std=0.02)
        if self.vision.pool_head is not None:
            nn.init.normal_(self.vision.pool_head.probe, std=0.02)

    def encode_image(self, images, normalize: bool = True):
        z = self.vision(images)
        if self.vision_proj is not None:
            z = self.vision_proj(z)
        return F.normalize(z, dim=-1) if normalize else z

    def encode_text(self, ids, normalize: bool = True):
        z = self.text_proj(self.text(ids))
        return F.normalize(z, dim=-1) if normalize else z

    def forward(self, images, ids):
        """Returns L2-normalised image and text embeddings, each [B, embed_dim]."""
        return self.encode_image(images), self.encode_text(ids)

    def logits(self, img_emb, txt_emb):
        """[n_img, n_txt] similarity logits = cos-sim * exp(t') (+ b for SigLIP)."""
        out = img_emb @ txt_emb.t() * self.logit_scale.exp()
        if self.logit_bias is not None:
            out = out + self.logit_bias
        return out

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
