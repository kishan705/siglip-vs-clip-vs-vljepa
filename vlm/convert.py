"""
Copy weights from the official Hugging Face SigLIP / CLIP models into our
from-scratch DualEncoder.

Why bother? If our model, loaded with Google's / OpenAI's weights, produces the
same embeddings as theirs (to ~1e-5), every architectural detail is right.
That is the strongest evidence you can show an examiner that "implemented from
scratch" actually means "implemented correctly". See tests/test_parity.py.
"""

from __future__ import annotations

import torch

from .model import DualEncoder, ModelConfig


def config_from_hf(hf_config) -> ModelConfig:
    """Build our ModelConfig from a transformers SiglipConfig or CLIPConfig."""
    arch = "clip" if hf_config.model_type == "clip" else "siglip"
    v, t = hf_config.vision_config, hf_config.text_config
    act = {"gelu_pytorch_tanh": "gelu_tanh", "quick_gelu": "quick_gelu", "gelu": "gelu"}[v.hidden_act]
    kw = dict(
        arch=arch, act=act, ln_eps=v.layer_norm_eps,
        image_size=v.image_size, patch_size=v.patch_size,
        v_width=v.hidden_size, v_depth=v.num_hidden_layers, v_heads=v.num_attention_heads,
        v_mlp=v.intermediate_size,
        vocab_size=t.vocab_size, context_len=t.max_position_embeddings,
        t_width=t.hidden_size, t_depth=t.num_hidden_layers, t_heads=t.num_attention_heads,
        t_mlp=t.intermediate_size,
    )
    if arch == "clip":
        kw.update(embed_dim=hf_config.projection_dim, eos_token_id=t.eos_token_id)
    else:
        kw.update(embed_dim=v.hidden_size)
    return ModelConfig(**kw)


def _block_map(prefix_hf: str, prefix_ours: str, depth: int) -> dict[str, str]:
    m = {}
    for i in range(depth):
        h, o = f"{prefix_hf}.encoder.layers.{i}", f"{prefix_ours}.blocks.{i}"
        for ln_h, ln_o in (("layer_norm1", "norm1"), ("layer_norm2", "norm2")):
            m[f"{h}.{ln_h}.weight"], m[f"{h}.{ln_h}.bias"] = f"{o}.{ln_o}.weight", f"{o}.{ln_o}.bias"
        for p_h, p_o in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("out_proj", "o")):
            for wb in ("weight", "bias"):
                m[f"{h}.self_attn.{p_h}.{wb}"] = f"{o}.attn.{p_o}.{wb}"
        for fc in ("fc1", "fc2"):
            for wb in ("weight", "bias"):
                m[f"{h}.mlp.{fc}.{wb}"] = f"{o}.mlp.{fc}.{wb}"
    return m


def convert_state_dict(hf_sd: dict, c: ModelConfig) -> dict[str, torch.Tensor]:
    """Rename HF parameter names to ours (and split the MAP head's fused q/k/v)."""
    m = {}
    # text tower
    m["text_model.embeddings.token_embedding.weight"] = "text.tok_embed.weight"
    m["text_model.embeddings.position_embedding.weight"] = "text.pos_embed"
    m.update(_block_map("text_model", "text", c.t_depth))
    m["text_model.final_layer_norm.weight"] = "text.final_norm.weight"
    m["text_model.final_layer_norm.bias"] = "text.final_norm.bias"
    # vision tower
    m["vision_model.embeddings.patch_embedding.weight"] = "vision.patch_embed.weight"
    m["vision_model.embeddings.position_embedding.weight"] = "vision.pos_embed"
    m.update(_block_map("vision_model", "vision", c.v_depth))
    m["vision_model.post_layernorm.weight"] = "vision.post_norm.weight"
    m["vision_model.post_layernorm.bias"] = "vision.post_norm.bias"
    m["logit_scale"] = "logit_scale"

    if c.arch == "siglip":
        m["vision_model.embeddings.patch_embedding.bias"] = "vision.patch_embed.bias"
        m["text_model.head.weight"], m["text_model.head.bias"] = "text_proj.weight", "text_proj.bias"
        m["logit_bias"] = "logit_bias"
        hp = "vision_model.head"
        m[f"{hp}.probe"] = "vision.pool_head.probe"
        m[f"{hp}.attention.out_proj.weight"] = "vision.pool_head.attn.o.weight"
        m[f"{hp}.attention.out_proj.bias"] = "vision.pool_head.attn.o.bias"
        m[f"{hp}.layernorm.weight"], m[f"{hp}.layernorm.bias"] = ("vision.pool_head.norm.weight",
                                                                  "vision.pool_head.norm.bias")
        for fc in ("fc1", "fc2"):
            for wb in ("weight", "bias"):
                m[f"{hp}.mlp.{fc}.{wb}"] = f"vision.pool_head.mlp.{fc}.{wb}"
    else:
        m["vision_model.embeddings.class_embedding"] = "vision.cls_token"
        m["vision_model.pre_layrnorm.weight"] = "vision.pre_norm.weight"
        m["vision_model.pre_layrnorm.bias"] = "vision.pre_norm.bias"
        m["visual_projection.weight"] = "vision_proj.weight"
        m["text_projection.weight"] = "text_proj.weight"

    out = {}
    for k_hf, k_ours in m.items():
        out[k_ours] = hf_sd[k_hf]
    if c.arch == "siglip":
        # torch.nn.MultiheadAttention stores q, k, v stacked in one matrix: split it.
        w = hf_sd["vision_model.head.attention.in_proj_weight"]
        b = hf_sd["vision_model.head.attention.in_proj_bias"]
        for name, wi, bi in zip("qkv", w.chunk(3, dim=0), b.chunk(3, dim=0)):
            out[f"vision.pool_head.attn.{name}.weight"] = wi
            out[f"vision.pool_head.attn.{name}.bias"] = bi
    return out


def from_hf_model(hf_model, fused_attn: bool = True) -> DualEncoder:
    c = config_from_hf(hf_model.config)
    c.fused_attn = fused_attn
    model = DualEncoder(c)
    model.load_state_dict(convert_state_dict(hf_model.state_dict(), c), strict=True)
    return model


def from_pretrained(model_id: str, fused_attn: bool = True) -> DualEncoder:
    """e.g. from_pretrained("google/siglip-base-patch16-224") -- needs internet."""
    from transformers import AutoModel
    hf = AutoModel.from_pretrained(model_id)
    return from_hf_model(hf, fused_attn=fused_attn)
