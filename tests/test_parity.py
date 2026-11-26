"""
Parity tests: our from-scratch model vs the official Hugging Face implementation.

    python -m pytest tests/ -q          # random small models, runs offline on CPU in seconds
    python tests/test_parity.py --pretrained google/siglip-base-patch16-224   # real weights (needs internet)
    python tests/test_parity.py --pretrained openai/clip-vit-base-patch32

Random-weight tests already prove the maths is identical; the --pretrained run
additionally shows it on the real released checkpoints.
"""

import argparse
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm.convert import from_hf_model  # noqa: E402
from vlm.losses import chunked_sigmoid_loss, sigmoid_loss, softmax_loss  # noqa: E402

transformers = pytest.importorskip("transformers")

SMALL_TEXT = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                  vocab_size=100, max_position_embeddings=16)
SMALL_VISION = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                    image_size=32, patch_size=8)
ATOL = 1e-5


def _random_hf_siglip():
    from transformers import SiglipConfig, SiglipModel
    torch.manual_seed(0)
    hf = SiglipModel(SiglipConfig(text_config=SMALL_TEXT, vision_config=SMALL_VISION)).eval()
    with torch.no_grad():   # HF initialises these to 0; use non-trivial values
        hf.logit_scale.fill_(2.3)
        hf.logit_bias.fill_(-10.0)
    return hf


def _random_hf_clip(eos_id=99):
    from transformers import CLIPConfig, CLIPModel
    torch.manual_seed(0)
    text = dict(SMALL_TEXT, eos_token_id=eos_id)
    return CLIPModel(CLIPConfig(text_config=text, vision_config=SMALL_VISION, projection_dim=32)).eval()


@pytest.mark.parametrize("fused", [True, False])
def test_siglip_matches_hf(fused):
    hf = _random_hf_siglip()
    ours = from_hf_model(hf, fused_attn=fused).eval()
    torch.manual_seed(1)
    images = torch.randn(6, 3, 32, 32)
    ids = torch.randint(0, 100, (6, 16))
    with torch.no_grad():
        out = hf(input_ids=ids, pixel_values=images, return_loss=True)
        img, txt = ours(images, ids)
        logits = ours.logits(img, txt)
        loss = sigmoid_loss(img, txt, ours.logit_scale, ours.logit_bias)
    torch.testing.assert_close(img, out.image_embeds, atol=ATOL, rtol=0)
    torch.testing.assert_close(txt, out.text_embeds, atol=ATOL, rtol=0)
    torch.testing.assert_close(logits, out.logits_per_image, atol=1e-4, rtol=0)
    torch.testing.assert_close(loss, out.loss, atol=1e-4, rtol=0)


@pytest.mark.parametrize("fused", [True, False])
def test_clip_matches_hf(fused):
    eos = 99
    hf = _random_hf_clip(eos)
    ours = from_hf_model(hf, fused_attn=fused).eval()
    torch.manual_seed(1)
    images = torch.randn(6, 3, 32, 32)
    ids = torch.randint(0, 98, (6, 16))
    for i, pos in enumerate([3, 15, 7, 1, 10, 5]):   # EOS at a different place in every caption
        ids[i, pos] = eos
    with torch.no_grad():
        out = hf(input_ids=ids, pixel_values=images, return_loss=True)
        img, txt = ours(images, ids)
        logits = ours.logits(img, txt)
        loss = softmax_loss(img, txt, ours.logit_scale)
    torch.testing.assert_close(img, out.image_embeds, atol=ATOL, rtol=0)
    torch.testing.assert_close(txt, out.text_embeds, atol=ATOL, rtol=0)
    torch.testing.assert_close(logits, out.logits_per_image, atol=1e-4, rtol=0)
    torch.testing.assert_close(loss, out.loss, atol=1e-4, rtol=0)


def test_chunked_sigmoid_equals_full():
    torch.manual_seed(0)
    img = torch.nn.functional.normalize(torch.randn(10, 8), dim=-1)
    txt = torch.nn.functional.normalize(torch.randn(10, 8), dim=-1)
    s, b = torch.tensor([2.3]), torch.tensor([-10.0])
    full = sigmoid_loss(img, txt, s, b)
    for chunk in (1, 3, 4, 10):
        torch.testing.assert_close(chunked_sigmoid_loss(img, txt, s, b, chunk), full)


def test_gradients_match_hf_siglip():
    """Same loss value is not enough for training -- check the gradients too."""
    hf = _random_hf_siglip().train()
    ours = from_hf_model(hf).train()
    torch.manual_seed(2)
    images, ids = torch.randn(4, 3, 32, 32), torch.randint(0, 100, (4, 16))
    hf(input_ids=ids, pixel_values=images, return_loss=True).loss.backward()
    img, txt = ours(images, ids)
    sigmoid_loss(img, txt, ours.logit_scale, ours.logit_bias).backward()
    torch.testing.assert_close(ours.vision.patch_embed.weight.grad,
                               hf.vision_model.embeddings.patch_embedding.weight.grad, atol=1e-5, rtol=1e-3)
    torch.testing.assert_close(ours.text.tok_embed.weight.grad,
                               hf.text_model.embeddings.token_embedding.weight.grad, atol=1e-5, rtol=1e-3)
    torch.testing.assert_close(ours.logit_bias.grad, hf.logit_bias.grad, atol=1e-5, rtol=1e-3)


# ----------------------------------------------------------------------------
# Real checkpoint check (run manually; needs internet to download weights)
# ----------------------------------------------------------------------------

def check_pretrained(model_id: str, image_path: str | None):
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    hf = AutoModel.from_pretrained(model_id).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    ours = from_hf_model(hf).eval()

    image = Image.open(image_path).convert("RGB") if image_path else Image.new("RGB", (256, 256), (200, 30, 30))
    texts = ["a photo of a cat", "a photo of a dog", "a plain red square", "a photo of a car"]
    # SigLIP was trained with captions padded to exactly 64 tokens -- you must do the same.
    pad = dict(padding="max_length") if hf.config.model_type == "siglip" else dict(padding=True)
    inputs = proc(text=texts, images=image, return_tensors="pt", **pad)

    with torch.no_grad():
        out = hf(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"])
        img, txt = ours(inputs["pixel_values"], inputs["input_ids"])
        logits = ours.logits(img, txt)

    print(f"{model_id}")
    print("  max |diff| image embeds:", (img - out.image_embeds).abs().max().item())
    print("  max |diff| text  embeds:", (txt - out.text_embeds).abs().max().item())
    probs = torch.sigmoid(logits) if hf.config.model_type == "siglip" else logits.softmax(-1)
    for t, p in zip(texts, probs[0].tolist()):
        print(f"  {p:6.3f}  {t}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="google/siglip-base-patch16-224")
    ap.add_argument("--image", default=None, help="optional path to an image to classify")
    a = ap.parse_args()
    check_pretrained(a.pretrained, a.image)
