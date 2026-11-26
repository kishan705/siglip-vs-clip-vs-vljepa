# SigLIP vs CLIP vs VL-JEPA

**A controlled comparison of contrastive and predictive vision-language learning**

IDD Thesis · Department of Mathematical Sciences, IIT (BHU) Varanasi
**Author:** Kishan Amaliya        
**Supervisor:** Dr. Sheela Verma

---

## 1. The idea

Vision-language models learn a **shared embedding space** for images and text. Matching
pairs (a photo and its caption) end up close together, and unrelated pairs far apart.
Many modern systems are built on this space: image search, zero-shot classification, and the
vision encoders inside large vision-language models (for example, SigLIP inside PaliGemma).

This thesis studies three different ways of *training* that space:

| Method | Year | Family | How it learns |
|---|---|---|---|
| **CLIP** | 2021 | Contrastive | For each image, pick the right caption out of the whole batch (**softmax** over the batch) |
| **SigLIP** | 2023 | Contrastive | Treat every (image, caption) pair as an independent yes/no question (**sigmoid** per pair) |
| **VL-JEPA** | 2026 | Predictive | Given the image (and optionally a question), **predict the embedding** of the target text |

CLIP and SigLIP differ only in the loss. VL-JEPA changes the idea itself. Instead of
matching two embeddings, a predictor network *predicts* the text embedding, in the spirit of
the Joint Embedding Predictive Architectures (JEPA) proposed for world models.

**Research questions**
1. With the same data, encoders and compute, how do the three objectives compare on
   image-text retrieval and zero-shot classification?
2. How sensitive is each objective to **batch size**? SigLIP's main claim is that the sigmoid loss
   works better than softmax at small batch sizes.
3. Does predicting in embedding space (VL-JEPA) help when the task depends on a **query**?
4. What does each method cost in GPU memory, training speed and trainable parameters?

**Why implement them from scratch?** The official CLIP, SigLIP and VL-JEPA checkpoints
were trained on different datasets, at different sizes, with very different compute budgets.
Comparing them directly would not show which *method* is better. Implementing all three in one
codebase lets us keep everything identical except the method. To show the implementation is
correct, it is checked against the official reference code (see Section 3).

---

## 2. How we compare: the core design

**Principle: change one thing at a time.** All three methods share the same data, encoders,
training loop, budget and evaluation. Only the objective changes.

```
                 ┌─────────────────────────────────────────────┐
                 │ SHARED: data · image encoder · text encoder │
                 │         training loop · budget · seeds       │
                 └─────────────────────────────────────────────┘
                        │                │                 │
                 ┌──────┴──────┐  ┌──────┴──────┐  ┌───────┴────────┐
                 │    CLIP     │  │   SigLIP    │  │    VL-JEPA     │
                 │ softmax over│  │ sigmoid per │  │ predictor →    │
                 │ the batch   │  │ pair + bias │  │ text embedding │
                 └──────┬──────┘  └──────┬──────┘  └───────┬────────┘
                        │                │                 │
                 ┌──────┴────────────────┴─────────────────┴──────┐
                 │ SAME EVALUATION: retrieval · zero-shot · cost   │
                 └─────────────────────────────────────────────────┘
```

### What is shared and what differs

| | CLIP | SigLIP | VL-JEPA |
|---|---|---|---|
| Image side | pooled image vector | pooled image vector | patch tokens (+ query) → **predictor** |
| Text side | text encoder | text encoder | text encoder (target embedding) |
| Training signal | softmax InfoNCE over the batch | pairwise sigmoid, learnable bias `b` | InfoNCE between predicted and true text embedding |
| Learnable scalars | temperature `t` | temperature `t`, bias `b` | temperature `t` |

### Experimental settings

The original models need far more compute than a thesis can use. Even the cheapest
from-scratch SigLIP in the paper used 16 TPUv4 chips for 3 days, and VL-JEPA's pretraining
took about 4 weeks on 192 H200 GPUs. So the study is done at a **small, controlled scale**,
where the *relative* comparison is the result.

- **Setting A: shared frozen image encoder (main comparison, all three methods).**
  One pretrained image encoder is frozen and shared by all methods. Only the text side and
  the method-specific part are trained, on COCO / Flickr30k. This is cheap enough for a single GPU,
  and it is fair because the image side is identical. It also matches VL-JEPA's own design,
  which keeps its vision encoder frozen.
- **Setting B: small models trained from scratch (CLIP vs SigLIP).**
  A small ViT and text encoder are trained from random weights with batch sizes 32 to 512, and
  every run sees the same number of image-text pairs. This tests the SigLIP paper's
  batch-size claim in exactly the regime where it says sigmoid should win (batch < 16k).
- **Reference row: official checkpoints.** The released SigLIP and CLIP weights, loaded
  into our implementation and evaluated with the same code. This gives context only and
  is *not* a fair comparison.

### Metrics

| What | How |
|---|---|
| Image-text retrieval | Recall@1/5/10, image→text and text→image (COCO 5k, Flickr30k 1k test) |
| Zero-shot classification | CIFAR-10 top-1, using class-name prompts ("a photo of a dog.") |
| Query-conditioned task | Where VL-JEPA's predictor should matter (e.g. question-based classification) |
| Cost | GPU memory, samples/second, trainable parameters |

### Rigour

- **3 seeds** per configuration, reported as mean ± std. A difference smaller than the std does not count as a result.
- **Equal compute:** every run sees the same number of image-text pairs.
- **Full traceability:** every run saves its arguments, model config, git commit, library versions and GPU (`runs/<name>/config.json`).
- **Correctness tests:** the implementation is checked against the reference code before any experiment.

### Final results table (to be filled)

| Method | Setting | Trainable params | t→i R@1 | i→t R@1 | CIFAR-10 0-shot | GPU mem | samples/s |
|---|---|---|---|---|---|---|---|
| CLIP | A | | | | | | |
| SigLIP | A | | | | | | |
| VL-JEPA | A | | | | | | |
| CLIP (official) | ref | | | | | | |
| SigLIP (official) | ref | | | | | | |

---

## 3. Current progress

**Done**
- [x] Architecture study of CLIP and SigLIP. The table below lists every design difference, and each one is a switch in the code.
- [x] **SigLIP and CLIP implemented from scratch in PyTorch** (`vlm/model.py`, `vlm/losses.py`): ViT image encoder, text transformer, MAP attention-pooling head, sigmoid and softmax losses, and a chunked sigmoid loss (the paper's memory-saving trick).
- [x] **Correctness verified against the Hugging Face reference implementation.** Embeddings, logits, loss and gradients all match to about 1e-5 (`tests/test_parity.py`, 7 tests passing). Parameter counts exactly match the official SigLIP-B/16 (203.16M) and CLIP-B/32 (151.28M).
- [x] Training pipeline (`train.py`): seeding, config and git tracking, resume after interruption, mixed precision, optional Weights & Biases logging.
- [x] Evaluation (`vlm/evaluate.py`, `eval.py`): retrieval Recall@K and zero-shot classification.
- [x] Sanity checks. Both losses fit a fixed batch perfectly (R@1 = 100%). On a generated toy dataset, text→image R@10 on held-out images rose from 3% (chance) to 66%.

**Architecture differences found (CLIP vs SigLIP)**

| | SigLIP | CLIP |
|---|---|---|
| Image pooling | MAP head (learned query attends over patches) | `[CLS]` token |
| LayerNorm after embeddings | no | yes |
| Text attention | bidirectional | causal |
| Text pooling | last position (fixed 64-token padding) | EOS token |
| Projections | text head only (with bias) | image + text (no bias) |
| Activation / LayerNorm eps | GELU-tanh / 1e-6 | QuickGELU / 1e-5 |
| Loss / init | sigmoid, `t' = log 10`, `b = −10` | softmax, `t = 1/0.07` |

**In progress**
- [ ] Parity check with the official pretrained checkpoints (`google/siglip-base-patch16-224`, `openai/clip-vit-base-patch32`)
- [ ] Reference evaluation of the official checkpoints on COCO / Flickr30k
- [ ] Setting B: batch-size ablation of CLIP vs SigLIP on COCO (2 losses × 5 batch sizes × 3 seeds)

---

## 4. Future remaining work

1. **VL-JEPA architecture study.** Paper facts: a frozen V-JEPA 2 ViT-L as the image encoder; a predictor initialised from the last 8 layers of Llama-3.2-1B with the causal mask removed; EmbeddingGemma-300M as the text encoder; bi-directional InfoNCE in a shared 1536-d space.
2. **VL-JEPA implementation at small scale.** A predictor built from this codebase's transformer `Block` reads [image patch tokens; query tokens], then mean-pools and projects. It is trained with the same InfoNCE loss already implemented (`softmax_loss`).
3. **Setting A for all three methods** on the shared frozen image encoder.
4. **A query-conditioned evaluation** where VL-JEPA's predictor can make a difference. Without a query, VL-JEPA is close to CLIP-style training with a deeper head, so this test is essential.
5. **Analysis and write-up.** Fill the final results table and plot batch-size curves, the temperature/bias dynamics during training, and cost. Then write the thesis and final presentation.

---

## Repository and how to run

```
vlm/model.py        ViT, text transformer, MAP head, dual encoder (no `transformers` model used)
vlm/losses.py       sigmoid, softmax (InfoNCE) and chunked sigmoid losses
vlm/convert.py      loads official Hugging Face weights into our model
vlm/data.py         datasets and tokenizers
vlm/evaluate.py     retrieval Recall@K, zero-shot classification
train.py            training (seeds, config tracking, resume, fp16)
eval.py             evaluate official checkpoints or our own runs
prepare_data.py     COCO / Flickr30k → jsonl
scripts/            batch-size ablation, results table, toy dataset
tests/              parity tests against Hugging Face, metric tests
```

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                                         # correctness tests (CPU, seconds)
python tests/test_parity.py --pretrained google/siglip-base-patch16-224   # check vs official weights

# sanity check: memorise one batch
python train.py --data synthetic --preset tiny-test --synthetic_size 64 --batch_size 64 \
    --max_steps 300 --lr 3e-3 --wd 0 --workers 0 --name overfit_sigmoid

# data, training, evaluation
python prepare_data.py coco --root /data/coco --out data/coco
python train.py --preset siglip-ti16 --loss sigmoid --batch_size 256 --total_samples 2000000 \
    --train_jsonl data/coco/train.jsonl --val_jsonl data/coco/val_1k.jsonl --image_root /data/coco
DATA=data/coco IMG_ROOT=/data/coco TOTAL=2000000 bash scripts/run_batch_size_ablation.sh
python scripts/summarize.py runs                                   # mean ± std table
python eval.py --init_from google/siglip-base-patch16-224 --val_jsonl data/coco/val_5k.jsonl \
    --image_root /data/coco --zeroshot_cifar10 data/cifar
```

Practical notes: SigLIP captions must be padded to exactly 64 tokens. The loss is computed in
fp32 even under mixed precision. COCO `val2017` is not the "Karpathy 5k" split used in papers,
so compare runs with each other, not with numbers in paper tables.

---

## References

1. A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML 2021. arXiv:2103.00020
2. X. Zhai, B. Mustafa, A. Kolesnikov, L. Beyer. *Sigmoid Loss for Language Image Pre-Training* (SigLIP). ICCV 2023. arXiv:2303.15343
3. D. Chen et al. *VL-JEPA: Joint Embedding Predictive Architecture for Vision-language*. ICLR 2026. arXiv:2512.10942
4. X. Zhai et al. *LiT: Zero-Shot Transfer with Locked-image text Tuning*. CVPR 2022. arXiv:2111.07991
