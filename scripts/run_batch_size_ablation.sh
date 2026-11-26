#!/usr/bin/env bash
# Thesis experiment 1: does SigLIP's sigmoid loss beat CLIP's softmax loss, and how does
# that depend on batch size? (The SigLIP paper's central claim, reproduced at small scale.)
#
# Everything except --loss and --batch_size is held fixed, and every run sees the same
# number of image-text pairs (TOTAL), so a bigger batch means fewer steps, not more compute.
#
# Usage:  DATA=data/coco IMG_ROOT=/data/coco TOTAL=2000000 bash scripts/run_batch_size_ablation.sh
# Extra args are passed to train.py, e.g.  ... run_batch_size_ablation.sh --image_size 128 --wandb
# Finished runs are skipped, so you can re-run it after a crash / Modal timeout.
set -euo pipefail
DATA=${DATA:-data/coco}
IMG_ROOT=${IMG_ROOT:-/data/coco}
TOTAL=${TOTAL:-2000000}
PRESET=${PRESET:-siglip-ti16}
BATCHES=${BATCHES:-"32 64 128 256 512"}
SEEDS=${SEEDS:-"0 1 2"}

for bs in $BATCHES; do
  for loss in sigmoid softmax; do
    for seed in $SEEDS; do
      name="${PRESET}_${loss}_bs${bs}_s${seed}"
      if [ -f "runs/$name/results.json" ]; then echo "skip $name (done)"; continue; fi
      python train.py --preset "$PRESET" --loss "$loss" --batch_size "$bs" --seed "$seed" \
        --total_samples "$TOTAL" --train_jsonl "$DATA/train.jsonl" --val_jsonl "$DATA/val_1k.jsonl" \
        --image_root "$IMG_ROOT" --name "$name" --resume "$@"
    done
  done
done
python scripts/summarize.py runs
