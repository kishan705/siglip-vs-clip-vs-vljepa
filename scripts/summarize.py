"""
Collect every finished run into one table: mean +- std over seeds.

    python scripts/summarize.py runs            # prints a markdown table, writes runs/summary.csv
"""
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else "runs"
METRICS = ["mean_R", "t2i_R@1", "i2t_R@1", "t2i_R@5", "i2t_R@5", "cifar10_zeroshot_top1"]

groups = defaultdict(list)
for res_path in glob.glob(os.path.join(root, "*", "results.json")):
    run = os.path.dirname(res_path)
    cfg = json.load(open(os.path.join(run, "config.json")))
    res = json.load(open(res_path))
    a = cfg["args"]
    # runs are "the same experiment" if every setting except the seed (and bookkeeping) matches
    ignore = {"seed", "name", "out", "resume", "workers", "log_every", "eval_every", "save_every",
              "wandb", "device", "deterministic"}
    setting = json.dumps({k: v for k, v in sorted(a.items()) if k not in ignore})
    data = "synthetic" if a["data"] == "synthetic" else os.path.basename(os.path.dirname(a["train_jsonl"]) or ".")
    key = (a["preset"], cfg["model_config"]["arch"], a["loss"], a["batch_size"], data,
           cfg["total_steps"], setting)
    groups[key].append(res)

if not groups:
    sys.exit(f"no finished runs (results.json) under {root}/")

present = [m for m in METRICS if any(m in r for rs in groups.values() for r in rs)]
rows = []
for key in sorted(groups, key=lambda k: (k[4], k[0], k[1], k[3], k[2], k[5])):
    rs = groups[key]
    row = dict(zip(["preset", "arch", "loss", "batch", "data", "steps"], key[:6]))
    row["seeds"] = len(rs)
    for m in present:
        vals = [r[m] for r in rs if m in r]
        if vals:
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            row[m] = f"{statistics.mean(vals):.2f} ± {sd:.2f}"
    rows.append(row)

cols = ["preset", "arch", "loss", "batch", "data", "steps", "seeds"] + present
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for r in rows:
    print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
with open(os.path.join(root, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
