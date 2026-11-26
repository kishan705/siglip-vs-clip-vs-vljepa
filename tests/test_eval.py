import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm.evaluate import recall_at_k  # noqa: E402


def test_recall_perfect_and_known_case():
    img = torch.eye(3)                                   # 3 images
    txt = torch.eye(3).repeat_interleave(2, 0)           # 2 captions per image, exact matches
    r = recall_at_k(img, txt, [0, 0, 1, 1, 2, 2], ks=(1, 2))
    assert r["t2i_R@1"] == 100 and r["i2t_R@1"] == 100

    # caption 0 now points mostly at image 1 (no exact ties, so the result is deterministic)
    txt[0] = torch.tensor([0.0, 0.9, 0.1])
    r = recall_at_k(img, txt, [0, 0, 1, 1, 2, 2], ks=(1,))
    assert abs(r["t2i_R@1"] - 100 * 5 / 6) < 1e-4      # 5 of 6 captions still find their image
    assert r["i2t_R@1"] == 100                          # image 0 still has caption 1 at rank 1
