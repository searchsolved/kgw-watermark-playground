"""Robustness: z-score vs truncation length, and vs random token deletion."""
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "lm-watermarking"))

from extended_watermark_processor import WatermarkDetector

random.seed(0)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
vocab = list(tokenizer.get_vocab().values())
detector = WatermarkDetector(vocab=vocab, gamma=0.25, seeding_scheme="selfhash",
                             device="cpu", tokenizer=tokenizer, z_threshold=4.0,
                             normalizers=[], ignore_repeated_ngrams=True)

wm_text = Path(__file__).with_name("wm_output.txt").read_text()
ids = tokenizer(wm_text, add_special_tokens=False)["input_ids"]

print(f"full watermarked text: {len(ids)} tokens\n")
print("truncation: keep first N tokens")
for n in [20, 40, 60, 100, 150, len(ids)]:
    s = detector.detect(tokenizer.decode(ids[:n]))
    flag = "DETECTED" if s["prediction"] else "missed"
    print(f"  N={n:3d}  z={s['z_score']:6.2f}  green={s['green_fraction']:.3f}  {flag}")

print("\nrandom token deletion: drop X% of tokens, re-detect (5 trials each)")
for frac in [0.1, 0.2, 0.3, 0.5]:
    zs = []
    for _ in range(5):
        kept = [t for t in ids if random.random() > frac]
        zs.append(detector.detect(tokenizer.decode(kept))["z_score"])
    avg = sum(zs) / len(zs)
    print(f"  drop {int(frac*100):2d}%  mean z={avg:5.2f}  ({'still detected' if avg > 4 else 'below threshold'})")
