"""Play with KGW watermarking: generate with/without watermark, then detect.

Uses the repo's recommended extended_watermark_processor (selfhash scheme).
Model: facebook/opt-125m, CPU, fully local.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).parent / "lm-watermarking"))

from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

MODEL = "facebook/opt-125m"
GAMMA = 0.25          # fraction of vocab in the green list each step
DELTA = 2.0           # logit boost given to green tokens
SCHEME = "selfhash"   # the paper's recommended seeding scheme
MAX_NEW = 200

device = "cpu"
torch.manual_seed(42)

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()
vocab = list(tokenizer.get_vocab().values())

prompt = (
    "The history of watermarking goes back centuries. Papermakers in "
    "thirteenth-century Italy pressed wire designs into wet pulp so that"
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
prompt_len = inputs["input_ids"].shape[-1]

wm_processor = WatermarkLogitsProcessor(
    vocab=vocab, gamma=GAMMA, delta=DELTA, seeding_scheme=SCHEME
)

gen_kwargs = dict(do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=MAX_NEW)

print("Generating WITHOUT watermark...")
plain_ids = model.generate(**inputs, **gen_kwargs)
plain_text = tokenizer.decode(plain_ids[0, prompt_len:], skip_special_tokens=True)

print("Generating WITH watermark...")
wm_ids = model.generate(
    **inputs, logits_processor=LogitsProcessorList([wm_processor]), **gen_kwargs
)
wm_text = tokenizer.decode(wm_ids[0, prompt_len:], skip_special_tokens=True)
Path(__file__).with_name("wm_output.txt").write_text(wm_text)  # input for robustness_play.py

human_text = (
    "Watermarks were first introduced in Fabriano, Italy, in 1282. They were "
    "made by adding thin wire patterns to the paper mould, which produced a "
    "slightly thinner, more translucent area in the finished sheet. Papermakers "
    "used them as trademarks to identify their workshops, and over time "
    "governments adopted them to deter counterfeiting of currency and official "
    "documents. Modern banknotes still rely on watermarks as a security feature, "
    "alongside holograms and security threads. The digital analogue emerged in "
    "the 1990s, when researchers began embedding imperceptible signals into "
    "images and audio to prove ownership and trace unauthorised copies."
)

detector = WatermarkDetector(
    vocab=vocab,
    gamma=GAMMA,
    seeding_scheme=SCHEME,
    device=device,
    tokenizer=tokenizer,
    z_threshold=4.0,
    normalizers=[],
    ignore_repeated_ngrams=True,
)

print("\n" + "=" * 78)
for label, text in [
    ("WATERMARKED generation", wm_text),
    ("UNwatermarked generation", plain_text),
    ("HUMAN-written control", human_text),
]:
    score = detector.detect(text)
    print(f"\n--- {label} ---")
    print(text[:400].replace("\n", " "))
    print(
        f"\n  tokens scored: {score['num_tokens_scored']}"
        f" | green fraction: {score['green_fraction']:.3f}"
        f" (expected if unwatermarked: {GAMMA})"
        f"\n  z-score: {score['z_score']:.2f}"
        f" | p-value: {score['p_value']:.2e}"
        f" | verdict at z>4: {'WATERMARK DETECTED' if score['prediction'] else 'not detected'}"
    )
print("\n" + "=" * 78)
