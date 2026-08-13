# KGW Watermark Playground

Interactive Streamlit app for exploring the Kirchenbauer et al. (2023) LLM
watermarking scheme from ["A Watermark for Large Language Models"](https://arxiv.org/abs/2301.10226)
and its robustness follow-up ["On the Reliability of Watermarks for Large Language Models"](https://arxiv.org/abs/2306.04634).

Runs fully locally on facebook/opt-125m (CPU). No API keys, no network calls
after the first model download.

**Live app:** [llm-watermark-checker.streamlit.app](https://llm-watermark-checker.streamlit.app)

## What it does

- **Generate & compare**: the same prompt with and without the watermark,
  side by side, with per-token green-list highlighting and detection stats
  (z-score, green token fraction) under each.
- **Detect**: score any text instantly. Bundled samples (a watermarked
  generation and a human-written control) work with one click, no generation
  needed. Edit the text and re-score for a hands-on robustness experiment.
- **Attack the watermark**: truncation, random-deletion and random-replacement
  sweeps; a dilution demo hiding a 60-token watermarked snippet in unwatermarked
  prose (whole-document scoring dilutes below threshold, the WinMax sliding
  window still finds the span); and a bring-your-own-chatbot paraphrase attack
  with a side-by-side before/after comparison.

Changing gamma or the seeding scheme after generating shows detection
collapse, which is the scheme's key property: without the exact key there
is nothing to test.

Sidebar controls expose gamma (green list fraction), delta (logit boost),
the seeding scheme, sampling settings, and the detection threshold.

## Run locally

    python3 -m venv venv
    ./venv/bin/pip install -r requirements.txt
    ./venv/bin/streamlit run app.py

`play_watermark.py` and `robustness_play.py` are CLI versions of the same
experiments (run `play_watermark.py` first; it writes the watermarked sample
that `robustness_play.py` attacks).

## Notes

- The first generation loads the model (~250MB download from Hugging Face),
  so it is slow once, then cached.
- Detection requires the same gamma and seeding scheme used at generation
  time. That is the point of the scheme: without the key, there is nothing
  to test.
- `lm-watermarking/` is a pruned vendored copy of the authors' reference
  implementation, [jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking)
  (Apache 2.0, license preserved in that directory). All watermarking and
  detection logic is theirs; this repo just wraps a UI around it.
- `samples/` holds a watermarked generation made at the default settings
  (gamma 0.25, delta 2.0, selfhash, seed 42) and a human-written control,
  so the Detect tab works without waiting for a generation.

Built by [Lee Foot](https://leefoot.com).
