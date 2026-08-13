"""KGW watermark playground.

Streamlit app for generating and detecting statistically watermarked LLM text,
using the reference implementation from Kirchenbauer et al. (2023),
"A Watermark for Large Language Models" (arXiv:2301.10226).

Runs fully locally on facebook/opt-125m. No APIs.
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).parent / "lm-watermarking"))

from extended_watermark_processor import WatermarkDetector, WatermarkLogitsProcessor  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
)

MODEL_NAME = "facebook/opt-125m"
DEVICE = "cpu"

# Chart / highlight colours (validated reference palette from the dataviz method)
SERIES_BLUE = "#2a78d6"
GREEN = "#008300"
TEXT_SECONDARY = "#52514e"

st.set_page_config(page_title="KGW Watermark Playground", page_icon="🖋️", layout="wide")


@st.cache_resource(show_spinner="Loading facebook/opt-125m (first run only)...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return tokenizer, model


def make_detector(tokenizer, gamma, scheme, z_threshold, ignore_repeats):
    return WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=gamma,
        seeding_scheme=scheme,
        device=DEVICE,
        tokenizer=tokenizer,
        z_threshold=z_threshold,
        normalizers=[],
        ignore_repeated_ngrams=ignore_repeats,
    )


def generate(prompt, watermark, gamma, delta, scheme, temperature, top_p, max_new, seed):
    tokenizer, model = load_model()
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    kwargs = dict(do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_new)
    if watermark:
        processor = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            seeding_scheme=scheme,
        )
        kwargs["logits_processor"] = LogitsProcessorList([processor])
    out = model.generate(**inputs, **kwargs)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)


def detect(text, gamma, scheme, z_threshold, ignore_repeats, with_mask=False):
    tokenizer, _ = load_model()
    detector = make_detector(tokenizer, gamma, scheme, z_threshold, ignore_repeats)
    return detector.detect(text, return_green_token_mask=with_mask)


def score_metrics(score, z_threshold):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("z-score", f"{score['z_score']:.2f}")
    c2.metric("Green fraction", f"{score['green_fraction']:.3f}")
    c3.metric("p-value", f"{score['p_value']:.2e}")
    c4.metric("Tokens scored", score["num_tokens_scored"])
    if score["prediction"]:
        st.success(f"Watermark detected (z > {z_threshold})")
    else:
        st.info(f"No watermark detected (z below the {z_threshold} threshold)")


def token_highlight_html(text, score):
    """Render tokens with green-list hits highlighted.

    Green tokens get a tinted background AND a solid underline, so the encoding
    never relies on colour alone.
    """
    tokenizer, _ = load_model()
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    mask = score.get("green_token_mask", [])
    prefix = len(ids) - len(mask)  # leading context tokens carry no verdict
    spans = []
    for i, tok_id in enumerate(ids):
        piece = tokenizer.decode([tok_id])
        piece = piece.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if i < prefix:
            spans.append(f'<span style="color:{TEXT_SECONDARY}">{piece}</span>')
        elif mask[i - prefix]:
            spans.append(
                f'<span style="background:{GREEN}22;border-bottom:2px solid {GREEN}">{piece}</span>'
            )
        else:
            spans.append(f"<span>{piece}</span>")
    legend = (
        f'<p style="font-size:0.85em;color:{TEXT_SECONDARY}">'
        f'<span style="background:{GREEN}22;border-bottom:2px solid {GREEN}">underlined</span>'
        " = token was on its step's green list · plain = red list · "
        "grey = leading context, not scored</p>"
    )
    return legend + f'<div style="line-height:1.9">{"".join(spans)}</div>'


def z_line_chart(df, x_field, x_title, z_threshold):
    base = alt.Chart(df).encode(
        x=alt.X(x_field, title=x_title),
        y=alt.Y("z:Q", title="Detection z-score"),
        tooltip=[alt.Tooltip(x_field, title=x_title), alt.Tooltip("z:Q", format=".2f")],
    )
    line = base.mark_line(color=SERIES_BLUE, strokeWidth=2) + base.mark_point(
        color=SERIES_BLUE, size=70, filled=True
    )
    rule = (
        alt.Chart(pd.DataFrame({"z": [z_threshold]}))
        .mark_rule(color=TEXT_SECONDARY, strokeDash=[4, 4])
        .encode(y="z:Q")
    )
    label = (
        alt.Chart(pd.DataFrame({"z": [z_threshold], "t": [f"detection threshold z = {z_threshold}"]}))
        .mark_text(align="left", dx=4, dy=-6, color=TEXT_SECONDARY)
        .encode(y="z:Q", text="t:N")
    )
    return (line + rule + label).properties(height=280)


# ---------------------------------------------------------------- sidebar

st.sidebar.title("Watermark settings")
gamma = st.sidebar.slider("gamma (green list fraction)", 0.1, 0.5, 0.25, 0.05)
delta = st.sidebar.slider("delta (green logit boost)", 0.5, 10.0, 2.0, 0.5)
scheme = st.sidebar.selectbox("Seeding scheme", ["selfhash", "simple_1", "minhash"], index=0)
z_threshold = st.sidebar.slider("Detection z threshold", 2.0, 6.0, 4.0, 0.5)
ignore_repeats = st.sidebar.checkbox("Ignore repeated ngrams when scoring", value=True)
st.sidebar.divider()
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
top_p = st.sidebar.slider("top_p", 0.5, 1.0, 0.95, 0.05)
max_new = st.sidebar.slider("Max new tokens", 50, 400, 200, 25)
seed = st.sidebar.number_input("Sampling seed", value=42, step=1)
st.sidebar.divider()
st.sidebar.caption(
    "Implementation: [Kirchenbauer et al. 2023](https://arxiv.org/abs/2301.10226), "
    "[lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking) (Apache 2.0). "
    f"Model: {MODEL_NAME}, CPU, fully local."
)

st.title("KGW watermark playground")
st.caption(
    "Generate text with and without a statistical watermark, then detect it. "
    "The scheme pseudorandomly favours a 'green list' of tokens at each step; "
    "the detector counts green tokens and computes how unlikely that count is by chance."
)

tab_gen, tab_detect, tab_robust = st.tabs(["Generate & compare", "Detect", "Robustness"])

# ---------------------------------------------------------------- generate

with tab_gen:
    prompt = st.text_area(
        "Prompt",
        "The history of watermarking goes back centuries. Papermakers in "
        "thirteenth-century Italy pressed wire designs into wet pulp so that",
        height=90,
    )
    if st.button("Generate both", type="primary"):
        with st.spinner("Generating without watermark..."):
            plain = generate(prompt, False, gamma, delta, scheme, temperature, top_p, max_new, seed)
        with st.spinner("Generating with watermark..."):
            marked = generate(prompt, True, gamma, delta, scheme, temperature, top_p, max_new, seed)
        st.session_state["plain"] = plain
        st.session_state["marked"] = marked

    if "marked" in st.session_state:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Without watermark")
            st.write(st.session_state["plain"])
            score_metrics(
                detect(st.session_state["plain"], gamma, scheme, z_threshold, ignore_repeats),
                z_threshold,
            )
        with col_b:
            st.subheader("With watermark")
            st.write(st.session_state["marked"])
            score_metrics(
                detect(st.session_state["marked"], gamma, scheme, z_threshold, ignore_repeats),
                z_threshold,
            )

# ---------------------------------------------------------------- detect

with tab_detect:
    default_text = st.session_state.get("marked", "")
    text_in = st.text_area(
        "Text to score (paste anything, or generate first and it lands here)",
        default_text,
        height=160,
    )
    if st.button("Detect") and text_in.strip():
        score = detect(text_in, gamma, scheme, z_threshold, ignore_repeats, with_mask=True)
        score_metrics(score, z_threshold)
        st.markdown(token_highlight_html(text_in, score), unsafe_allow_html=True)
        st.caption(
            f"An unwatermarked text should sit near a green fraction of gamma = {gamma}. "
            "Detection only works with the same gamma and seeding scheme used at generation time."
        )

# ---------------------------------------------------------------- robustness

with tab_robust:
    st.caption(
        "How well does the watermark survive editing? Truncate the watermarked text, "
        "or delete random tokens, and watch the z-score."
    )
    source = st.session_state.get("marked", "")
    if not source:
        st.info("Generate a watermarked text first (Generate & compare tab).")
    elif st.button("Run robustness sweeps"):
        tokenizer, _ = load_model()
        ids = tokenizer(source, add_special_tokens=False)["input_ids"]

        rows = []
        for n in range(20, len(ids) + 1, max(10, len(ids) // 10)):
            s = detect(tokenizer.decode(ids[:n]), gamma, scheme, z_threshold, ignore_repeats)
            rows.append({"n": n, "z": s["z_score"]})
        trunc_df = pd.DataFrame(rows)

        rng = torch.Generator().manual_seed(0)
        rows = []
        for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            zs = []
            for _ in range(5):
                keep = torch.rand(len(ids), generator=rng) > frac
                kept = [t for t, k in zip(ids, keep.tolist()) if k]
                zs.append(detect(tokenizer.decode(kept), gamma, scheme, z_threshold, ignore_repeats)["z_score"])
            rows.append({"frac": int(frac * 100), "z": sum(zs) / len(zs)})
        del_df = pd.DataFrame(rows)

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Truncation")
            st.altair_chart(
                z_line_chart(trunc_df, "n:Q", "Tokens kept (from start)", z_threshold),
                width='stretch',
            )
        with col_r:
            st.subheader("Random token deletion")
            st.altair_chart(
                z_line_chart(del_df, "frac:Q", "% of tokens deleted (mean of 5 trials)", z_threshold),
                width='stretch',
            )
        with st.expander("Data table"):
            st.dataframe(trunc_df, hide_index=True)
            st.dataframe(del_df, hide_index=True)
