"""KGW watermark playground.

Streamlit app for generating and detecting statistically watermarked LLM text,
using the reference implementation from Kirchenbauer et al. (2023),
"A Watermark for Large Language Models" (arXiv:2301.10226).

Runs fully locally on facebook/opt-125m. No APIs.
"""

import math
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
SAMPLES_DIR = Path(__file__).parent / "samples"

PAPER_MAIN = "https://arxiv.org/abs/2301.10226"
PAPER_RELIABILITY = "https://arxiv.org/abs/2306.04634"
UPSTREAM_REPO = "https://github.com/jwkirchenbauer/lm-watermarking"
APP_REPO = "https://github.com/searchsolved/kgw-watermark-playground"
AUTHOR_URL = "https://leefoot.com"

# Chart / highlight colours (validated reference palette from the dataviz method)
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
GREEN = "#008300"
TEXT_SECONDARY = "#52514e"

st.set_page_config(
    page_title="KGW Watermark Playground",
    page_icon="🖋️",
    layout="wide",
    menu_items={
        "Get help": APP_REPO,
        "Report a bug": APP_REPO + "/issues",
        "About": (
            "Interactive demo of the Kirchenbauer et al. LLM watermarking scheme. "
            f"Built by [Lee Foot]({AUTHOR_URL}) on the authors' "
            f"[reference implementation]({UPSTREAM_REPO})."
        ),
    },
)


@st.cache_resource(show_spinner="Loading facebook/opt-125m (first run only)...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return tokenizer, model


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


@st.cache_data(show_spinner=False)
def detect(text, gamma, scheme, z_threshold, ignore_repeats, window_size=None):
    tokenizer, _ = load_model()
    detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=gamma,
        seeding_scheme=scheme,
        device=DEVICE,
        tokenizer=tokenizer,
        z_threshold=z_threshold,
        normalizers=[],
        ignore_repeated_ngrams=ignore_repeats,
    )
    if window_size is not None:
        score = detector.detect(text, window_size=window_size)
        score["z_score"] = float(score["z_score"])
        return score
    return detector.detect(text, return_green_token_mask=True)


def odds_phrase(p):
    if p >= 0.01:
        return f"about 1 in {1 / p:.0f}"
    if p < 1e-300:
        return "smaller than 1 in 10^300"
    return f"about 1 in 10^{int(math.floor(-math.log10(p)))}"


def verdict_banner(score, z_threshold):
    if score["prediction"]:
        st.success(
            f"Watermark detected. The chance of an unwatermarked text scoring this high "
            f"is {odds_phrase(score['p_value'])}."
        )
    else:
        st.info(f"No watermark detected (z below the {z_threshold} threshold).")


def score_metrics(score, gamma):
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "z-score",
        f"{score['z_score']:.2f}",
        help="How many standard deviations the green token count sits above chance. "
        "Higher = stronger evidence of the watermark.",
    )
    c2.metric(
        "Green fraction",
        f"{score['green_fraction']:.3f}",
        help=f"Share of tokens that landed on their step's green list. "
        f"Unwatermarked text should sit near gamma = {gamma}.",
    )
    c3.metric(
        "Tokens scored",
        score["num_tokens_scored"],
        help="Unique token contexts contributing evidence. Detection strength grows "
        "with length, so short texts are the hard case.",
    )


def highlight_legend():
    st.markdown(
        f'<p style="font-size:0.85em;color:{TEXT_SECONDARY}">'
        f'<span style="background:{GREEN}22;border-bottom:2px solid {GREEN}">underlined</span>'
        " = token was on its step's green list · plain = red list · "
        "grey = leading context, not scored</p>",
        unsafe_allow_html=True,
    )


def token_highlight_html(text, score):
    """Tokens with green-list hits get a tint AND an underline (never colour alone)."""
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
    return f'<div style="line-height:1.9">{"".join(spans)}</div>'


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


def dilution_chart(df, z_threshold):
    base = alt.Chart(df).encode(
        x=alt.X(
            "share:Q",
            title="Watermarked share of the document (%)",
            scale=alt.Scale(reverse=True),
        ),
        y=alt.Y("z:Q", title="Detection z-score"),
        color=alt.Color(
            "method:N",
            title=None,
            scale=alt.Scale(
                domain=["Whole document", "Windowed (WinMax)"],
                range=[SERIES_BLUE, SERIES_ORANGE],
            ),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[
            alt.Tooltip("method:N", title="Method"),
            alt.Tooltip("share:Q", title="Watermarked share %"),
            alt.Tooltip("z:Q", format=".2f"),
        ],
    )
    line = base.mark_line(strokeWidth=2) + base.mark_point(size=70, filled=True)
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
    return (line + rule + label).properties(height=300)


# ---------------------------------------------------------------- sidebar

st.sidebar.title("Settings")
st.sidebar.subheader("Watermark")
gamma = st.sidebar.slider(
    "gamma (green list fraction)",
    0.1,
    0.5,
    0.25,
    0.05,
    help="Fraction of the vocabulary placed on the green list at each step. "
    "Detection compares the observed green fraction against this chance baseline.",
)
delta = st.sidebar.slider(
    "delta (green logit boost)",
    0.5,
    10.0,
    2.0,
    0.5,
    help="How hard generation is pushed toward green tokens. Higher = stronger, "
    "easier-to-detect watermark, but push it far enough and text quality degrades. "
    "Try 8+ to see the trade-off.",
)
scheme = st.sidebar.selectbox(
    "Seeding scheme",
    ["selfhash", "simple_1", "minhash"],
    index=0,
    help="How each step's green list is seeded from the preceding tokens. "
    "The detector must use the same scheme (and gamma) as the generator. "
    "selfhash is the papers' recommendation.",
)
z_threshold = st.sidebar.slider(
    "Detection z threshold",
    2.0,
    6.0,
    4.0,
    0.5,
    help="Significance bar for calling a text watermarked. z = 4 corresponds to a "
    "false positive rate of roughly 3 in 100,000.",
)
ignore_repeats = st.sidebar.checkbox(
    "Ignore repeated ngrams when scoring",
    value=True,
    help="Score each unique token context once, so repetitive text cannot inflate "
    "the evidence. Recommended by the reliability paper.",
)
st.sidebar.subheader("Generation")
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
top_p = st.sidebar.slider("top_p", 0.5, 1.0, 0.95, 0.05)
max_new = st.sidebar.slider(
    "Max new tokens",
    50,
    400,
    200,
    25,
    help="More tokens = slower generation but stronger detection evidence.",
)
seed = st.sidebar.number_input("Sampling seed", value=42, step=1)
st.sidebar.divider()
st.sidebar.caption(
    f"Papers: [A Watermark for Large Language Models]({PAPER_MAIN}) · "
    f"[On the Reliability of Watermarks]({PAPER_RELIABILITY})\n\n"
    f"Built by [Lee Foot]({AUTHOR_URL}) on the authors' "
    f"[reference implementation]({UPSTREAM_REPO}) (Apache 2.0). "
    f"[App source]({APP_REPO}). Model: {MODEL_NAME}, CPU, fully local, no APIs."
)

# ---------------------------------------------------------------- header

st.title("KGW watermark playground")
st.caption(
    "Statistical text watermarking, live: generate watermarked text, see exactly "
    "which tokens carry the signal, and stress-test how much editing it survives."
)

with st.expander("How the watermark works (60-second version)"):
    st.markdown(
        f"""
A language model picks each next token by scoring its whole vocabulary. The
watermark from [Kirchenbauer et al. 2023]({PAPER_MAIN}) sits at that layer:

1. At each step, a hash of the preceding tokens seeds a pseudorandom split of the
   vocabulary into a **green list** (a fraction *gamma*, default 25%) and a red list.
2. Every green token gets a small boost (*delta*) to its score, so generation
   **leans** toward green tokens. Nothing is forced, and the text reads normally.
3. Anyone who knows the seeding scheme can replay the split for every position in
   any text and **count the green hits**. No model needed, just the tokenizer and the key.
4. Unwatermarked text lands near *gamma* by chance. Watermarked text lands far above.
   The gap becomes a **z-score** with a calculable false positive rate, which makes
   this a hypothesis test rather than an AI-detector-style guess.

The follow-up paper, [On the Reliability of Watermarks]({PAPER_RELIABILITY}),
shows the signal survives substantial editing and paraphrasing, which you can
verify yourself in the **Robustness** tab. Implementation:
[jwkirchenbauer/lm-watermarking]({UPSTREAM_REPO}).
"""
    )

tab_detect, tab_gen, tab_robust = st.tabs(
    ["1 · Detect", "2 · Generate & compare", "3 · Attack the watermark"]
)

# ---------------------------------------------------------------- generate

with tab_gen:
    st.markdown(
        "The watermark can only be embedded **while text is being generated**, so this "
        "tab is the watermark factory. It generates the same continuation twice, once "
        "normally and once with the watermark, then compares. Green highlighting shows "
        "which tokens landed on their step's green list. Your watermarked output feeds "
        "tabs 1 and 3."
    )
    prompt = st.text_area(
        "Prompt",
        "The history of watermarking goes back centuries. Papermakers in "
        "thirteenth-century Italy pressed wire designs into wet pulp so that",
        height=90,
    )
    if st.button("Generate both versions", type="primary"):
        with st.status("Generating on CPU, roughly a minute for both...", expanded=False) as status:
            st.write("Generating without watermark...")
            plain = generate(prompt, False, gamma, delta, scheme, temperature, top_p, max_new, seed)
            st.write("Generating with watermark...")
            marked = generate(prompt, True, gamma, delta, scheme, temperature, top_p, max_new, seed)
            status.update(label="Done", state="complete")
        st.session_state["plain"] = plain
        st.session_state["marked"] = marked
        st.session_state["gen_settings"] = (gamma, delta, scheme)

    if "marked" in st.session_state:
        if st.session_state.get("gen_settings", (gamma, delta, scheme))[::2] != (gamma, scheme):
            st.warning(
                "The watermark settings in the sidebar have changed since these texts "
                "were generated, so the detector is now checking the wrong green lists. "
                "Regenerate to line them back up (or change the settings back). "
                "This mismatch is exactly why third parties cannot detect a watermark "
                "without the key."
            )
        highlight_legend()
        col_a, col_b = st.columns(2)
        for col, title, key in (
            (col_a, "Without watermark", "plain"),
            (col_b, "With watermark", "marked"),
        ):
            with col:
                st.subheader(title)
                score = detect(st.session_state[key], gamma, scheme, z_threshold, ignore_repeats)
                verdict_banner(score, z_threshold)
                score_metrics(score, gamma)
                with st.container(height=320, border=True):
                    st.markdown(
                        token_highlight_html(st.session_state[key], score),
                        unsafe_allow_html=True,
                    )
        marked_score = detect(st.session_state["marked"], gamma, scheme, z_threshold, ignore_repeats)
        if marked_score["prediction"]:
            st.caption(
                f"The watermarked output put {marked_score['green_fraction']:.0%} of its "
                f"tokens on green lists against {gamma:.0%} expected by chance. The odds "
                f"of that happening without the watermark are "
                f"{odds_phrase(marked_score['p_value'])}."
            )

# ---------------------------------------------------------------- detect

with tab_detect:
    st.markdown(
        "Score any text against the current watermark settings. Start with the bundled "
        "watermarked sample for an instant positive, then try your own writing and watch "
        "it score at exactly the chance rate: the detector has a calculable false positive "
        "rate and essentially never accuses a human.\n\n"
        "One thing this is **not**: an AI detector. Text pasted from ChatGPT, Claude or "
        "Gemini will correctly come back *not detected*, because their outputs are not "
        "watermarked with this scheme's key. Detection is a key check. To see a positive, "
        "the watermark must be embedded at generation time (tab 2)."
    )
    b1, b2, b3 = st.columns(3)
    if b1.button("Load watermarked sample"):
        st.session_state["detect_text"] = (SAMPLES_DIR / "watermarked_default_settings.txt").read_text()
    if b2.button("Load human-written sample"):
        st.session_state["detect_text"] = (SAMPLES_DIR / "human_control.txt").read_text()
    if b3.button("Use my last generation", disabled="marked" not in st.session_state):
        st.session_state["detect_text"] = st.session_state["marked"]
    text_in = st.text_area("Text to score", height=170, key="detect_text")
    if text_in and text_in.strip():
        score = detect(text_in, gamma, scheme, z_threshold, ignore_repeats)
        verdict_banner(score, z_threshold)
        score_metrics(score, gamma)
        highlight_legend()
        st.markdown(token_highlight_html(text_in, score), unsafe_allow_html=True)
        st.caption(
            "The bundled watermarked sample was generated at the default settings "
            "(gamma 0.25, delta 2.0, selfhash). Change gamma or the seeding scheme in "
            "the sidebar and watch detection collapse: without the exact key there is "
            "nothing to test. Editing the text above and re-scoring is a hands-on "
            "robustness experiment."
        )
    else:
        st.info(
            "Load a sample above, or paste anything: your own writing scores near "
            "chance (that is the false-positive story), and chatbot output scores "
            "near chance too, because nobody's public model carries this key."
        )

# ---------------------------------------------------------------- robustness

with tab_robust:
    st.markdown(
        "How much damage does the watermark survive? These sweeps take a watermarked "
        "text, attack it progressively, and re-run detection at each step. The "
        f"[reliability paper]({PAPER_RELIABILITY}) studies exactly this.\n\n"
        "The strongest attack in that paper is **paraphrasing with another LLM**, and "
        "you can run it yourself right now: copy the watermarked text below, ask any "
        "chatbot to paraphrase it, and paste the result into the Detect tab. Per-token "
        "evidence weakens, but with enough tokens the signal often still accumulates."
    )
    if "marked" in st.session_state:
        source = st.session_state["marked"]
        st.caption("Using your last watermarked generation.")
    else:
        source = (SAMPLES_DIR / "watermarked_default_settings.txt").read_text()
        st.caption(
            "Using the bundled watermarked sample (generated at default settings). "
            "Generate your own in tab 2 to sweep that instead."
        )
    with st.expander("Show the text under attack"):
        st.code(source, language=None, wrap_lines=True)
    if st.button("Run robustness sweeps", type="primary"):
        tokenizer, _ = load_model()
        ids = tokenizer(source, add_special_tokens=False)["input_ids"]

        with st.status("Running truncation, deletion, replacement and dilution sweeps...", expanded=False):
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

            rows = []
            vocab_size = len(tokenizer)
            for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                zs = []
                for _ in range(5):
                    swap = torch.rand(len(ids), generator=rng) < frac
                    rand_ids = torch.randint(0, vocab_size, (len(ids),), generator=rng)
                    attacked = [int(r) if m else t for t, m, r in zip(ids, swap.tolist(), rand_ids.tolist())]
                    zs.append(detect(tokenizer.decode(attacked), gamma, scheme, z_threshold, ignore_repeats)["z_score"])
                rows.append({"frac": int(frac * 100), "z": sum(zs) / len(zs)})
            repl_df = pd.DataFrame(rows)

            insert_ids = ids[: min(60, len(ids))]
            insert_text = tokenizer.decode(insert_ids)
            filler_ids = tokenizer(
                (SAMPLES_DIR / "unwatermarked_filler.txt").read_text(), add_special_tokens=False
            )["input_ids"]
            rows = []
            for n_fill in [0, 60, 130, 260, 390, 520]:
                pre = tokenizer.decode(filler_ids[: n_fill // 2])
                post = tokenizer.decode(filler_ids[n_fill // 2 : n_fill])
                doc = (pre + " " + insert_text + " " + post).strip()
                share = round(100 * len(insert_ids) / (len(insert_ids) + n_fill))
                z_full = detect(doc, gamma, scheme, z_threshold, ignore_repeats)["z_score"]
                z_win = detect(doc, gamma, scheme, z_threshold, ignore_repeats, window_size="max")["z_score"]
                rows.append({"share": share, "z": z_full, "method": "Whole document"})
                rows.append({"share": share, "z": z_win, "method": "Windowed (WinMax)"})
            dil_df = pd.DataFrame(rows)

        st.session_state["sweeps"] = {
            "trunc": trunc_df, "del": del_df, "repl": repl_df, "dil": dil_df
        }

    if "sweeps" in st.session_state:
        sw = st.session_state["sweeps"]
        col_l, col_m, col_r = st.columns(3)
        with col_l:
            st.subheader("Truncation")
            st.caption("Keep only the first N tokens. How short can a quote get and still convict?")
            st.altair_chart(
                z_line_chart(sw["trunc"], "n:Q", "Tokens kept (from start)", z_threshold),
                width="stretch",
            )
        with col_m:
            st.subheader("Random deletion")
            st.caption("Delete a growing share of tokens at random (mean of 5 trials per point).")
            st.altair_chart(
                z_line_chart(sw["del"], "frac:Q", "% of tokens deleted", z_threshold),
                width="stretch",
            )
        with col_r:
            st.subheader("Random replacement")
            st.caption(
                "Swap tokens for random ones. Each swap also corrupts the neighbouring "
                "contexts, so this bites harder than deletion."
            )
            st.altair_chart(
                z_line_chart(sw["repl"], "frac:Q", "% of tokens replaced", z_threshold),
                width="stretch",
            )

        st.subheader("Dilution: hiding a watermarked quote in unwatermarked text")
        st.markdown(
            "The realistic case is not a fully watermarked document but a **watermarked "
            "passage inside ordinary text**, such as one generated paragraph in an "
            "otherwise self-written article. Here a 60-token watermarked snippet is "
            "buried in growing amounts of unwatermarked prose. Scoring the whole "
            "document dilutes the signal below the threshold, but a **sliding-window "
            "scan (WinMax, from the reliability paper)** hunts for the hottest span "
            "and keeps convicting."
        )
        st.altair_chart(dilution_chart(sw["dil"], z_threshold), width="stretch")

        with st.expander("Data tables"):
            st.dataframe(sw["trunc"], hide_index=True)
            st.dataframe(sw["del"], hide_index=True)
            st.dataframe(sw["repl"], hide_index=True)
            st.dataframe(sw["dil"], hide_index=True)

    st.divider()
    st.subheader("Paraphrase attack (bring your own chatbot)")
    st.markdown(
        "The strongest known attack needs another LLM. Copy the text under attack "
        "above, ask any chatbot to paraphrase it, and paste the result here."
    )
    para = st.text_area("Paraphrased version", height=140, key="para_text")
    if para and para.strip():
        orig_score = detect(source, gamma, scheme, z_threshold, ignore_repeats)
        para_score = detect(para, gamma, scheme, z_threshold, ignore_repeats)
        col_o, col_p = st.columns(2)
        with col_o:
            st.markdown("**Original watermarked text**")
            verdict_banner(orig_score, z_threshold)
            score_metrics(orig_score, gamma)
        with col_p:
            st.markdown("**Your paraphrase**")
            verdict_banner(para_score, z_threshold)
            score_metrics(para_score, gamma)
        if orig_score["z_score"] > 0:
            retained = max(0.0, para_score["z_score"] / orig_score["z_score"] * 100)
            if para_score["prediction"]:
                st.caption(
                    f"The paraphrase retains {retained:.0f}% of the original z-score and "
                    "is still detected. Per-token evidence weakened, but enough survived."
                )
            else:
                st.caption(
                    f"The paraphrase retains {retained:.0f}% of the original z-score and "
                    "drops below the threshold at this length. The reliability paper's "
                    "finding is that evidence re-accumulates with length, so longer "
                    "paraphrased texts become detectable again."
                )

st.divider()
st.caption(
    f"Papers: [Kirchenbauer et al. 2023]({PAPER_MAIN}) and the "
    f"[reliability follow-up]({PAPER_RELIABILITY}) · "
    f"[Reference implementation]({UPSTREAM_REPO}) · [App source]({APP_REPO}) · "
    f"Built by [Lee Foot]({AUTHOR_URL})."
)
