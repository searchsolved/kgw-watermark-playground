from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=900)
at.run()
assert not at.exception, at.exception

# shrink generation for test speed: sidebar sliders
# 0 delta, 1 max_new, then advanced: 2 gamma, 3 z_threshold, 4 temperature, 5 top_p
at.sidebar.slider[1].set_value(80)
at.run()

# --- tab 2: bundled sample detects with no generation needed
buttons = {b.label: b for b in at.button}
buttons["Load watermarked sample"].click()
at.run()
assert not at.exception, at.exception
assert any("Watermark detected" in str(s.value) for s in at.success), "sample not detected"
html_blocks = [el.value for el in at.markdown if "border-bottom" in str(el.value)]
assert html_blocks, "token highlight html missing"
print("sample detection ok, highlight rendered")

buttons = {b.label: b for b in at.button}
buttons["Load human-written sample"].click()
at.run()
assert not at.exception, at.exception
assert any("No watermark" in str(i.value) for i in at.info), "human sample should not detect"
print("human control not detected, correct")

# --- tab 1: generate both and compare
buttons = {b.label: b for b in at.button}
buttons["Generate both versions"].click()
at.run()
assert not at.exception, at.exception
assert "marked" in at.session_state and len(at.session_state["marked"]) > 50
zlabels = [m.label for m in at.metric]
assert zlabels.count("z-score") >= 2, zlabels
print("compare metrics rendered:", [(m.label, m.value) for m in at.metric[:3]])

# anti-repetition params: no verbatim loops, watermark still detected
assert any("Watermark detected" in str(s.value) for s in at.success), "generated watermark not detected"
marked = at.session_state["marked"]
frags = [marked[i : i + 20] for i in range(0, len(marked) - 40, 40)]
assert len(set(frags)) > len(frags) * 0.6, "generation still looping"
compare_html = "".join(str(el.value) for el in at.markdown if "line-height:1.9" in str(el.value))
assert "�" not in compare_html, "mojibake in compare highlight"
print("generation quality ok: no loops, no mojibake, still detected")

# highlighter handles multi-byte characters spanning several BPE tokens
at.text_area(key="detect_text").set_value("He said “watermarks” are naïve… 水印 \U0001f58b️ test")
buttons = {b.label: b for b in at.button}
buttons["Score this text"].click()
at.run()
assert not at.exception, at.exception
uni_html = [str(el.value) for el in at.markdown if "line-height:1.9" in str(el.value)]
assert uni_html and "�" not in uni_html[-1], "mojibake in unicode highlight"
print("unicode highlight decode ok")

# settings-mismatch warning appears when gamma changes after generation
at.sidebar.slider[2].set_value(0.4)
at.run()
assert any("settings" in str(w.value).lower() for w in at.warning), "mismatch warning missing"
at.sidebar.slider[2].set_value(0.25)
at.run()

# --- tab 3: robustness sweeps incl. dilution
buttons = {b.label: b for b in at.button}
buttons["Attack this text"].click()
at.run()
assert not at.exception, at.exception
assert len(at.dataframe) == 4
dil = at.session_state["sweeps"]["dil"]
win = dil[dil.method.str.startswith("Window")]
full = dil[dil.method == "Whole document"]
assert win.z.min() > 4, f"WinMax should stay above threshold, got {win.z.min():.2f}"
assert full.z.min() < 4, f"whole-doc z should dilute below threshold, got {full.z.min():.2f}"
print(f"dilution ok: whole-doc z falls to {full.z.min():.2f}, WinMax holds at {win.z.min():.2f}")

# --- paraphrase before/after box scores both sides
metrics_before = len(at.metric)
at.text_area(key="para_text").set_value(open("samples/human_control.txt").read())
buttons = {b.label: b for b in at.button}
buttons["Score my paraphrase"].click()
at.run()
assert not at.exception, at.exception
assert len(at.metric) >= metrics_before + 6, "paraphrase comparison metrics missing"
print("paraphrase comparison rendered")
print("ALL APP TESTS PASSED")
