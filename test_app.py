from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=900)
at.run()
assert not at.exception, at.exception

# shrink generation for test speed: sidebar sliders
# 0 gamma, 1 delta, 2 z_threshold, 3 temperature, 4 top_p, 5 max_new
at.sidebar.slider[5].set_value(80)
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

# settings-mismatch warning appears when gamma changes after generation
at.sidebar.slider[0].set_value(0.4)
at.run()
assert any("settings" in str(w.value).lower() for w in at.warning), "mismatch warning missing"
at.sidebar.slider[0].set_value(0.25)
at.run()

# --- tab 3: robustness sweeps incl. dilution
buttons = {b.label: b for b in at.button}
buttons["Run robustness sweeps"].click()
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
at.run()
assert not at.exception, at.exception
assert len(at.metric) >= metrics_before + 6, "paraphrase comparison metrics missing"
print("paraphrase comparison rendered")
print("ALL APP TESTS PASSED")
